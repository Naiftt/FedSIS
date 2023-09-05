import torch 
import torch.nn as nn
import timm
import copy 
import random 
import os 
import pickle as pkl
import numpy as np 
from hter_metrics_compute import main_metrics
from sklearn.metrics import f1_score, balanced_accuracy_score
from matplotlib.backends.backend_pdf import PdfPages
from utils import weight_vec
import sys
seed = 105
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


class MLP_cls_new(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.norm = nn.LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        self.identity = nn.Identity()
        self.fc = nn.Linear(in_features=768, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.norm(x)
        x = self.identity(x)
        x = self.fc(x)
        return x 

class ResidualBlock(nn.Module):
    def __init__(self, in_channels=768, out_channels=768, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        self.pool = nn.AvgPool2d(14, stride=1)

    def forward(self, x):
        if len(x.shape) == 3: 
            x = torch.permute(x,(0,-1,1))
            x = x.reshape(x.shape[0], x.shape[1] , 14, 14)
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.pool(out)
        return out.reshape(-1,768)

class FEDSIS(nn.Module): 
    def __init__(
        self, ViT_name, num_classes, drop_out_mlp_server, std, mean, diff_privacy, embedding=768, 
        num_clients=6, in_channels=3, ViT_pretrained=False,
        initial_block=6, final_block=12
        ) -> None:
        super().__init__()
        self.initial_block = initial_block
        self.final_block = final_block
        self.drop_out_mlp_server = drop_out_mlp_server
        self.diff_privacy = diff_privacy
        if diff_privacy:
            self.std = std
            self.mean = mean
        self.vit = timm.create_model(
            model_name = ViT_name,
            pretrained = ViT_pretrained,
            num_classes = num_classes,
            in_chans = in_channels
        )   

        self.resnet50 = self.vit.patch_embed
        
        self.resnet50_clients = nn.ModuleList([copy.deepcopy(self.resnet50) for i in range(num_clients)])
        self.common_network = ResidualBlock()
        client_tail = MLP_cls_new(num_classes= num_classes)
        self.mlp_clients_tail =  nn.ModuleList([copy.deepcopy(client_tail) for i in range(num_clients)])

        self.real_features_client0 = None
        self.feature_sent_client = None

    def forward(self, x, chosen_block, client_idx):
        x = self.resnet50_clients[client_idx](x)

        if self.diff_privacy == True:
            noise = torch.randn(size= x.shape).cuda() * self.std + self.mean
            x = x + noise

        for block_num in range(chosen_block):
            x = self.vit.blocks[block_num](x)
        for block_num in range(12):
            constant_output = self.vit.blocks[block_num](x)
        y = self.common_network(constant_output)
        x = self.mlp_clients_tail[client_idx](y)
        return x,y


class FedNetwork():
    def __init__(
        self, num_clients, device, network, 
        criterion, network_name, base_dir, 
        avg_body = False
        ):
        """
        args:
            num_clients
            device: cuda vs cpu
            network: ViT model
            criterion: loss function to be used
            network_name: used for saving purposes 
            base_dir: where to save pickles/model files
        """
        
        self.device = device
        self.num_clients = num_clients
        self.criterion = criterion
        self.network = network
        self.network_name = network_name
        self.base_dir = base_dir
        self.avg_body = avg_body
        # save initial model 
        # torch.save(self.network, os.path.join(base_dir, 'initial_model.pt'))

            

    def init_logs(self):
        """
        This method initializes dictionaries for the metrics

        """
        self.f1s_macro = {'train':[[] for i in range(self.num_clients)], 'test':[[] for i in range(self.num_clients)]}
        self.f1s_weighted = {'train':[[] for i in range(self.num_clients)], 'test':[[] for i in range(self.num_clients)]}
        self.losses  = {'train':[[] for i in range(self.num_clients)], 'test':[[] for i in range(self.num_clients)]}
        self.rocs = {'train':[[] for i in range(self.num_clients)], 'test':[[] for i in range(self.num_clients)]}
        self.balanced_accs = {'train':[[] for i in range(self.num_clients)], 'test':[[] for i in range(self.num_clients)]}
        self.whole_probs_client = {f"client_{i}":{} for i in range(self.num_clients)}
        self.whole_labels_client = {f"client_{i}":{} for i in range(self.num_clients)}
        self.whole_videoid_client = {f"client_{i}":{} for i in range(self.num_clients)}
        self.max_acc = 0
        self.min_hter = 100000
        self.hters = {'test':[[] for i in range(self.num_clients)]}
        self.aucs  = {'test':[[] for i in range(self.num_clients)]}
        self.rates = {'test':[[] for i in range(self.num_clients)]}

    def set_optimizer(self, name, lr, weight_decay):
        """
        name: Optimizer name, e.g. Adam 
        lr: learning rate 

        """
        if name == 'Adam':
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr = lr, weight_decay = weight_decay)


    def train_round(self, client_i):
        """
        Training loop. 

        client_i: Client index.

        """
        running_loss_client_i = 0
        whole_labels = []
        whole_preds = []
        whole_probs = []
        copy_network = copy.deepcopy(self.network)
        weight_dic = {'blocks':None, 'cls':None, 'pos_embed':None}
        for data in self.CLIENTS_DATALOADERS[client_i]: 
            self.optimizer.zero_grad()
            imgs, labels = data[0].to(device), data[1].to(device)
            labels = labels.reshape(labels.shape[0])
            tail_output = self.network(imgs, client_i)
            loss = self.criterion(tail_output[0], labels)
            loss.backward()
            self.optimizer.step()
            running_loss_client_i+= loss.item() 
            _, predicted = torch.max(tail_output[0], 1)
            whole_probs.append(torch.nn.Softmax(dim = -1)(tail_output[0]).detach().cpu())
            whole_labels.append(labels.detach().cpu())
            whole_preds.append(predicted.detach().cpu()) 
        self.metrics(client_i, whole_labels, whole_preds, running_loss_client_i, len(self.CLIENTS_DATALOADERS[client_i]), whole_probs, train = True)
        if self.avg_body:
            weight_dic['blocks'] = weight_vec(self.network.vit.blocks).detach().cpu()
            weight_dic['cls'] = self.network.vit.cls_token.detach().cpu()
            weight_dic['pos_embed'] = self.network.vit.pos_embed.detach().cpu()

            self.network.vit.blocks = copy.deepcopy(copy_network.vit.blocks)
            self.network.vit.cls_token = copy.deepcopy(copy_network.vit.cls_token)
            self.network.vit.pos_embed =  copy.deepcopy(copy_network.vit.pos_embed)
            return weight_dic
          
    def eval_round(self, client_i):
        """
        Evaluation loop. 

        client_i: Client index.
                
        """
        running_loss_client_i = 0
        whole_labels = []
        whole_preds = []
        whole_probs = []
        with torch.no_grad():
            for data in self.testloader: 
                imgs, labels = data[0].to(device), data[1].to(device)
                tail_output = self.network(imgs, client_i)[0]
                loss = self.criterion(tail_output, labels)
                running_loss_client_i+= loss.item() 
                _, predicted = torch.max(tail_output, 1)
                whole_probs.append(torch.nn.Softmax(dim = -1)(tail_output).detach().cpu())
                whole_labels.append(labels.detach().cpu())
                whole_preds.append(predicted.detach().cpu())    
            self.metrics(client_i, whole_labels, whole_preds, running_loss_client_i, len(self.testloader), whole_probs, train= False)

    def metrics(self, client_i, whole_labels, whole_preds, whole_video_id,  running_loss_client_i, len_loader, whole_probs, train, r):
        """
        Save metrics as pickle files and the model as .pt file.
        
        """
        whole_labels = torch.cat(whole_labels)
        whole_preds = torch.cat(whole_preds)
        whole_probs = torch.cat(whole_probs)
        if train == False:
            whole_video_id = torch.cat(whole_video_id).detach().cpu().numpy()
            HTER, AUC, TPR = main_metrics(whole_probs.detach().cpu().numpy(), whole_labels.detach().cpu().numpy(), whole_video_id)
        f1_train_macro = f1_score(whole_labels.detach().cpu(), whole_preds.detach().cpu(), average = 'macro')
        f1_train_weighted = f1_score(whole_labels.detach().cpu(), whole_preds.detach().cpu(), average = 'weighted')
        loss_epoch = running_loss_client_i/len_loader
        balanced_acc = balanced_accuracy_score(whole_labels.detach().cpu(), whole_preds.detach().cpu())
        if train == True:
            eval_name = 'train'
        else:
            eval_name = 'test'
            if balanced_acc > self.max_acc:
                # torch.save(self.network.state_dict(), os.path.join(self.base_dir, "best_model.pt"))
                self.max_acc = balanced_acc
            if HTER < self.min_hter:
                torch.save(self.network, os.path.join(self.base_dir, "best_model_HTER.pt"))
                self.min_hter = HTER
        self.f1s_macro[eval_name][client_i].append(f1_train_macro)
        self.f1s_weighted[eval_name][client_i].append(f1_train_weighted)
        self.losses[eval_name][client_i].append(loss_epoch)
        self.balanced_accs[eval_name][client_i].append(balanced_acc)
        # saving the preicted outcomes for each client
        if train == False:
            self.whole_probs_client[f"client_{client_i}"][r] = whole_probs
            self.whole_labels_client[f"client_{client_i}"][r] = whole_labels
        if train == False:
            self.whole_videoid_client[f"client_{client_i}"][r] = whole_video_id
        # print(f"client{client_i}_{eval_name}:")
        print(f"f1 (macro): {f1_train_macro:.3f}")
        print(f"f1 (weighted): {f1_train_weighted:.3f}")
        print(f"Loss: {loss_epoch:.3f}")
        print(f"balanced accuracy:{balanced_acc:.3f}")
        if train == False:
            self.hters[eval_name][client_i].append(HTER)
            self.aucs[eval_name][client_i].append(AUC)
            self.rates[eval_name][client_i].append(TPR)
            print(f"HTER:{HTER:3f}, AUC:{AUC:3f}, TPR:{TPR:3f}")
        print("-----------------")
    def save_pickles(self, base_dir): 
        with open(os.path.join(base_dir,'f1s_macro'), 'wb') as handle:
            pkl.dump(self.f1s_macro, handle)
        with open(os.path.join(base_dir,'f1s_weighted'), 'wb') as handle:
            pkl.dump(self.f1s_weighted, handle)
        with open(os.path.join(base_dir,'loss_epoch'), 'wb') as handle:
            pkl.dump(self.losses, handle)
        with open(os.path.join(base_dir,'rocs'), 'wb') as handle:
            pkl.dump(self.rocs, handle)
        with open(os.path.join(base_dir,'balanced_accs'), 'wb') as handle:
            pkl.dump(self.balanced_accs, handle)
        with open(os.path.join(base_dir,'predictions'), 'wb') as handle:
            pkl.dump(self.whole_probs_client, handle)
        with open(os.path.join(base_dir,'labels'), 'wb') as handle:
            pkl.dump(self.whole_labels_client, handle)
        with open(os.path.join(base_dir,'videoid'), 'wb') as handle:
            pkl.dump(self.whole_videoid_client, handle)
        with open(os.path.join(base_dir,'HTER'), 'wb') as handle:
            pkl.dump(self.hters, handle)
        with open(os.path.join(base_dir,'AUC'), 'wb') as handle:
            pkl.dump(self.aucs, handle)
        with open(os.path.join(base_dir,'TPR'), 'wb') as handle:
            pkl.dump(self.rates, handle)
              

class SPLIT_FEDSIS(FedNetwork):
    def __init__(
        self, num_clients, device, 
        network, criterion, network_name, base_dir, 
        initial_block, final_block
        ):

        self.initial_block = initial_block
        self.final_block   = final_block    
        self.num_clients = num_clients
        self.device = device
        self.network = network
        self.criterion = criterion
        self.network_name = network_name
        self.base_dir = base_dir
        self.train_chosen_blocks = [0 for i in range(self.num_clients)]
    
    def train_round(self, client_i, r):
        """
        Training loop. s

        client_i: Client index.

        """
        
        running_loss_client_i = 0
        whole_labels = []
        whole_preds = []
        whole_probs = []
        self.chosen_block = np.random.randint(low = self.initial_block, high= self.final_block+1) 
        self.train_chosen_blocks[client_i] =  self.chosen_block
        copy_network = copy.deepcopy(self.network)
        weight_dic = {}
        weight_dic['blocks'] = None
        weight_dic['cls'] = None
        weight_dic['pos_embed'] = None
        weight_dic['resnet'] = None
        print("==== Training ====")
        print(f"Block {self.chosen_block} chosen for client {client_i}")
        self.network.train()
        for counter, data in enumerate(self.CLIENTS_DATALOADERS[client_i]): 
            self.optimizer.zero_grad()
            imgs, labels = data[0].to(device), data[1].to(device)
            labels = labels.reshape(labels.shape[0])
            tail_output,_ = self.network(x=imgs, chosen_block=self.chosen_block ,client_idx = client_i)
            m = nn.Sigmoid()
            loss = self.criterion(m(tail_output), labels.reshape(labels.shape[0],1).float())
            loss.backward()
            self.optimizer.step()
            running_loss_client_i+= loss.item() 
            predicted = m(tail_output).reshape(-1).detach().cpu().numpy().round()
            whole_probs.append(m(tail_output).detach().cpu())
            whole_labels.append(labels.detach().cpu())
            whole_preds.append(torch.tensor(predicted)) 

        self.metrics(client_i = client_i, whole_labels = whole_labels, whole_preds = whole_preds,
                    running_loss_client_i = running_loss_client_i, len_loader = len(self.CLIENTS_DATALOADERS[client_i]), 
                    whole_probs = whole_probs, train = True, r = None, whole_video_id=None)
        
        weight_dic['blocks'] = weight_vec(self.network.vit.blocks).detach().cpu()
        weight_dic['cls'] = self.network.vit.cls_token.detach().cpu()
        weight_dic['pos_embed'] = self.network.vit.pos_embed.detach().cpu()
        self.network.vit.blocks = copy.deepcopy(copy_network.vit.blocks)
        self.network.vit.cls_token = copy.deepcopy(copy_network.vit.cls_token)
        self.network.vit.pos_embed =  copy.deepcopy(copy_network.vit.pos_embed)
        return weight_dic

    def eval_round(self, client_i, r):
        """
        Evaluation loop. 

        client_i: Client index.
                
        """
        running_loss_client_i = 0
        whole_labels = []
        whole_preds = []
        whole_probs = []
        whole_video_id = []
        # num_b = np.random.randint(low = self.initial_block, high=self.final_block+1) 
        num_b = self.train_chosen_blocks[client_i]
        print("==== Testing ====")
        print(f"Block {num_b} chosen for client {client_i}")
        self.network.eval()
        with torch.no_grad():
            for data in self.testloader: 
                imgs, labels, video_id = data[0].to(device), data[1].to(device), data[2].to(device)
                tail_output,_ = self.network(x=imgs, chosen_block=num_b, client_idx = client_i)        
                m = nn.Sigmoid()
                loss = self.criterion(m(tail_output), labels.reshape(labels.shape[0],1).float())
                running_loss_client_i+= loss.item() 
                predicted = m(tail_output).reshape(-1).detach().cpu().numpy().round()
                whole_probs.append(m(tail_output).detach().cpu())
                whole_labels.append(labels.detach().cpu())
                whole_preds.append(torch.tensor(predicted))
                whole_video_id.append(video_id)


            self.metrics(client_i = client_i, whole_labels = whole_labels, whole_preds = whole_preds,
                running_loss_client_i = running_loss_client_i, len_loader = len(self.CLIENTS_DATALOADERS[client_i]), 
                whole_probs = whole_probs, train = False, r = r, whole_video_id=whole_video_id)
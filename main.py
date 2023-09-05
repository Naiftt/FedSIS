import os 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # used to select which gpu to use if you're in a machine with more than one gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # gpu number
import numpy as np
import models
from data_split import choose_dataset
import torch
from torch import nn
from tqdm import tqdm
import copy
from argument_parser import parse_arguments
from utils import weight_dec_global, weight_vec, set_seed
from metadata import write_metadata_to_file
import datetime


# 1. Parse Arguments and Set Seed
args = parse_arguments()
set_seed(args)

# 2. Parse Arguments into Variables
dataset_name = args.datasetname
rounds = args.rounds
combination = args.combination
diff_privacy = args.diff_privacy == 'True'
pretrained = args.pretrained == 'imageNet'
festa = args.festa == 'True'
initial_block = args.initial_block
final_block = args.final_block
local_step = args.local_step
epsilon = args.epsilon
lr = args.lr
celebA = args.celebA == 'True'
pretrained_start = celebA and args.pretrained_start == 'True'
num_clients = args.num_clients
balanced = args.balanced == 'True'
batch_size = args.batch_size
celebABS = args.celebABS # batch size for celebA

# Other Constants
delta = 1e-5 # parameter for diffrential privacy
weight_decay = 1e-6
drop_out_mlp_server = 0.5 # drop out for mlp at the server
fraction_pretraining = 0.9 # pretraining if it's 0.9, it will take 90% of celebA for pretraining (The file is saved)
drop_last = False
model_name = 'vit_base_r50_s16_224_in21k'
save_every_epochs = 3
in_chans = 3
num_blocks = 12
embedding = 768
std = np.sqrt(2 * np.math.log(1.25/delta)) / epsilon 
mean = 0
# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Get current date and time
current_datetime = datetime.datetime.now()

# Convert to a formatted string
current_datetime_string = current_datetime.strftime('%Y-%m-%d %H:%M:%S')

# Create the directory for experiment output
dir_name = f'exps_output/{current_datetime_string}'
os.makedirs(dir_name, exist_ok=True)
print(f"Logging to: {dir_name}")

# Save metadata to a text file for the current run
write_metadata_to_file(args, dir_name)

# Choosing the dataset and determining the split
DATALOADERS, test_loader, num_classes = choose_dataset(dataset_name, combination, batch_size, celebA, drop_last, celebABS, balanced, pretrained_start, num_clients)

# Create the FEDSIS model
if pretrained_start:
    fedsis = models.FEDSIS(
        ViT_name=model_name, num_classes=num_classes, drop_out_mlp_server=drop_out_mlp_server,
        embedding=embedding, num_clients=1, in_channels=in_chans, ViT_pretrained=pretrained,
        initial_block=initial_block, final_block=final_block,
        std=std, mean=mean, diff_privacy=diff_privacy
    ).to(device)

    if fraction_pretraining == 1:
        print("Loading the model trained on 100 percent of the data!")
        fedsis.load_state_dict(torch.load("celebA_pretrained_model100.pt"))
    elif fraction_pretraining == 0.9: 
        print("Loading the model trained on 90 percent of the data!")
        fedsis.load_state_dict(torch.load("celebA_pretrained_model.pt"))

    fedsis.resnet50_clients = nn.ModuleList([copy.deepcopy(fedsis.resnet50) for _ in range(num_clients)])
    fedsis.mlp_clients_tail = nn.ModuleList([copy.deepcopy(fedsis.mlp_clients_tail[0]) for _ in range(num_clients)])
else:
    fedsis = models.FEDSIS(
        ViT_name=model_name, num_classes=num_classes, drop_out_mlp_server=drop_out_mlp_server, 
        embedding=embedding, num_clients=num_clients, in_channels=in_chans, ViT_pretrained=pretrained,
        initial_block=initial_block, final_block=final_block
    ).to(device)


# Defining the loss    
criterion = nn.BCELoss()

Split = models.SPLIT_FEDSIS(
    num_clients=num_clients, device = device, network = fedsis, 
    criterion = criterion, network_name=model_name, base_dir=dir_name, 
    initial_block = initial_block,
    final_block= final_block)

# Assign the dataloaders to clients
Split.CLIENTS_DATALOADERS = DATALOADERS
Split.testloader = test_loader
# choose opt for protocol
Split.set_optimizer('Adam', lr = lr, weight_decay=weight_decay)
Split.init_logs()

# start training
for r in tqdm(range(rounds)):
    print(f"Round {r+1} / {rounds}")
    agg_weights = None
    for client_i in range(num_clients):
        # training 
        weight_dict = Split.train_round(client_i, r)
        if client_i == 0: 
            agg_weights = weight_dict
        else: 
            agg_weights['blocks'] +=  weight_dict['blocks']
            agg_weights['cls'] +=  weight_dict['cls']
            agg_weights['pos_embed'] +=  weight_dict['pos_embed']

    agg_weights['blocks'] /= num_clients
    agg_weights['cls'] /= num_clients
    agg_weights['pos_embed'] /= num_clients  

    Split.network.vit.blocks = weight_dec_global(
        Split.network.vit.blocks,
        agg_weights['blocks'].to(device)
        )
    
    Split.network.vit.cls_token.data = agg_weights['cls'].to(device) + 0.0
    Split.network.vit.pos_embed.data = agg_weights['pos_embed'].to(device) + 0.0



    if festa == True and (r % local_step == 0 and r!= 0):
        tails_weights = []
        head_weights = []
        for head, tail in zip(Split.network.resnet50_clients, Split.network.mlp_clients_tail):
            head_weights.append(weight_vec(head).detach().cpu())
            tails_weights.append(weight_vec(tail).detach().cpu())
        
        mean_avg_tail = torch.mean(torch.stack(tails_weights), axis = 0)
        mean_avg_head = torch.mean(torch.stack(head_weights), axis = 0)

        for i in range(num_clients):
            Split.network.mlp_clients_tail[i] = weight_dec_global(Split.network.mlp_clients_tail[i], 
                                                                mean_avg_tail.to(device))
            Split.network.resnet50_clients[i] = weight_dec_global(Split.network.resnet50_clients[i], 
                                                                mean_avg_head.to(device))

    for client_i in range(num_clients):
        Split.eval_round(client_i,r)
        
    print('---------')
    # Saving logs ...        
    if r%save_every_epochs==0 and r!=0: 
        Split.save_pickles(dir_name)
    print('============================================')
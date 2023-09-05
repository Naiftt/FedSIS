import torch
import os
import numpy as np 
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image
import cv2
from torch.utils.data import WeightedRandomSampler
import random 

seed = 105
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)




class FASDataset(Dataset):

  def __init__(self, data, transforms=None, train=True):
    self.base_dir = 'data/MCIO/frame/'
    self.train = train
    self.photo_path = data[0] + data[1]
    self.photo_label = [0 for i in range(len(data[0]))
                       ] + [1 for i in range(len(data[1]))]

    # MCIO
    u, indices = np.unique(
        np.array([
            i.replace('frame0.png', '').replace('frame1.png', '')
            for i in data[0] + data[1]
        ]),
        return_inverse=True)

    self.photo_belong_to_video_ID = indices

    if transforms is None:
      if not train:
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
      else:
        self.transforms = T.Compose([
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
      self.transforms = transforms

  def __len__(self):
    return len(self.photo_path)

  def __getitem__(self, item):
    if self.train:
      img_path = self.base_dir + self.photo_path[item] 
      label = self.photo_label[item]
      img = cv2.imread(img_path)
      img = img.astype(np.float32)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
      if np.random.randint(2):
        img[..., 1] *= np.random.uniform(0.8, 1.2)
      img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
      img = Image.fromarray(img.astype(np.uint8)).resize((224, 224))
      img = self.transforms(img)
      return img, label

    else:
      videoID = self.photo_belong_to_video_ID[item]
      img_path = self.base_dir + self.photo_path[item] 
      label = self.photo_label[item]
      img = cv2.imread(img_path)
      img = img.astype(np.float32)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img = Image.fromarray(img.astype(np.uint8)).resize((224, 224))
      img = self.transforms(img)
      return img, label, videoID, img_path

class WCSDataset(Dataset):

    def __init__(self, data, transforms=None, train=True,):
        self.train = train
        self.base_path = 'data/WCS/frame/'
        self.photo_path = data[0] + data[1]
        self.photo_label = [0] * len(data[0]) + [1] * len(data[1])

        # MCIO
        unique_paths, indices = np.unique(
            [os.path.splitext(os.path.basename(i))[0] for i in self.photo_path],
            return_inverse=True)

        self.photo_belong_to_video_ID = indices

        default_transforms = [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        if train:
            default_transforms.insert(0, T.RandomHorizontalFlip())
        self.transforms = T.Compose(default_transforms) if transforms is None else transforms

    def __len__(self):
        return len(self.photo_path)

    def __getitem__(self, item):
        img_path = os.path.join(self.base_path, self.photo_path[item])
        label = self.photo_label[item]
        img = cv2.imread(img_path)
        img = img.astype(np.float32)
        
        if self.train:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            if np.random.randint(2):
                img[..., 1] *= np.random.uniform(0.8, 1.2)
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        
        img = Image.fromarray(img.astype(np.uint8)).resize((224, 224))
        img = self.transforms(img)

        if self.train:
            return img, label
        else:
            videoID = self.photo_belong_to_video_ID[item]
            return img, label, videoID, img_path


def weighted_sampler(dataset):
    num_samples = len(dataset)
    labels = [label for _, label in dataset]
    class_counts = torch.tensor([labels.count(i) for i in range(2)])
    weights = 1.0 / class_counts
    sample_weights = [weights[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples, replacement=True)
    return sampler

# Function to load data from a file and strip whitespace
def load_data(file_path):
    with open(file_path) as f:
        lines = [line.strip() for line in f.readlines()]
    return lines

# Function to modify frame names from 'frame0' to 'frame1'
def modify_frame_name(line):
    return line.replace('frame0', 'frame1')

# Function to load data from a file and apply frame modification
def load_data_with_frame_modification(file_path):
    lines = load_data(file_path)
    modified_lines = [modify_frame_name(line) for line in lines]
    return lines + modified_lines


# Main function to split and prepare data
def split_mcio(batch_size, celebABS, balanced, drop_last, celebA=False):
    base_dir='data/MCIO//txt/'
    # List of dataset names and training/testing phases
    datasets = ['casia', 'msu', 'oulu', 'replay']
    phases = ['train', 'test']

    # Empty lists to store DataLoader objects and datasets
    clients_train_loader = []
    clients_test_loader = []
    all_datasets = []

    # Iterate through datasets and phases
    for dataset in datasets:
        for phase in phases:
            # Load and modify fake and real data
            data_fake = load_data_with_frame_modification(os.path.join(base_dir, dataset + f'_fake_{phase}.txt'))
            data_real = load_data_with_frame_modification(os.path.join(base_dir, dataset + f'_real_{phase}.txt'))
            
            # Create a dataset object
            dataset_train = FASDataset(data=[data_fake, data_real], transforms=None, train=(phase == 'train'))
            
            # Add the dataset to the list of all datasets
            all_datasets.append(dataset_train)

            if balanced:
                # Perform weighted sampling for balanced data
                weighted_sampling = weighted_sampler(dataset_train)
                
                # Create a DataLoader object with weighted sampling
                train_loader = DataLoader(dataset_train, shuffle=False, num_workers=8, batch_size=batch_size,
                                          sampler=weighted_sampling, drop_last=drop_last)
            else:
                # Create a DataLoader object with random shuffling
                train_loader = DataLoader(dataset_train, shuffle=True, num_workers=8, batch_size=batch_size,
                                          drop_last=drop_last)

            if phase == 'train':
                # Append train loader to the list of train loaders
                clients_train_loader.append(train_loader)
            else:
                # Create a test loader with no shuffling
                test_loader = DataLoader(dataset_train, shuffle=False, num_workers=8, batch_size=8)
                
                # Append test loader to the list of test loaders
                clients_test_loader.append(test_loader)

    if celebA:
        # Load and modify CelebA fake and real data
        celebA_data_fake = load_data_with_frame_modification(os.path.join(base_dir, 'celeb_fake_train.txt'))
        celebA_data_real = load_data_with_frame_modification(os.path.join(base_dir, 'celeb_real_train.txt'))
        
        # Create a CelebA dataset object
        celebA_dataset_train = FASDataset(data=[celebA_data_fake, celebA_data_real], transforms=None, train=True)
        all_datasets.append(celebA_dataset_train)
        
        if balanced:
            # Perform weighted sampling for balanced CelebA data
            weighted_sampling_celebA = weighted_sampler(celebA_dataset_train)
            
            # Create a CelebA DataLoader object with weighted sampling
            celebA_train_loader = DataLoader(celebA_dataset_train, shuffle=False, num_workers=8, batch_size=celebABS,
                                             sampler=weighted_sampling_celebA, drop_last=drop_last)
        else:
            # Create a CelebA DataLoader object with random shuffling
            celebA_train_loader = DataLoader(celebA_dataset_train, shuffle=True, num_workers=8, batch_size=celebABS,
                                             drop_last=drop_last)

        # Append CelebA train loader to the list of train loaders
        clients_train_loader.append(celebA_train_loader)

    # Return lists of train loaders, test loaders, and all datasets
    return clients_train_loader, clients_test_loader, all_datasets

def split_wcs(batch_size, celebABS,  balanced, drop_last, celebA= False):
    base_dir = 'data/WCS/txt'
    base_dir_celeb = 'data/MCIO/txt'
    cefa = 'cefa'
    surf = 'surf'
    wmca = 'wmca'
    train = 'train'
    test = 'test'

    # train data 
    cefa_data_train_fake  = [ i.strip() for i in open(f'{base_dir}/{cefa}' + f'_fake_{train}.txt').readlines() ] 
    cefa_data_train_real  = [ i.strip() for i in open(f'{base_dir}/{cefa}' + f'_real_{train}.txt').readlines() ]

    surf_data_train_fake  = [ i.strip() for i in open(f'{base_dir}/{surf}' + f'_fake_{train}.txt').readlines() ]
    surf_data_train_real  = [ i.strip() for i in open(f'{base_dir}/{surf}' + f'_real_{train}.txt').readlines() ]

    wmca_data_train_fake  = [ i.strip() for i in open(f'{base_dir}/{wmca}' + f'_fake_{train}.txt').readlines() ]
    wmca_data_train_real  = [ i.strip() for i in open(f'{base_dir}/{wmca}' + f'_real_{train}.txt').readlines() ]

    celebA_data_train_fake = [ i.strip() for i in open(f'{base_dir_celeb}/celeb' + f'_fake_{train}.txt').readlines() ]
    celebA_data_train_real = [ i.strip() for i in open(f'{base_dir_celeb}/celeb' + f'_real_{train}.txt').readlines() ]

    # test data 
    cefa_data_test_fake  = [ i.strip() for i in open(f'{base_dir}/{cefa}' + f'_fake_{test}.txt').readlines() ] + \
    [i.strip().replace('frame0', 'frame1') for i in open(f"{base_dir}/{cefa}" + f'_fake_{test}.txt').readlines()]

    cefa_data_test_real  = [ i.strip() for i in open(f'{base_dir}/{cefa}' + f'_real_{test}.txt').readlines() ] + \
    [i.strip().replace('frame0', 'frame1') for i in open(f"{base_dir}/{cefa}" + f'_real_{test}.txt').readlines()]
    # ==================================
    surf_data_test_fake  = [ i.strip() for i in open(f'{base_dir}/{surf}' + f'_fake_{test}.txt').readlines() ] + \
    [i.strip().replace('frame0', 'frame1') for i in open(f"{base_dir}/{surf}" + f'_fake_{test}.txt').readlines()]

    surf_data_test_real  = [ i.strip() for i in open(f'{base_dir}/{surf}' + f'_real_{test}.txt').readlines() ] + \
    [i.strip().replace('frame0', 'frame1') for i in open(f"{base_dir}/{surf}" + f'_real_{test}.txt').readlines()]
    # ==================================
    wmca_data_test_fake  = [ i.strip() for i in open(f'{base_dir}/{wmca}' + f'_fake_{test}.txt').readlines() ] + \
    [i.strip().replace('frame0', 'frame1') for i in open(f"{base_dir}/{wmca}" + f'_fake_{test}.txt').readlines()]

    wmca_data_test_real  = [ i.strip() for i in open(f'{base_dir}/{wmca}' + f'_real_{test}.txt').readlines() ] + \
    [i.strip().replace('frame0', 'frame1') for i in open(f"{base_dir}/{wmca}" + f'_real_{test}.txt').readlines()]

    # dataset class train
    cefa_dataset_train = WCSDataset(data=[cefa_data_train_fake, cefa_data_train_real], transforms=None, train=True)
    surf_dataset_train = WCSDataset(data=[surf_data_train_fake, surf_data_train_real], transforms=None, train=True)
    wmca_dataset_train = WCSDataset(data=[wmca_data_train_fake, wmca_data_train_real], transforms=None, train=True)
    celebA_dataset_train = FASDataset(data=[celebA_data_train_fake, celebA_data_train_real], transforms=None, train=True) 

    all_datasets = [cefa_dataset_train, surf_dataset_train, wmca_dataset_train, celebA_dataset_train]
    if balanced:
        # weighted sampling
        weighted_sampling_cefa = weighted_sampler(cefa_dataset_train)
        weighted_sampling_surf =  weighted_sampler(surf_dataset_train)
        weighted_sampling_wmca =  weighted_sampler(wmca_dataset_train)
        weighted_sampling_celebA = weighted_sampler(celebA_dataset_train)
        # train loaders
        cefa_dataset_train_loader = torch.utils.data.DataLoader(cefa_dataset_train, shuffle = False, num_workers = 8, batch_size = batch_size, 
                                                                sampler = weighted_sampling_cefa, drop_last = drop_last)
        surf_dataset_train_loader = torch.utils.data.DataLoader(surf_dataset_train, shuffle = False, num_workers = 8, batch_size = batch_size, 
                                                                sampler = weighted_sampling_surf, drop_last = drop_last)
        wmca_dataset_train_loader = torch.utils.data.DataLoader(wmca_dataset_train, shuffle = False, num_workers = 8, batch_size = batch_size, 
                                                                sampler = weighted_sampling_wmca, drop_last = drop_last)
        celebA_dataset_train_loader = torch.utils.data.DataLoader(celebA_dataset_train, shuffle = False, num_workers = 8, batch_size = celebABS, 
                                                                sampler = weighted_sampling_celebA, drop_last = drop_last)
    else:
        # train loaders
        cefa_dataset_train_loader = torch.utils.data.DataLoader(cefa_dataset_train, shuffle = True, num_workers = 8, batch_size = batch_size, drop_last = drop_last)
        surf_dataset_train_loader = torch.utils.data.DataLoader(surf_dataset_train, shuffle = True, num_workers = 8, batch_size = batch_size, drop_last = drop_last)
        wmca_dataset_train_loader = torch.utils.data.DataLoader(wmca_dataset_train, shuffle = True, num_workers = 8, batch_size = batch_size, drop_last = drop_last)
        celebA_dataset_train_loader = torch.utils.data.DataLoader(celebA_dataset_train, shuffle = True, num_workers = 8, batch_size = celebABS, drop_last = drop_last)


    # dataset class test
    cefa_dataset_test = WCSDataset(data=[cefa_data_test_fake, cefa_data_test_real], transforms=None, train=False)
    surf_dataset_test = WCSDataset(data=[surf_data_test_fake, surf_data_test_real], transforms=None, train=False)
    wmca_dataset_test = WCSDataset(data=[wmca_data_test_fake, wmca_data_test_real], transforms=None, train=False)

    # test loaders
    cefa_dataset_test_loader = torch.utils.data.DataLoader(cefa_dataset_test, shuffle = False, num_workers = 8, batch_size = 32)
    surf_dataset_test_loader = torch.utils.data.DataLoader(surf_dataset_test, shuffle = False, num_workers = 8, batch_size = 32)
    wmca_dataset_test_loader = torch.utils.data.DataLoader(wmca_dataset_test, shuffle = False, num_workers = 8, batch_size = 32)
    CLIENTS_TRAIN_LOADER = [cefa_dataset_train_loader, surf_dataset_train_loader, wmca_dataset_train_loader]
    if celebA:
        CLIENTS_TRAIN_LOADER.append(celebA_dataset_train_loader)
    CLIENTS_TEST_LOADER = [cefa_dataset_test_loader, surf_dataset_test_loader, wmca_dataset_test_loader]
    all_datasets_test = [cefa_dataset_test, surf_dataset_test, wmca_dataset_test]
    return CLIENTS_TRAIN_LOADER, CLIENTS_TEST_LOADER, all_datasets, all_datasets_test


# Effect of one attack type per client - experiments 1 and 2
def mcio_exp1(batch_size, celebABS,  balanced, drop_last, celebA= False):
    base_dir = 'data/one_attack_per_client/mcio_exp1'
    base_dir_celeb = 'data/MCIO/txt'
    casia = 'casia'
    msu = 'msu'
    oulu = 'oulu'
    replay = 'replay'
    train = 'train'
    test = 'test'

    # train data for (client1) casia_1 
    casia_fake_train_1_photo  = [ i.strip() for i in open(f'{base_dir}/{casia}' + f'_fake_{train}_1_photo.txt').readlines() ] 
    casia_real_train_1  = [ i.strip() for i in open(f'{base_dir}/{casia}' + f'_real_{train}_1.txt').readlines() ]

    # train data for (client2) casia_2
    casia_fake_train_2_video  = [ i.strip() for i in open(f'{base_dir}/{casia}' + f'_fake_{train}_2_video.txt').readlines() ] 
    casia_real_train_2 = [ i.strip() for i in open(f'{base_dir}/{casia}' + f'_real_{train}_2.txt').readlines() ]
    
    # train_data for (client_3) msu_3
    msu_fake_train_1_photo  = [ i.strip() for i in open(f'{base_dir}/{msu}' + f'_fake_{train}_1_photo.txt').readlines() ]
    msu_real_train_1  = [ i.strip() for i in open(f'{base_dir}/{msu}' + f'_real_{train}_1.txt').readlines()]

    # train_data for (client_4) msu_4
    msu_fake_train_2_video  = [ i.strip() for i in open(f'{base_dir}/{msu}' + f'_fake_{train}_2_video.txt').readlines() ]
    msu_real_train_2  = [ i.strip() for i in open(f'{base_dir}/{msu}' + f'_real_{train}_2.txt').readlines()]

    # train_data for (client_5) oulu_5
    oulu_fake_train_1_photo  = [ i.strip() for i in open(f'{base_dir}/{oulu}' + f'_fake_{train}_1_photo.txt').readlines() ]
    oulu_real_train_1  = [ i.strip() for i in open(f'{base_dir}/{oulu}' + f'_real_{train}_1.txt').readlines()]

    # train_data for (client_6) oulu_6
    oulu_fake_train_2_video  = [ i.strip() for i in open(f'{base_dir}/{oulu}' + f'_fake_{train}_2_video.txt').readlines() ]
    oulu_real_train_2  = [ i.strip() for i in open(f'{base_dir}/{oulu}' + f'_real_{train}_2.txt').readlines()]

    # train_data for (client_7) replay_7
    replay_fake_train_1_photo  = [ i.strip() for i in open(f'{base_dir}/{replay}' + f'_fake_{train}_1_photo.txt').readlines() ]
    replay_real_train_1  = [ i.strip() for i in open(f'{base_dir}/{replay}' + f'_real_{train}_1.txt').readlines()]

    # train_data for (client_8) replay_8
    replay_fake_train_2_video  = [ i.strip() for i in open(f'{base_dir}/{replay}' + f'_fake_{train}_2_video.txt').readlines() ]
    replay_real_train_2  = [ i.strip() for i in open(f'{base_dir}/{replay}' + f'_real_{train}_2.txt').readlines()]

    # train_data for (client_9) celeb_9
    celebA_data_train_fake = [ i.strip() for i in open(f'{base_dir_celeb}/celeb' + f'_fake_{train}.txt').readlines() ]
    celebA_data_train_real = [ i.strip() for i in open(f'{base_dir_celeb}/celeb' + f'_real_{train}.txt').readlines() ]

    # test data 
    casia_data_test_fake  = [ i.strip() for i in open(f'{base_dir}/{casia}' + f'_fake_{test}.txt').readlines() ] + \
    [i.strip().replace('frame0', 'frame1') for i in open(f"{base_dir}/{casia}" + f'_fake_{test}.txt').readlines()]

    casia_data_test_real  = [ i.strip() for i in open(f'{base_dir}/{casia}' + f'_real_{test}.txt').readlines() ] + \
    [i.strip().replace('frame0', 'frame1') for i in open(f"{base_dir}/{casia}" + f'_real_{test}.txt').readlines()]
    # ==================================
    msu_data_test_fake  = [ i.strip() for i in open(f'{base_dir}/{msu}' + f'_fake_{test}.txt').readlines() ] + \
    [i.strip().replace('frame0', 'frame1') for i in open(f"{base_dir}/{msu}" + f'_fake_{test}.txt').readlines()]

    msu_data_test_real  = [ i.strip() for i in open(f'{base_dir}/{msu}' + f'_real_{test}.txt').readlines() ] + \
    [i.strip().replace('frame0', 'frame1') for i in open(f"{base_dir}/{msu}" + f'_real_{test}.txt').readlines()]
    # ==================================
    oulu_data_test_fake  = [ i.strip() for i in open(f'{base_dir}/{oulu}' + f'_fake_{test}.txt').readlines() ] + \
    [i.strip().replace('frame0', 'frame1') for i in open(f"{base_dir}/{oulu}" + f'_fake_{test}.txt').readlines()]

    oulu_data_test_real  = [ i.strip() for i in open(f'{base_dir}/{oulu}' + f'_real_{test}.txt').readlines() ] + \
    [i.strip().replace('frame0', 'frame1') for i in open(f"{base_dir}/{oulu}" + f'_real_{test}.txt').readlines()]
    # ======================================
    replay_data_test_fake  = [ i.strip() for i in open(f'{base_dir}/{replay}' + f'_fake_{test}.txt').readlines() ] + \
    [i.strip().replace('frame0', 'frame1') for i in open(f"{base_dir}/{replay}" + f'_fake_{test}.txt').readlines()]

    replay_data_test_real  = [ i.strip() for i in open(f'{base_dir}/{replay}' + f'_real_{test}.txt').readlines() ] + \
    [i.strip().replace('frame0', 'frame1') for i in open(f"{base_dir}/{replay}" + f'_real_{test}.txt').readlines()]

    # dataset class train
    casia_dataset_train_1 = FASDataset(data=[casia_fake_train_1_photo, casia_real_train_1], transforms=None, train=True)
    casia_dataset_train_2 = FASDataset(data=[casia_fake_train_2_video, casia_real_train_2], transforms=None, train=True)

    msu_dataset_train_1 = FASDataset(data=[msu_fake_train_1_photo, msu_real_train_1], transforms=None, train=True)
    msu_dataset_train_2 = FASDataset(data=[msu_fake_train_2_video, msu_real_train_2], transforms=None, train=True)

    oulu_dataset_train_1 = FASDataset(data=[oulu_fake_train_1_photo, oulu_real_train_1], transforms=None, train=True)
    oulu_dataset_train_2 = FASDataset(data=[oulu_fake_train_2_video, oulu_real_train_2], transforms=None, train=True)

    replay_dataset_train_1 = FASDataset(data=[replay_fake_train_1_photo, replay_real_train_1], transforms=None, train=True)
    replay_dataset_train_2 = FASDataset(data=[replay_fake_train_2_video, replay_real_train_2], transforms=None, train=True)

    celebA_dataset_train = FASDataset(data=[celebA_data_train_fake, celebA_data_train_real], transforms=None, train=True) 

    all_datasets = [casia_dataset_train_1, casia_dataset_train_2, msu_dataset_train_1, msu_dataset_train_2, oulu_dataset_train_1, 
                    oulu_dataset_train_2, replay_dataset_train_1, replay_dataset_train_2]
    if balanced:
        # weighted sampling
        weighted_sampling_casia_1 = weighted_sampler(casia_dataset_train_1)
        weighted_sampling_casia_2 = weighted_sampler(casia_dataset_train_2)

        weighted_sampling_msu_1 = weighted_sampler(msu_dataset_train_1)
        weighted_sampling_msu_2 = weighted_sampler(msu_dataset_train_2)

        weighted_sampling_oulu_1 = weighted_sampler(oulu_dataset_train_1)
        weighted_sampling_oulu_2 = weighted_sampler(oulu_dataset_train_2)

        weighted_sampling_replay_1 = weighted_sampler(replay_dataset_train_1)
        weighted_sampling_replay_2 = weighted_sampler(replay_dataset_train_2)
        weighted_sampling_celebA = weighted_sampler(celebA_dataset_train)
        # train loaders
        casia_dataset_train_loader_1 = torch.utils.data.DataLoader(casia_dataset_train_1, shuffle = False, num_workers = 8, batch_size = batch_size, 
                                                                sampler = weighted_sampling_casia_1, drop_last = drop_last)
        casia_dataset_train_loader_2 = torch.utils.data.DataLoader(casia_dataset_train_2, shuffle = False, num_workers = 8, batch_size = batch_size, 
                                                                sampler = weighted_sampling_casia_2, drop_last = drop_last)
        msu_dataset_train_loader_1 = torch.utils.data.DataLoader(msu_dataset_train_1, shuffle = False, num_workers = 8, batch_size = batch_size, 
                                                                sampler = weighted_sampling_msu_1, drop_last = drop_last)
        msu_dataset_train_loader_2 = torch.utils.data.DataLoader(msu_dataset_train_2, shuffle = False, num_workers = 8, batch_size = batch_size, 
                                                                sampler = weighted_sampling_msu_2, drop_last = drop_last)
        oulu_dataset_train_loader_1 = torch.utils.data.DataLoader(oulu_dataset_train_1, shuffle = False, num_workers = 8, batch_size = batch_size, 
                                                                sampler = weighted_sampling_oulu_1, drop_last = drop_last)
        oulu_dataset_train_loader_2 = torch.utils.data.DataLoader(oulu_dataset_train_2, shuffle = False, num_workers = 8, batch_size = batch_size, 
                                                                sampler = weighted_sampling_oulu_2, drop_last = drop_last)
        replay_dataset_train_loader_1 = torch.utils.data.DataLoader(replay_dataset_train_1, shuffle = False, num_workers = 8, batch_size = batch_size, 
                                                                sampler = weighted_sampling_replay_1, drop_last = drop_last)
        replay_dataset_train_loader_2 = torch.utils.data.DataLoader(replay_dataset_train_2, shuffle = False, num_workers = 8, batch_size = batch_size, 
                                                                sampler = weighted_sampling_replay_2, drop_last = drop_last)
        celebA_dataset_train_loader = torch.utils.data.DataLoader(celebA_dataset_train, shuffle = False, num_workers = 8, batch_size = celebABS, 
                                                                sampler = weighted_sampling_celebA, drop_last = drop_last)
    else:
       # train loaders
        casia_dataset_train_loader_1 = torch.utils.data.DataLoader(casia_dataset_train_1, shuffle = True, num_workers = 8, batch_size = batch_size, 
                                                                drop_last = drop_last)
        casia_dataset_train_loader_2 = torch.utils.data.DataLoader(casia_dataset_train_2, shuffle = True, num_workers = 8, batch_size = batch_size, 
                                                                drop_last = drop_last)
        msu_dataset_train_loader_1 = torch.utils.data.DataLoader(msu_dataset_train_1, shuffle = True, num_workers = 8, batch_size = batch_size, 
                                                                drop_last = drop_last)
        msu_dataset_train_loader_2 = torch.utils.data.DataLoader(msu_dataset_train_2, shuffle = True, num_workers = 8, batch_size = batch_size, 
                                                                 drop_last = drop_last)
        oulu_dataset_train_loader_1 = torch.utils.data.DataLoader(oulu_dataset_train_1, shuffle = True, num_workers = 8, batch_size = batch_size, 
                                                                 drop_last = drop_last)
        oulu_dataset_train_loader_2 = torch.utils.data.DataLoader(oulu_dataset_train_2, shuffle = True, num_workers = 8, batch_size = batch_size, 
                                                                 drop_last = drop_last)
        replay_dataset_train_loader_1 = torch.utils.data.DataLoader(replay_dataset_train_1, shuffle = True, num_workers = 8, batch_size = batch_size, 
                                                                 drop_last = drop_last)
        replay_dataset_train_loader_2 = torch.utils.data.DataLoader(replay_dataset_train_2, shuffle = True, num_workers = 8, batch_size = batch_size, 
                                                                 drop_last = drop_last)
        celebA_dataset_train_loader = torch.utils.data.DataLoader(celebA_dataset_train, shuffle = True, num_workers = 8, batch_size = celebABS, 
                                                                 drop_last = drop_last)


    # dataset class test
    casia_dataset_test = FASDataset(data=[casia_data_test_fake, casia_data_test_real], transforms=None, train=False)
    msu_dataset_test = FASDataset(data=[msu_data_test_fake, msu_data_test_real], transforms=None, train=False)
    oulu_dataset_test = FASDataset(data=[oulu_data_test_fake, oulu_data_test_real], transforms=None, train=False)
    replay_dataset_test = FASDataset(data=[replay_data_test_fake, replay_data_test_real], transforms=None, train=False)

    # test loaders
    casia_dataset_test_loader = torch.utils.data.DataLoader(casia_dataset_test, shuffle = False, num_workers = 8, batch_size = 32)
    msu_dataset_test_loader = torch.utils.data.DataLoader(msu_dataset_test, shuffle = False, num_workers = 8, batch_size = 32)
    oulu_dataset_test_loader = torch.utils.data.DataLoader(oulu_dataset_test, shuffle = False, num_workers = 8, batch_size = 32)
    replay_dataset_test_loader = torch.utils.data.DataLoader(replay_dataset_test, shuffle = False, num_workers = 8, batch_size = 32)
    CLIENTS_TRAIN_LOADER = [casia_dataset_train_loader_1, casia_dataset_train_loader_2, msu_dataset_train_loader_1, msu_dataset_train_loader_2,
                            oulu_dataset_train_loader_1, oulu_dataset_train_loader_2, replay_dataset_train_loader_1, 
                            replay_dataset_train_loader_2]


    if celebA:
        CLIENTS_TRAIN_LOADER.append(celebA_dataset_train_loader)
    CLIENTS_TEST_LOADER = [casia_dataset_test_loader, msu_dataset_test_loader, oulu_dataset_test_loader, replay_dataset_test_loader]
    all_datasets_test = [casia_dataset_test, msu_dataset_test, oulu_dataset_test, replay_dataset_test]
    
    return CLIENTS_TRAIN_LOADER, CLIENTS_TEST_LOADER, all_datasets, all_datasets_test

def mcio_exp2(batch_size, celebABS,  balanced, drop_last, celebA= False):
    base_dir = 'data/one_attack_per_client/mcio_exp2'
    base_dir_celeb = 'data/MCIO/txt'
    casia = 'casia'
    msu = 'msu'
    oulu = 'oulu'
    replay = 'replay'
    train = 'train'
    test = 'test'

    # train data for (client1) casia_1 
    casia_fake_train_1_photo  = [ i.strip() for i in open(f'{base_dir}/{casia}' + f'_fake_{train}_1_photo.txt').readlines() ] 
    casia_real_train_1  = [ i.strip() for i in open(f'{base_dir}/{casia}' + f'_real_{train}_1.txt').readlines() ]

    # train data for (client2) casia_2
    casia_fake_train_2_video  = [ i.strip() for i in open(f'{base_dir}/{casia}' + f'_fake_{train}_2_video.txt').readlines() ] 
    casia_real_train_2 = [ i.strip() for i in open(f'{base_dir}/{casia}' + f'_real_{train}_2.txt').readlines() ]
    
    # train_data for (client_3) msu_3
    msu_fake_train_1_photo  = [ i.strip() for i in open(f'{base_dir}/{msu}' + f'_fake_{train}_1_photo.txt').readlines() ]
    msu_real_train_1  = [ i.strip() for i in open(f'{base_dir}/{msu}' + f'_real_{train}_1.txt').readlines()]

    # train_data for (client_4) msu_4
    msu_fake_train_2_video  = [ i.strip() for i in open(f'{base_dir}/{msu}' + f'_fake_{train}_2_video.txt').readlines() ]
    msu_real_train_2  = [ i.strip() for i in open(f'{base_dir}/{msu}' + f'_real_{train}_2.txt').readlines()]

    # train_data for (client_5) oulu_5
    oulu_fake_train_1_photo  = [ i.strip() for i in open(f'{base_dir}/{oulu}' + f'_fake_{train}_1_photo.txt').readlines() ]
    oulu_real_train_1  = [ i.strip() for i in open(f'{base_dir}/{oulu}' + f'_real_{train}_1.txt').readlines()]

    # train_data for (client_6) oulu_6
    oulu_fake_train_2_video  = [ i.strip() for i in open(f'{base_dir}/{oulu}' + f'_fake_{train}_2_video.txt').readlines() ]
    oulu_real_train_2  = [ i.strip() for i in open(f'{base_dir}/{oulu}' + f'_real_{train}_2.txt').readlines()]

    # train_data for (client_7) replay_7
    replay_fake_train_1_photo  = [ i.strip() for i in open(f'{base_dir}/{replay}' + f'_fake_{train}_1_photo.txt').readlines() ]
    replay_real_train_1  = [ i.strip() for i in open(f'{base_dir}/{replay}' + f'_real_{train}_1.txt').readlines()]

    # train_data for (client_8) replay_8
    replay_fake_train_2_video  = [ i.strip() for i in open(f'{base_dir}/{replay}' + f'_fake_{train}_2_video.txt').readlines() ]
    replay_real_train_2  = [ i.strip() for i in open(f'{base_dir}/{replay}' + f'_real_{train}_2.txt').readlines()]

    # train_data for (client_9) celeb_9
    celebA_data_train_fake = [ i.strip() for i in open(f'{base_dir_celeb}/celeb' + f'_fake_{train}.txt').readlines() ]
    celebA_data_train_real = [ i.strip() for i in open(f'{base_dir_celeb}/celeb' + f'_real_{train}.txt').readlines() ]

    # test data 
    casia_data_test_fake  = [ i.strip() for i in open(f'{base_dir}/{casia}' + f'_fake_{test}.txt').readlines() ] + \
    [i.strip().replace('frame0', 'frame1') for i in open(f"{base_dir}/{casia}" + f'_fake_{test}.txt').readlines()]

    casia_data_test_real  = [ i.strip() for i in open(f'{base_dir}/{casia}' + f'_real_{test}.txt').readlines() ] + \
    [i.strip().replace('frame0', 'frame1') for i in open(f"{base_dir}/{casia}" + f'_real_{test}.txt').readlines()]
    # ==================================
    msu_data_test_fake  = [ i.strip() for i in open(f'{base_dir}/{msu}' + f'_fake_{test}.txt').readlines() ] + \
    [i.strip().replace('frame0', 'frame1') for i in open(f"{base_dir}/{msu}" + f'_fake_{test}.txt').readlines()]

    msu_data_test_real  = [ i.strip() for i in open(f'{base_dir}/{msu}' + f'_real_{test}.txt').readlines() ] + \
    [i.strip().replace('frame0', 'frame1') for i in open(f"{base_dir}/{msu}" + f'_real_{test}.txt').readlines()]
    # ==================================
    oulu_data_test_fake  = [ i.strip() for i in open(f'{base_dir}/{oulu}' + f'_fake_{test}.txt').readlines() ] + \
    [i.strip().replace('frame0', 'frame1') for i in open(f"{base_dir}/{oulu}" + f'_fake_{test}.txt').readlines()]

    oulu_data_test_real  = [ i.strip() for i in open(f'{base_dir}/{oulu}' + f'_real_{test}.txt').readlines() ] + \
    [i.strip().replace('frame0', 'frame1') for i in open(f"{base_dir}/{oulu}" + f'_real_{test}.txt').readlines()]
    # ======================================
    replay_data_test_fake  = [ i.strip() for i in open(f'{base_dir}/{replay}' + f'_fake_{test}.txt').readlines() ] + \
    [i.strip().replace('frame0', 'frame1') for i in open(f"{base_dir}/{replay}" + f'_fake_{test}.txt').readlines()]

    replay_data_test_real  = [ i.strip() for i in open(f'{base_dir}/{replay}' + f'_real_{test}.txt').readlines() ] + \
    [i.strip().replace('frame0', 'frame1') for i in open(f"{base_dir}/{replay}" + f'_real_{test}.txt').readlines()]

    # dataset class train
    casia_dataset_train_1 = FASDataset(data=[casia_fake_train_1_photo, casia_real_train_1], transforms=None, train=True)
    casia_dataset_train_2 = FASDataset(data=[casia_fake_train_2_video, casia_real_train_2], transforms=None, train=True)

    msu_dataset_train_1 = FASDataset(data=[msu_fake_train_1_photo, msu_real_train_1], transforms=None, train=True)
    msu_dataset_train_2 = FASDataset(data=[msu_fake_train_2_video, msu_real_train_2], transforms=None, train=True)

    oulu_dataset_train_1 = FASDataset(data=[oulu_fake_train_1_photo, oulu_real_train_1], transforms=None, train=True)
    oulu_dataset_train_2 = FASDataset(data=[oulu_fake_train_2_video, oulu_real_train_2], transforms=None, train=True)

    replay_dataset_train_1 = FASDataset(data=[replay_fake_train_1_photo, replay_real_train_1], transforms=None, train=True)
    replay_dataset_train_2 = FASDataset(data=[replay_fake_train_2_video, replay_real_train_2], transforms=None, train=True)

    celebA_dataset_train = FASDataset(data=[celebA_data_train_fake, celebA_data_train_real], transforms=None, train=True) 




    all_datasets = [casia_dataset_train_1, casia_dataset_train_2, msu_dataset_train_1, msu_dataset_train_2, oulu_dataset_train_1, 
                    oulu_dataset_train_2, replay_dataset_train_1, replay_dataset_train_2]
    if balanced:
        # weighted sampling
        weighted_sampling_casia_1 = weighted_sampler(casia_dataset_train_1)
        weighted_sampling_casia_2 = weighted_sampler(casia_dataset_train_2)

        weighted_sampling_msu_1 = weighted_sampler(msu_dataset_train_1)
        weighted_sampling_msu_2 = weighted_sampler(msu_dataset_train_2)

        weighted_sampling_oulu_1 = weighted_sampler(oulu_dataset_train_1)
        weighted_sampling_oulu_2 = weighted_sampler(oulu_dataset_train_2)

        weighted_sampling_replay_1 = weighted_sampler(replay_dataset_train_1)
        weighted_sampling_replay_2 = weighted_sampler(replay_dataset_train_2)
        weighted_sampling_celebA = weighted_sampler(celebA_dataset_train)
        # train loaders
        casia_dataset_train_loader_1 = torch.utils.data.DataLoader(casia_dataset_train_1, shuffle = False, num_workers = 8, batch_size = batch_size, 
                                                                sampler = weighted_sampling_casia_1, drop_last = drop_last)
        casia_dataset_train_loader_2 = torch.utils.data.DataLoader(casia_dataset_train_2, shuffle = False, num_workers = 8, batch_size = batch_size, 
                                                                sampler = weighted_sampling_casia_2, drop_last = drop_last)
        msu_dataset_train_loader_1 = torch.utils.data.DataLoader(msu_dataset_train_1, shuffle = False, num_workers = 8, batch_size = batch_size, 
                                                                sampler = weighted_sampling_msu_1, drop_last = drop_last)
        msu_dataset_train_loader_2 = torch.utils.data.DataLoader(msu_dataset_train_2, shuffle = False, num_workers = 8, batch_size = batch_size, 
                                                                sampler = weighted_sampling_msu_2, drop_last = drop_last)
        oulu_dataset_train_loader_1 = torch.utils.data.DataLoader(oulu_dataset_train_1, shuffle = False, num_workers = 8, batch_size = batch_size, 
                                                                sampler = weighted_sampling_oulu_1, drop_last = drop_last)
        oulu_dataset_train_loader_2 = torch.utils.data.DataLoader(oulu_dataset_train_2, shuffle = False, num_workers = 8, batch_size = batch_size, 
                                                                sampler = weighted_sampling_oulu_2, drop_last = drop_last)
        replay_dataset_train_loader_1 = torch.utils.data.DataLoader(replay_dataset_train_1, shuffle = False, num_workers = 8, batch_size = batch_size, 
                                                                sampler = weighted_sampling_replay_1, drop_last = drop_last)
        replay_dataset_train_loader_2 = torch.utils.data.DataLoader(replay_dataset_train_2, shuffle = False, num_workers = 8, batch_size = batch_size, 
                                                                sampler = weighted_sampling_replay_2, drop_last = drop_last)
        celebA_dataset_train_loader = torch.utils.data.DataLoader(celebA_dataset_train, shuffle = False, num_workers = 8, batch_size = celebABS, 
                                                                sampler = weighted_sampling_celebA, drop_last = drop_last)
    else:
       # train loaders
        casia_dataset_train_loader_1 = torch.utils.data.DataLoader(casia_dataset_train_1, shuffle = True, num_workers = 8, batch_size = batch_size, 
                                                                drop_last = drop_last)
        casia_dataset_train_loader_2 = torch.utils.data.DataLoader(casia_dataset_train_2, shuffle = True, num_workers = 8, batch_size = batch_size, 
                                                                drop_last = drop_last)
        msu_dataset_train_loader_1 = torch.utils.data.DataLoader(msu_dataset_train_1, shuffle = True, num_workers = 8, batch_size = batch_size, 
                                                                drop_last = drop_last)
        msu_dataset_train_loader_2 = torch.utils.data.DataLoader(msu_dataset_train_2, shuffle = True, num_workers = 8, batch_size = batch_size, 
                                                                 drop_last = drop_last)
        oulu_dataset_train_loader_1 = torch.utils.data.DataLoader(oulu_dataset_train_1, shuffle = True, num_workers = 8, batch_size = batch_size, 
                                                                 drop_last = drop_last)
        oulu_dataset_train_loader_2 = torch.utils.data.DataLoader(oulu_dataset_train_2, shuffle = True, num_workers = 8, batch_size = batch_size, 
                                                                 drop_last = drop_last)
        replay_dataset_train_loader_1 = torch.utils.data.DataLoader(replay_dataset_train_1, shuffle = True, num_workers = 8, batch_size = batch_size, 
                                                                 drop_last = drop_last)
        replay_dataset_train_loader_2 = torch.utils.data.DataLoader(replay_dataset_train_2, shuffle = True, num_workers = 8, batch_size = batch_size, 
                                                                 drop_last = drop_last)
        celebA_dataset_train_loader = torch.utils.data.DataLoader(celebA_dataset_train, shuffle = True, num_workers = 8, batch_size = celebABS, 
                                                                 drop_last = drop_last)


    # dataset class test
    casia_dataset_test = FASDataset(data=[casia_data_test_fake, casia_data_test_real], transforms=None, train=False)
    msu_dataset_test = FASDataset(data=[msu_data_test_fake, msu_data_test_real], transforms=None, train=False)
    oulu_dataset_test = FASDataset(data=[oulu_data_test_fake, oulu_data_test_real], transforms=None, train=False)
    replay_dataset_test = FASDataset(data=[replay_data_test_fake, replay_data_test_real], transforms=None, train=False)

    # test loaders
    casia_dataset_test_loader = torch.utils.data.DataLoader(casia_dataset_test, shuffle = False, num_workers = 8, batch_size = 32)
    msu_dataset_test_loader = torch.utils.data.DataLoader(msu_dataset_test, shuffle = False, num_workers = 8, batch_size = 32)
    oulu_dataset_test_loader = torch.utils.data.DataLoader(oulu_dataset_test, shuffle = False, num_workers = 8, batch_size = 32)
    replay_dataset_test_loader = torch.utils.data.DataLoader(replay_dataset_test, shuffle = False, num_workers = 8, batch_size = 32)
    CLIENTS_TRAIN_LOADER = [casia_dataset_train_loader_1, casia_dataset_train_loader_2, msu_dataset_train_loader_1, msu_dataset_train_loader_2,
                            oulu_dataset_train_loader_1, oulu_dataset_train_loader_2, replay_dataset_train_loader_1, 
                            replay_dataset_train_loader_2]


    if celebA:
        CLIENTS_TRAIN_LOADER.append(celebA_dataset_train_loader)
    CLIENTS_TEST_LOADER = [casia_dataset_test_loader, msu_dataset_test_loader, oulu_dataset_test_loader, replay_dataset_test_loader]
    all_datasets_test = [casia_dataset_test, msu_dataset_test, oulu_dataset_test, replay_dataset_test]
    
    return CLIENTS_TRAIN_LOADER, CLIENTS_TEST_LOADER, all_datasets, all_datasets_test

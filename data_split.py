import torch
import dataset


def choose_dataset(dataset_name, combination, batch_size, celebA, drop_last, celebABS, balanced, pretrained_start, num_clients):
    if dataset_name == 'mcio':
        CLIENTS_TRAIN_DATALOADERS, CLIENTS_TEST_DATALOADERS, _ = dataset.split_mcio(
            batch_size=batch_size, celebA=celebA, drop_last=drop_last, celebABS=celebABS, balanced=balanced
        )
        
        combination_mapping = {
            'cmo_r': (0, 1, 2, 3),
            'mor_c': (1, 2, 3, 0),
            'orc_m': (2, 3, 0, 1),
            'rcm_o': (3, 0, 1, 2),
        }
        
        DATALOADERS = [CLIENTS_TRAIN_DATALOADERS[i] for i in combination_mapping[combination]]
        test_loader = CLIENTS_TEST_DATALOADERS[combination_mapping[combination].index(0)]

        if celebA and num_clients == 4:
            if pretrained_start:
                print("adding only 10 percent of celebA")
                _, test_celebA = torch.load('celebASplit_data.pt')
                test_celebA_dataloader = torch.utils.data.DataLoader(
                    test_celebA, batch_size=celebABS, num_workers=8, shuffle=True, drop_last=drop_last
                )
                DATALOADERS.append(test_celebA_dataloader)
            else:
                print("celebA added to the training (Make sure you have 4 clients now!!!)")
                DATALOADERS.append(CLIENTS_TRAIN_DATALOADERS[4])

        num_classes = 1

    elif dataset_name == 'wcs':
        CLIENTS_TRAIN_DATALOADERS, CLIENTS_TEST_DATALOADERS, _, _ = dataset.split_wcs(
            batch_size=batch_size, celebA=celebA, drop_last=drop_last, celebABS=celebABS, balanced=balanced
        )
        
        combination_mapping = {
            'cs_w': (0, 1, 2),
            'sw_c': (1, 2, 0),
            'wc_s': (2, 0, 1),
        }
        
        DATALOADERS = [CLIENTS_TRAIN_DATALOADERS[i] for i in combination_mapping[combination]]
        test_loader = CLIENTS_TEST_DATALOADERS[combination_mapping[combination].index(0)]

        if celebA and num_clients == 3:
            if pretrained_start:
                print("adding only 10 percent of celebA to client 3 (this is WCS!)")
                _, test_celebA = torch.load('celebASplit_data.pt')
                test_celebA_dataloader = torch.utils.data.DataLoader(
                    test_celebA, batch_size=celebABS, num_workers=8, shuffle=True, drop_last=drop_last
                )
                DATALOADERS.append(test_celebA_dataloader)
            else:
                print(f"100% celebA added to the training (Make sure you have 4 clients now!!!)")
                DATALOADERS.append(CLIENTS_TRAIN_DATALOADERS[3])

        num_classes = 1

    elif dataset_name == 'mcio_exp1':
        CLIENTS_TRAIN_DATALOADERS, CLIENTS_TEST_DATALOADERS, _, _ = dataset.mcio_exp1(
            batch_size=batch_size, celebA=celebA, drop_last=drop_last, celebABS=celebABS, balanced=balanced
        )
        
        combination_mapping = {
            'cmo_r': (0, 1, 2, 3, 4, 5),
            'mor_c': (2, 3, 4, 5, 6, 7),
            'orc_m': (4, 5, 6, 7, 0, 1),
            'rcm_o': (6, 7, 0, 1, 2, 3),
        }
        
        DATALOADERS = [CLIENTS_TRAIN_DATALOADERS[i] for i in combination_mapping[combination]]
        test_loader = CLIENTS_TEST_DATALOADERS[combination_mapping[combination].index(0)]

        if celebA and num_clients == 7:
            if pretrained_start:
                print("adding only 10 percent of celebA")
                _, test_celebA = torch.load('celebASplit_data.pt') 
                test_celebA_dataloader = torch.utils.data.DataLoader(
                    test_celebA, batch_size=celebABS, num_workers=8, shuffle=True, drop_last=drop_last
                )
                DATALOADERS.append(test_celebA_dataloader)
            else:
                print("celebA added to the training (Make sure you have 7 clients now!!!)")
                DATALOADERS.append(CLIENTS_TRAIN_DATALOADERS[8])

        num_classes = 1

    elif dataset_name == 'mcio_exp2':
        CLIENTS_TRAIN_DATALOADERS, CLIENTS_TEST_DATALOADERS, _, _ = dataset.mcio_exp2(
            batch_size=batch_size, celebA=celebA, drop_last=drop_last, celebABS=celebABS, balanced=balanced
        )

        combination_mapping = {
            'cmo_r': (0, 1, 2, 3, 4, 5),
            'mor_c': (2, 3, 4, 5, 6, 7),
            'orc_m': (4, 5, 6, 7, 0, 1),
            'rcm_o': (6, 7, 0, 1, 2, 3),
        }

        DATALOADERS = [CLIENTS_TRAIN_DATALOADERS[i] for i in combination_mapping[combination]]
        test_loader = CLIENTS_TEST_DATALOADERS[combination_mapping[combination].index(0)]

        if celebA and num_clients == 7:
            if pretrained_start:
                print("adding only 10 percent of celebA")
                _, test_celebA = torch.load('celebASplit_data.pt')
                test_celebA_dataloader = torch.utils.data.DataLoader(
                    test_celebA, batch_size=celebABS, num_workers=8, shuffle=True, drop_last=drop_last
                )
                DATALOADERS.append(test_celebA_dataloader)
            else:
                print("celebA added to the training (Make sure you have 7 clients now!!!)")
                DATALOADERS.append(CLIENTS_TRAIN_DATALOADERS[8])

        num_classes = 1

    return DATALOADERS, test_loader, num_classes
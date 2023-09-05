import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument options for the protocol ')
    # Dataset options
    parser.add_argument('--datasetname', type=str,
                        help='Choose dataset: [mcio, wcs, mcio_mixed_data, mcio_exp1, mcio_exp2, mcio_unseen_print, mcio_unseen_replay]')
    # Number of rounds for training
    parser.add_argument('--rounds', type=int, default=200,
                        help='Choose number of rounds for the protocol')
    # Combination options
    parser.add_argument('--combination', type=str,
                        help='Choose combination mode and training/testing dataset: [cmo_r, mor_c, orc_m, rcm_o]')
    # Pretrained options
    parser.add_argument('--pretrained', type=str,
                        help='Choose pretrained model: [imageNet, no]')
    parser.add_argument('--seed', type=int,
                        help='Set seed value')
    parser.add_argument('--lr', type=float,
                        help='Set learning rate')
    parser.add_argument('--batch_size', type=int,
                        help='Set batch size for c, m, o, r')
    parser.add_argument('--celebABS', type=int,
                        help='Set batch size for celebA dataset')
    parser.add_argument('--initial_block', type=int,
                        help='Set initial block number')
    parser.add_argument('--final_block', type=int,
                        help='Set final block number')
    # Pretrained start option
    parser.add_argument('--pretrained_start', type=str,
                        help='Start with pretrained model on celebA: [True, False]')
    # Federation options
    parser.add_argument('--local_step', type=int,
                        help='Set number of rounds for federation of the head and tail')
    # Festa option
    parser.add_argument('--festa', type=str,
                        help='Perform federated learning between head and tail: [True, False]')
    # Number of clients option
    parser.add_argument('--num_clients', type=int,
                        help='Set number of participating clients')
    # CelebA option
    parser.add_argument('--celebA', type=str,
                        help='Add CelebA dataset to the system  [True, False]')
    # Balanced option
    parser.add_argument('--balanced', type=str,
                        help='Balance data between classes in each batch: [True, False]')
    # Differential privacy options
    parser.add_argument('--diff_privacy', type=str,
                        help='Enable differential privacy: [True, False]')
    parser.add_argument('--epsilon', type=float,
                        help='Set parameter for differential privacy')
    args = parser.parse_args()
    
    return args

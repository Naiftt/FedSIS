#!/bin/bash

# Move up one directory to the parent directory
cd ..

# Run the main Python script with the following arguments
    # seed                  : Set the random seed for reproducibility
    # datasetname           : Specify the dataset name
    # combination           : Specify the combination type (cs_w, sw_c, wc_s)
    # pretrained            : Specify the pretrained model type (ImageNet, False)
    # pretrained_start      : Whether to start with a pretrained model trained on celebA (True, False) (Note: if this is True, it will ignore the ImageNet model)
    # celebA                : Whether to include CelebA dataset in training (True, False)
    # lr                    : Set the learning rate
    # batch_size            : Set the batch size of each client
    # celebABS              : Set the batch size for CelebA client
    # rounds                : Specify the total number of training rounds
    # initial_block         : Specify the range of blocks to sample from initial block to final_block (Note: can't be greater than 12)
    # final_block           : Specify the range of blocks to sample from initial block to final_block (Note: can't be greater than 12)  
    # local_step            : Specify how often to federate head and tail
    # festa                 : Whether to federate head and tail (True, False)
    # num_clients           : Specify the number of clients
    # balanced              : Whether to use balanced classes in each batch (True, False)
    # diff_privacy          : Whether to use differential privacy
    # epsilon               : Specify the privacy parameter epsilon for differential privacy

python main.py \
    --seed 0 \
    --datasetname wcs \
    --combination cs_w \
    --pretrained imageNet \
    --pretrained_start True \
    --celebA True \
    --lr 1e-5 \
    --batch_size 6 \
    --celebABS 32 \
    --rounds 201 \
    --initial_block 1 \
    --final_block 6 \
    --local_step 10 \
    --festa True \
    --num_clients 3 \
    --balanced False \
    --diff_privacy False \
    --epsilon 0.1

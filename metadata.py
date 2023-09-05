import os 

def write_metadata_to_file(args, dir_name):
    metadata_file_path = os.path.join(dir_name, 'metadata.txt')
    with open(metadata_file_path, 'w') as metadata_file:
        metadata_file.write(f'Dataset: {args.datasetname}\n')
        metadata_file.write(f'Rounds: {args.rounds}\n')
        metadata_file.write(f'Combination: {args.combination}\n')
        metadata_file.write(f'Differential Privacy: {args.diff_privacy}\n')
        metadata_file.write(f'Pretrained: {args.pretrained == "imageNet"}\n')
        metadata_file.write(f'Festa: {args.festa == "True"}\n')
        metadata_file.write(f'Initial Block: {args.initial_block}\n')
        metadata_file.write(f'Final Block: {args.final_block}\n')
        metadata_file.write(f'Local Step: {args.local_step}\n')
        metadata_file.write(f'Epsilon: {args.epsilon}\n')
        metadata_file.write(f'Learning Rate: {args.lr}\n')
        metadata_file.write(f'CelebA: {args.celebA == "True"}\n')
        metadata_file.write(f'Pretrained Start: {args.celebA and args.pretrained_start == "True"}\n')
        metadata_file.write(f'Number of Clients: {args.num_clients}\n')
        metadata_file.write(f'Balanced: {args.balanced == "True"}\n')
        metadata_file.write(f'Batch Size: {args.batch_size}\n')
        metadata_file.write(f'CelebABS: {args.celebABS}\n')
    print(f"Metadata saved to: {metadata_file_path}")
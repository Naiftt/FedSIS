import numpy as np
import os
import argparse
def eval_exp(file_name):
    """
    Evaluate an experiment based on evaluation metrics.

    Parameters:
        file_name (str): The path to the directory containing evaluation metric files.

    Returns:
        tuple: A tuple containing the best Area Under the Curve (AUC),
           True Positive Rate (TPR), and Half Total Error Rate (HTER).
    """

    # Load evaluation metrics
    auc_test = np.array(np.load(os.path.join(file_name, 'AUC'), allow_pickle=True)['test'])
    tpr_test = np.array(np.load(os.path.join(file_name, 'TPR'), allow_pickle=True)['test'])
    hter_test = np.array(np.load(os.path.join(file_name, 'HTER'), allow_pickle=True)['test'])

    # Find the index (client, round) with the best HTER value
    best_index = np.unravel_index(np.argmin(hter_test), hter_test.shape)
    best_auc = auc_test[best_index]
    best_tpr = tpr_test[best_index]
    best_hter = hter_test[best_index]
    
    return best_auc, best_tpr, best_hter


parser = argparse.ArgumentParser(description="Evaluate experiment results")
parser.add_argument("--exp_path", type=str, help="path of the experiment folder")
args = parser.parse_args()
auc, hter, tpr = eval_exp(args.exp_path)

print(f"Best AUC: {auc:.2%}")
print(f"Best HTER: {hter:.2%}")
print(f"Best TPR: {tpr:.2%}")
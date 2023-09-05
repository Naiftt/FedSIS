import math
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def eval_state(probs, labels, thr):
  predict = probs >= thr
  TN = np.sum((labels == 0) & (predict == False))
  FN = np.sum((labels == 1) & (predict == False))
  FP = np.sum((labels == 0) & (predict == True))
  TP = np.sum((labels == 1) & (predict == True))
  return TN, FN, FP, TP


def calculate(probs, labels):
  TN, FN, FP, TP = eval_state(probs, labels, 0.5)
  APCER = 1.0 if (FP + TN == 0) else FP / float(FP + TN)
  NPCER = 1.0 if (FN + TP == 0) else FN / float(FN + TP)
  ACER = (APCER + NPCER) / 2.0
  ACC = (TP + TN) / labels.shape[0]
  return APCER, NPCER, ACER, ACC


def calculate_threshold(probs, labels, threshold):
  TN, FN, FP, TP = eval_state(probs, labels, threshold)
  ACC = (TP + TN) / labels.shape[0]
  return ACC


def get_threshold(probs, grid_density):
  Min, Max = min(probs), max(probs)
  thresholds = []
  for i in range(grid_density + 1):
    thresholds.append(0.0 + i * 1.0 / float(grid_density))
  thresholds.append(1.1)
  return thresholds


def get_EER_states(probs, labels, grid_density=10000):
  thresholds = get_threshold(probs, grid_density)
  min_dist = 1.0
  min_dist_states = []
  FRR_list = []
  FAR_list = []
  for thr in thresholds:
    TN, FN, FP, TP = eval_state(probs, labels, thr)
    if (FN + TP == 0):
      FRR = TPR = 1.0
      FAR = FP / float(FP + TN)
      TNR = TN / float(TN + FP)
    elif (FP + TN == 0):
      TNR = FAR = 1.0
      FRR = FN / float(FN + TP)
      TPR = TP / float(TP + FN)
    else:
      FAR = FP / float(FP + TN)
      FRR = FN / float(FN + TP)
      TNR = TN / float(TN + FP)
      TPR = TP / float(TP + FN)
    dist = math.fabs(FRR - FAR)
    FAR_list.append(FAR)
    FRR_list.append(FRR)
    if dist <= min_dist:
      min_dist = dist
      min_dist_states = [FAR, FRR, thr]
  EER = (min_dist_states[0] + min_dist_states[1]) / 2.0
  thr = min_dist_states[2]
  return EER, thr, FRR_list, FAR_list


def get_HTER_at_thr(probs, labels, thr):
  TN, FN, FP, TP = eval_state(probs, labels, thr)
  if (FN + TP == 0):
    FRR = 1.0
    FAR = FP / float(FP + TN)
  elif (FP + TN == 0):
    FAR = 1.0
    FRR = FN / float(FN + TP)
  else:
    FAR = FP / float(FP + TN)
    FRR = FN / float(FN + TP)
  HTER = (FAR + FRR) / 2.0
  return HTER



# ---------------------- Main function ----------------------
def main_metrics(prob, label, videoID):
    """
    The prob should be the signmoid output of the model
    all threee array should be numpy array
    """
    prob_dict = {}
    label_dict = {}

    # Group the prob and label by videoID
    for i in range(len(prob)):
        if (videoID[i] in prob_dict.keys()):
          prob_dict[videoID[i]].append(prob[i])
          label_dict[videoID[i]].append(label[i])

        else:
          prob_dict[videoID[i]] = []
          label_dict[videoID[i]] = []
          prob_dict[videoID[i]].append(prob[i])
          label_dict[videoID[i]].append(label[i])

    # Average the prob and label by videoID
    prob_list = []
    label_list = []
    for key in prob_dict.keys():
        avg_single_video_prob = sum(prob_dict[key]) / len(prob_dict[key])
        avg_single_video_label = sum(label_dict[key]) / len(label_dict[key])
        prob_list = np.append(prob_list, avg_single_video_prob)
        label_list = np.append(label_list, avg_single_video_label)

    # Compute the metrics
    auc_score = roc_auc_score(label_list, prob_list)
    cur_EER_valid, threshold, _, _ = get_EER_states(prob_list, label_list)
    ACC_threshold = calculate_threshold(prob_list, label_list, threshold)
    cur_HTER_valid = get_HTER_at_thr(prob_list, label_list, threshold)

    fpr, tpr, thr = roc_curve(label_list, prob_list)
    tpr_filtered = tpr[fpr <= 1 / 100]
    if len(tpr_filtered) == 0:
        rate = 0
    else:
        rate = tpr_filtered[-1]
    
    # Return the metrics (HTER, AUC, TPR@FPR=1/100)
    return cur_HTER_valid, auc_score, rate
# ---------------------- Main function ----------------------

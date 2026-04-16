import pickle 
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import sys
import matplotlib.pyplot as plt
#import seaborn as sns
import os

def load_validate_dump(pkl_name, pkl_dir, verbose=False, confidence_threshold=0.5):
    #print(pkl_dir + pkl_name)
    with open(pkl_dir + pkl_name, "rb") as pkl_file:
        evaluate_dict = pickle.load(pkl_file)
        target_all = evaluate_dict['target_all']
        pred_score_all = evaluate_dict['pred_score_all']
        
        if verbose: 
            print("Working On:", pkl_name )
            pred_label = (pred_score_all>confidence_threshold)
            print("confusion_matrix")
            print( confusion_matrix(target_all, pred_label))

    return pred_score_all, target_all

def compute_rpfa_conf(target,pred):
    
    conf_mat = confusion_matrix(target,pred)
    
    # Recall - TP/(TP+FN)
    recall = conf_mat[1,1]/(conf_mat[1,1]+conf_mat[1,0]) 
    
    #Precision - TP/(TP+FP)
    precision = conf_mat[1,1]/(conf_mat[1,1]+conf_mat[0,1]) 
    
    #FPR - FP/(FP+TN)
    FPR = conf_mat[0,1]/(conf_mat[0,1]+conf_mat[0,0]) 
    
    #Accuracy - (TP+TN)/sum
    accuracy = (conf_mat[1,1]+conf_mat[0,0])/np.sum(conf_mat) 
    
    return (recall,precision,FPR,accuracy)


def generate_group_eval_plots(SNR_list, grp_size_list, base_name, base_name_resume, pkl_file_name, graph_name, confidence_threshold=0.5):

    for grp_size in grp_size_list:

        recall_list, recall_list_resume = [], []
        FPR_list, FPR_list_resume = [], []
        Valid_acc_list, Valid_acc_list_resume = [], []
        min_acc, max_acc = 0, 100
        min_fpr, max_fpr = 0, 10

        for snr in SNR_list:

            base_name_grp = base_name.format(grp_size, abs(snr))
            base_name_resume_grp = base_name_resume.format(grp_size, abs(snr))
            
            if not os.path.exists(base_name_grp) or not os.path.exists(base_name_resume_grp):
                print("Path does not exist: ", base_name_grp, base_name_resume_grp)
                continue

            print("Loading from: ", base_name_grp)
            gt_score, gt_target = load_validate_dump(pkl_dir=base_name_grp, pkl_name=pkl_file_name, verbose=True, confidence_threshold=confidence_threshold)
        
            print("Loading from: ", base_name_resume_grp)
            gt_score_resume, gt_target_resume = load_validate_dump(pkl_dir=base_name_resume_grp, pkl_name=pkl_file_name, verbose=True, confidence_threshold=confidence_threshold)

            recall, _, FPR, accuracy = compute_rpfa_conf(gt_target, gt_score>confidence_threshold)
            recall_resume, _, FPR_resume, accuracy_resume = compute_rpfa_conf(gt_target_resume, gt_score_resume>confidence_threshold)
            print(f"Group Size: {grp_size}, SNR: {snr} dB --> Recall: {recall*100:.2f}%, FPR: {FPR*100:.2f}%, Accuracy: {accuracy*100:.2f}%")
            print(f"Group Size: {grp_size}, SNR: {snr} dB (Resume) --> Recall: {recall_resume*100:.2f}%, FPR: {FPR_resume*100:.2f}%, Accuracy: {accuracy_resume*100:.2f}%")

            recall_list.append(round(recall*100, 2))
            recall_list_resume.append(round(recall_resume*100, 2))
            FPR_list.append(round(FPR*100, 2))
            FPR_list_resume.append(round(FPR_resume*100, 2))
            Valid_acc_list.append(round(accuracy*100, 2))
            Valid_acc_list_resume.append(round(accuracy_resume*100, 2))
            print("\n")
        
        fig, ax = plt.subplots(1,3, figsize=(15, 5))
        ax[0].plot(SNR_list, Valid_acc_list, marker='o', label = 'First-Cut')
        ax[0].plot(SNR_list, Valid_acc_list_resume, marker='s', label = 'Retrained')
        if min(Valid_acc_list) < min(Valid_acc_list_resume):
            min_acc = min(Valid_acc_list)
        else:
            min_acc = min(Valid_acc_list_resume)
        if max(Valid_acc_list) > max(Valid_acc_list_resume):
            max_acc = max(Valid_acc_list)
        else:
            max_acc = max(Valid_acc_list_resume)
        ax[0].set_ylim(round(min_acc) - 2, round(max_acc) + 1)
        ax[0].grid(True)
        ax[0].set_yticks(np.arange(round(min_acc) - 2, round(max_acc) + 2, 1))
        ax[0].set_xticks(np.arange(-15, -5, 1))
        ax[0].set_xlabel('SNR (dB)')
        ax[0].set_ylabel('Accuracy (%)')
        # plt.plot(SNR_list, FPR, marker='s', label='False Positive Rate (%)')
        ax[0].set_title('Accuracy vs SNR')
        ax[0].legend()

        ax[1].plot(SNR_list, FPR_list, marker='o', label = 'First-Cut')
        ax[1].plot(SNR_list, FPR_list_resume, marker='s', label = 'Retrained')
        if min(FPR_list) < min(FPR_list_resume):
            min_fpr = min(FPR_list)
        else:
            min_fpr = min(FPR_list_resume)
        if max(FPR_list) > max(FPR_list_resume):
            max_fpr = max(FPR_list)
        else:
            max_fpr = max(FPR_list_resume)
        ax[1].set_ylim(round(min_fpr), round(max_fpr) + 1)
        ax[1].set_yticks(np.arange(round(min_fpr)-1, round(max_fpr) + 3, 2))
        ax[1].set_xticks(np.arange(-15, -5, 1))
        ax[1].grid(True)
        ax[1].set_xlabel('SNR (dB)')
        ax[1].set_ylabel('False Positive Rate (%)')
        ax[1].set_title('False Positive Rate vs SNR')
        ax[1].legend()

        ax[2].bar(SNR_list, recall_list, width=0.4, edgecolor='black', label='First-Cut', align='center')
        ax[2].bar(SNR_list, recall_list_resume, width=0.4, edgecolor='black', label='Retrained', align='edge')
        ax[2].set_ylim(20, 120)
        ax[2].grid(True, axis='y')
        ax[2].set_xlabel('SNR (dB)')
        ax[2].set_yticks(np.arange(20, 110, 10))
        ax[2].set_xticks(np.arange(-15, -5, 1))
        ax[2].set_ylabel('Recall (%)')
        ax[2].set_title('Recall vs SNR')
        ax[2].legend(loc='upper right')

        if grp_size == 0:
            fig.suptitle('ITIT: (Grp Size=1)', fontsize=16, fontweight='bold')
        else:
            fig.suptitle(f'GTGT-FM: (Grp Size={grp_size+1})', fontsize=16, fontweight='bold')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(graph_name.format(grp_size+1))


if __name__ == "__main__":

    # TOTAL_VAL_IMAGES = 48800
    # group_size = 8
    SNR_list = np.arange(-15, -6, 1)
    grp_size_list =  [7]#0 #[3, 7, 15]  # 0 implies group size of 1
    confidence_threshold = 0.5
    # total_groups = int(TOTAL_VAL_IMAGES/group_size)


    pkl_file_name = "model_validate_dump.pkl"

    # NGT_k7_Alg2_batchrand_SNR_neg7_train
    # NGT_k15_Alg2_Alt_phase_new_SNR_neg12_resume
    # NGT_k7_Alg2_batchrand_SNR_neg7_train_resume
    # NGT_k7_Alg2_Alt_phase_batchrand_SNR_neg7_train
    base_name= "./Validate/NGT_k{}_Alg2_batchrand_SNR_neg{}_train/"
    #"./Trained_Models/NGT_k{}_Alg2_new_SNR_neg{}/"
    base_name_resume = "./Validate/NGT_k{}_Alg2_batchrand_SNR_neg{}_train_resume/"

    graph_name = "./Train_data_Visual_Rep_k{}_no_phase.png"
    # base_name = base_name.format(0, 15)
    # base_name_resume = base_name_resume.format(0, 15)
    generate_group_eval_plots(SNR_list, grp_size_list, base_name, base_name_resume, pkl_file_name, graph_name, confidence_threshold=confidence_threshold)

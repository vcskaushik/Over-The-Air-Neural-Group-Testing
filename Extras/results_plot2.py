import pickle 
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import sys

#ROOT_DIR = 'plotout/'
GROUP_TESTING_DATASET_PATH = 'data/GroupTestingDataset'

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
    
    
    
    

def main_analysis(pkl_name_k0,pkl_dir_k0,pkl_name,pkl_dir,group_size):
    
    ##################################
    # Individual Testing Baseline; K=0
    ##################################

    print("##################################")
    print("[ Individual Testing K0 Baseline ]")
    K0_score, K0_target = load_validate_dump(pkl_name=pkl_name_k0,pkl_dir = pkl_dir_k0 , verbose=True)
        
    K0_recall = 2 * np.sum( np.logical_and(K0_target, K0_score>0.5) ) 
    K0_FPR = 100 * np.sum(   np.logical_and(K0_target==0, K0_score>0.5) ) / np.sum(K0_target==0) # False Positive Rate 
    K0_acc = 100 * np.sum(np.logical_not(np.logical_xor(K0_target==1,K0_score>0.5)))/K0_target.size
    print("Recall(%): {} FPR(%): {:3f}  Acc(%): {}".format(K0_recall, K0_FPR, K0_acc))
    print(" No of negatives ",np.sum(K0_target==0) , " No of positives ", np.sum(K0_target==1) )
    
    K0_recall2,K0_precision2,K0_FPR2,K0_acc2 =  compute_rpfa_conf(K0_target,K0_score>0.5)
    
    print("version 2 Recall(%): {} FPR(%): {:3f}  Acc(%): {}".format(K0_recall2*100, K0_FPR2*100, K0_acc2*100))
    
    K0_tests = len(K0_target)
    # print("Number of Tests (1st Round): ", K0_tests)
    each_K0_GigaMACs = 16.5 # 16.5 GMacs per test 
    K0_MACs = each_K0_GigaMACs /1000 * K0_tests # TMacs 10^12
    print("Number of Tests (1st Round): ", K0_tests)
    print("Total Computation: {:.1f} TMACs".format(K0_MACs))
    
    K0G1_result_dict = {
            'method_score': K0_score, # raw outputs 
            'method_target': K0_target, # raw outputs 
            'method_recall': K0_recall, # performance metrics 
            'method_FPR': K0_FPR, # performance metrics
            'method_acc': K0_acc, # performance metrics
            'method_tests': K0_tests, # computation cost metrics
            'method_TeraMACs_total': K0_MACs, # computation cost metrics
        }

    del K0_score, K0_target, each_K0_GigaMACs
    
    ##################################
    # note that  each_K0_GigaMACs and K0_score would be re-used by downstream modules 
    ##################################
    
    if group_size == 1:
        each_method_GigaMACs=16.5
    elif group_size == 2:
        each_method_GigaMACs=20.16
    elif group_size == 4:
        each_method_GigaMACs=27.46
    elif group_size == 8:
        each_method_GigaMACs=42.06
    elif group_size == 16:
        each_method_GigaMACs=71.27
    elif group_size == 32:
        each_method_GigaMACs=129.68
    else:
        each_method_GigaMACs=0
        
    
    ##################################
    # Algorithm 1 Wrapper Function
    ##################################
    def algorithm_1_wrapper(
        pkl_name:str, 
        exp_title:str, 
        each_method_GigaMACs:float, # GMacs per test. M images in total. 
        group_size:int, # M value in the paper 
        pkl_dir:str, # default root dir 
        pkl_dir_k0:str, # default root dir 
        pkl_name_k0:str,
        confidence_threshold:float=0.5,
        ):
        print("##################################")
        print(exp_title)
        method_score, method_target = load_validate_dump(pkl_dir=pkl_dir, pkl_name=pkl_name, verbose=True, confidence_threshold=confidence_threshold)
        
        group_recall2,group_precision2,group_FPR2,group_acc2 =  compute_rpfa_conf(method_target,method_score>confidence_threshold)
    
        print("version 2 Group Recall(%): {} FPR(%): {:3f}  Acc(%): {}".format(group_recall2*100, group_FPR2*100, group_acc2*100))
        
        method_tests_Round_1 = len(method_target)
        method_TeraMACs_Round_1 = each_method_GigaMACs / 1000 * method_tests_Round_1 # TMacs 10^12
        print("Number of Tests (1st Round): ", method_tests_Round_1, "\t Computation: {:.1f} TMACs".format(method_TeraMACs_Round_1))

        method_Round_1_next = np.repeat( (method_score>confidence_threshold), group_size) # times group size 

        print("Number Of Samples After the 1st round:", np.sum(method_Round_1_next))
        
        K0_score, K0_target = load_validate_dump(pkl_dir=pkl_dir_k0, pkl_name=pkl_name_k0, verbose=False)
        
        group_check = np.sum(K0_target[:method_Round_1_next.shape[0]].reshape(-1,group_size),1)>0
        if np.sum(group_check!=method_target)==0:
            K0_score = K0_score[:method_Round_1_next.shape[0]]
            K0_target = K0_target[:method_Round_1_next.shape[0]]
        else:
            print("Values dont match")
        #
        #print(group_check.shape,method_target.shape)
        #print(np.sum(group_check!=method_target))
        
        print(K0_score.shape,method_Round_1_next.shape)
        method_recall = 100 * np.sum( np.logical_and(
            np.logical_and(K0_target, K0_score>0.5), 
            method_Round_1_next) ) / np.sum(K0_target==1) # use K0 model as the second round 
        method_FPR = 100 * np.sum(   np.logical_and(
            np.logical_and(K0_target==0, K0_score>0.5), 
            method_Round_1_next) 
            ) / np.sum(K0_target==0) # False Positive Rate 
        
        method_acc = 100 * np.sum(np.logical_not(np.logical_xor(K0_target==1,np.logical_and(K0_score>0.5,method_Round_1_next))))/K0_target.size
        
        print("Recall(%): {} FPR(%): {:3f}  Acc(%): {}".format(method_recall, method_FPR, method_acc))
        
        method_pred = np.logical_and(K0_score>0.5,method_Round_1_next)
        method_recall2,method_precision2,method_FPR2,method_acc2 =  compute_rpfa_conf(K0_target.flatten(),method_pred.flatten())
        
        print("version 2 Group Recall(%): {} FPR(%): {:3f}  Acc(%): {}".format(method_recall2*100, method_FPR2*100, method_acc2*100))

        method_tests_Round_2 = np.sum(method_Round_1_next) 
        each_K0_GigaMACs = 16.5 # 16.5 GMacs per test, same as the baseline model 
        method_TeraMACs_Round_2 = each_K0_GigaMACs / 1000 * method_tests_Round_2 # TMacs 10^12
        print("Number of Tests (2nd Round): ", method_tests_Round_2, "\t Computation: {:.1f} TMACs".format(method_TeraMACs_Round_2))

        method_TeraMACs_total = method_TeraMACs_Round_1 + method_TeraMACs_Round_2
        method_tests_total = method_tests_Round_1 + method_tests_Round_2
        print("Total Computation: {:.1f} TeraMACs".format(method_TeraMACs_total), "Total Tests:", method_tests_total, "Relative Cost", method_TeraMACs_total/805.2)

        result_dict = {
            'method_score': method_score, # raw outputs 
            'method_target': method_target, # raw outputs 
            'method_recall': method_recall, # performance metrics 
            'method_FPR': method_FPR, # performance metrics
            'method_acc': method_acc, # performance metrics
            'method_tests_Round_1': method_tests_Round_1, # computation cost metrics
            'method_tests_Round_2': method_tests_Round_2, # computation cost metrics
            'method_TeraMACs_Round_1': method_TeraMACs_Round_1, # computation cost metrics
            'method_TeraMACs_Round_2': method_TeraMACs_Round_2, # computation cost metrics
            'method_TeraMACs_total': method_TeraMACs_total, # computation cost metrics
        }
        return result_dict


    ##################################
    # Now We Explore Design 2. And group size 8, 16. And potentially Algorithm 2. 
    # Start with Design 2 + Algorithm 1 + Group Size  + LayerGroup 1/2 
    ##################################
    
    

    ##################################
    # Try G2 K=1
    ##################################
    K3G4_result_dict = algorithm_1_wrapper(
        pkl_name=pkl_name, 
        pkl_dir = pkl_dir,
        pkl_name_k0=pkl_name_k0, 
        pkl_dir_k0 = pkl_dir_k0,
        exp_title='K=3 + Design 2 (G2) + Algorithm 1 Two-Round', 
        each_method_GigaMACs=each_method_GigaMACs, 
        group_size=group_size,
        )

    
    
    return K0G1_result_dict , K3G4_result_dict

if __name__ == '__main__':


    ##################################
    # Individual Testing Baseline; K=0
    ##################################
    
    #Names_list = ["No Noise","SNR 0","SNR 2","SNR 5"]
    Names_list = ["No Noise","SNR 0","SNR 1","SNR 2","SNR 3","SNR 5"]
    
    group_size = 4 #int(sys.argv[1])
    K= group_size-1
    #A= int(sys.argv[2])
    #base_type = int(sys.argv[3])
    
    
#     pkl_dirs_k0 =  ["Trained_Models/ResNeXt_K0_A1/","Trained_Models/ResNeXt_K0_A1_SNR0/","Trained_Models/ResNeXt_K0_A1_SNR1/", "Trained_Models/ResNeXt_K0_A1_SNR2/","Trained_Models/ResNeXt_K0_A1_SNR3/","Trained_Models/ResNeXt_K0_A1_SNR5/"]
    
    # print("K0 Alg is A2")
    # pkl_dirs_k0 =  ["Trained_Models/ResNeXt_K0_A2/","Trained_Models/ResNeXt_K0_A2_SNR0/","Trained_Models/ResNeXt_K0_A2_SNR1/", "Trained_Models/ResNeXt_K0_A2_SNR2/","Trained_Models/ResNeXt_K0_A2_SNR3/","Trained_Models/ResNeXt_K0_A2_SNR5/"]
    #pkl_dirs_k0 = ["Trained_Models/ResNeXt_K0_A1/","Trained_Models/ResNeXt_K0_A1_SNR0/", "Trained_Models/ResNeXt_K0_A1_SNR2/","Trained_Models/ResNeXt_K0_A1_SNR5/"]
    pkl_name_k0 = "model_validate_dump.pkl"
    pkl_dirs_k0 =  ["./VAL_DEBUG_P0.01pct/"]
    
    pkl_name = "model_validate_dump.pkl"
    pkl_dirs = ["./VAL_DEBUG_PK3_0.01pct/"]
    
    
    
    #SNR_list = [0,2,5]
    # SNR_list =[0,1,2,3,5]
    # base_name = "Trained_Models/ResNeXt_K{}_A{}/"
    
    # if base_type == 2:
    #     base_name2 = "Trained_Models/ResNeXt_K{}_A{}_SNR{}_2/"
    # else:
    #     base_name2 = "Trained_Models/ResNeXt_K{}_A{}_SNR{}/"
        
    # if (A==3) or (A==5):
    #     pkl_dirs = [base_name.format(K,2)]
    # else:
    #     pkl_dirs = [base_name.format(K,A)]
        
    # pkl_dirs += [base_name2.format(K,A,snr_val) for snr_val in SNR_list]
    
    

    # Schemes_list = ["ITIT","GTGT"]
    
    # Ns = 224*224*3
    # Nf = 512*28*28
    
    
    
    # df = {"Noise level":[],"Scheme":[],"Recall":[],"FPR":[],"Acc":[],"Round 1 Tests":[],"Total Tests":[],"Round 1 TMACs":[],"Total TMACs":[],"Comp Relative to ITIT (%)":[],"Total Channel Uses":[],"Channel Uses Relative to ITIT (%) ":[]}
    
    for i in range(len(pkl_dirs_k0)):
        # print("**********************",Names_list[i],"**********************")
        
        K0G1_result_dict , K3G4_result_dict = main_analysis(pkl_name_k0=pkl_name_k0,pkl_dir_k0 = pkl_dirs_k0[i],pkl_name = pkl_name, pkl_dir = pkl_dirs[i],group_size=group_size)
        
    #     df["Noise level"] += [Names_list[i]]*2
    #     df["Scheme"] += Schemes_list
    #     df["Recall"] += [K0G1_result_dict['method_recall'] , K3G4_result_dict['method_recall']]
    #     df["FPR"] += [K0G1_result_dict['method_FPR'] , K3G4_result_dict['method_FPR']]
    #     df["Acc"] += [K0G1_result_dict['method_acc'] , K3G4_result_dict['method_acc']]
    #     df["Round 1 Tests"] += [0, K3G4_result_dict['method_tests_Round_1']]
    #     df["Total Tests"] += [K0G1_result_dict['method_tests'] , K3G4_result_dict['method_tests_Round_1'] + K3G4_result_dict['method_tests_Round_2'] ]
    #     df["Round 1 TMACs"] += [0 , K3G4_result_dict['method_TeraMACs_Round_1']]
    #     df["Total TMACs"] += [K0G1_result_dict['method_TeraMACs_total'] , K3G4_result_dict['method_TeraMACs_total']]
        
    #     df["Comp Relative to ITIT (%)"] += [100, (K3G4_result_dict['method_TeraMACs_total']/K0G1_result_dict['method_TeraMACs_total']) *100]
        
    #     K0_channel_use = Ns*K0G1_result_dict['method_tests']
        
    #     if (A == 2):
    #         K3_channel_use = Nf*K3G4_result_dict['method_tests_Round_1'] + Ns*K3G4_result_dict['method_tests_Round_2']
    #     elif (A == 4):
    #         K3_channel_use = Ns*K3G4_result_dict['method_tests_Round_1'] + Ns*K3G4_result_dict['method_tests_Round_2']
    #     else:
    #         K3_channel_use = K0_channel_use
            
            
    #     df["Total Channel Uses"] += [K0_channel_use,K3_channel_use]
    #     df["Channel Uses Relative to ITIT (%) "] += [100,(K3_channel_use/K0_channel_use)*100]
        
        
        
    #     print("********************************************")
    
    # df = pd.DataFrame(df)
    
    # if base_type == 2:
    #     df.to_csv("Final_result_K{}_A{}_3.csv".format(K,A))
    # else:
    #     df.to_csv("Final_result_K{}_A{}.csv".format(K,A))
        

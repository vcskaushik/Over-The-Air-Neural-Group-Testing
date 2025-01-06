import pickle 
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import sys

#ROOT_DIR = 'plotout/'
GROUP_TESTING_DATASET_PATH = 'data/GroupTestingDataset'
TOTAL_VAL_IMAGES = 48800 # Divisible by all group sizes considered 

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
    
    
    
    

def main_analysis(pkl_name_k0,pkl_dir_k0,pkl_name_list,pkl_dir_list,group_size_list,title_list):
    
    ## Precomputed GMACS
    each_GSz_GigaMACs ={1: 16.5 ,2: 20.16 ,4: 27.46 ,8: 42.06 ,16: 71.27 ,32: 129.68} 
    
    
    ##################################
    # Individual Testing Baseline; K=0
    ##################################

    
    
    K0_score, K0_target = load_validate_dump(pkl_name=pkl_name_k0,pkl_dir = pkl_dir_k0 , verbose=True)
    K0_score = K0_score[:TOTAL_VAL_IMAGES] # divisible by all group sizes
    K0_target = K0_target[:TOTAL_VAL_IMAGES] # divisible by all group sizes
    K0_recall,K0_precision,K0_FPR,K0_acc =  compute_rpfa_conf(K0_target,K0_score>0.5)
    K0_tests = len(K0_target)
    K0_MACs = each_GSz_GigaMACs[1] /1000 * K0_tests # TMacs 10^12
    
    
    print("##################################")
    print("[ Individual Testing K0 Baseline ]")
    print(f'No of negatives {np.sum(K0_target==0)} , No of positives, {np.sum(K0_target==1)}')
    print(f'Recall(%): {K0_recall*100} FPR(%): {K0_FPR*100 :.3f}  Acc(%): {K0_acc*100}')
    print(f'Number of Tests (1st Round): {K0_tests}')
    print(f'Total Computation: {K0_MACs:.1f} TMACs')
    
    K0G1_result_dict = {
            "itit_score": K0_score, # raw outputs 
            "itit_target": K0_target, # raw outputs 
            "Group Tests": 0,
            "Recall": K0_recall*100, # performance metrics 
            "FPR": K0_FPR*100, # performance metrics
            "Accuracy": K0_acc*100, # performance metrics
            "Total Tests": K0_tests/TOTAL_VAL_IMAGES, # computation cost metrics
            "Total GMACs": K0_MACs*1000/TOTAL_VAL_IMAGES, # computation cost metrics
        }
    
    ##################################
    # note that  each_K0_GigaMACs and K0_score would be re-used by downstream modules 
    ##################################
    
#     if group_size == 1:
#         each_method_GigaMACs=16.5
#     elif group_size == 2:
#         each_method_GigaMACs=20.16
#     elif group_size == 4:
#         each_method_GigaMACs=27.46
#     elif group_size == 8:
#         each_method_GigaMACs=42.06
#     elif group_size == 16:
#         each_method_GigaMACs=71.27
#     elif group_size == 32:
#         each_method_GigaMACs=129.68
#     else:
#         each_method_GigaMACs=0
        
    
    ##################################
    # Algorithm 1 Wrapper Function
    ##################################
    def algorithm_1_wrapper(
        pkl_name:str, 
        exp_title:str, 
        each_GSz_GigaMACs, # GMacs per test. M images in total. 
        group_size:int, # M value in the paper 
        pkl_dir:str, # default root dir 
        pkl_dir_k0:str, # default root dir 
        pkl_name_k0:str,
        confidence_threshold:float=0.5,
        ):
        
        print("##################################")
        print(exp_title)
        
        # Limit groups to total images (48800)/group size
        total_groups = int(TOTAL_VAL_IMAGES/group_size)
        
        ### Compute Round 1 Group Test results 
        gt_score, gt_target = load_validate_dump(pkl_dir=pkl_dir, pkl_name=pkl_name, verbose=True, confidence_threshold=confidence_threshold)
        
        gt_score = gt_score[:total_groups]
        gt_target = gt_target[:total_groups]
        
        group_recall,group_precision,group_FPR,group_acc =  compute_rpfa_conf(gt_target,gt_score>confidence_threshold)
        num_of_group_tests = len(gt_target)
        gt_TeraMACs = each_GSz_GigaMACs[group_size] / 1000 * num_of_group_tests 
        
        print(f"version 2 Group Recall(%): {group_recall*100} FPR(%): {group_FPR*100:3f}  Acc(%): {group_acc*100}")
        print("Number of Group Tests (1st Round): ", num_of_group_tests, "\t Computation: {:.1f} TMACs".format(gt_TeraMACs))
        
        ### Compute Round 1 (Group) + Round 2(Individual) Test results
        individual_gt_results = np.repeat( (gt_score>confidence_threshold), group_size) # times group size 
        K0_score, K0_target = load_validate_dump(pkl_dir=pkl_dir_k0, pkl_name=pkl_name_k0, verbose=False)
        print("Number Of Samples After the 1st round:", np.sum(individual_gt_results))
        
        
        
        group_check = np.sum(K0_target[:individual_gt_results.shape[0]].reshape(-1,group_size),1)>0
        if np.sum(group_check!=gt_target)==0:
            K0_score = K0_score[:individual_gt_results.shape[0]]
            K0_target = K0_target[:individual_gt_results.shape[0]]
        else:
            print("Values dont match")
        
        print(K0_score.shape,individual_gt_results.shape)
        
        final_pred = np.logical_and(K0_score>0.5,individual_gt_results).flatten()
        final_recall,final_precision,final_FPR,final_acc =  compute_rpfa_conf(K0_target.flatten(), final_pred)
        
        print(f"version 2 Group Recall(%): {final_recall*100} FPR(%): {final_FPR*100 :3f}  Acc(%): {final_acc*100}")

        num_of_individual_tests = np.sum(individual_gt_results)  
        it_TeraMACs = each_GSz_GigaMACs[1] / 1000 * num_of_individual_tests # TMacs 10^12
        print(f"Number of Individual Tests (2nd Round): {num_of_individual_tests}, \t Computation: {it_TeraMACs :.1f} TMACs")

        TeraMACs_total = gt_TeraMACs + it_TeraMACs
        tests_total = num_of_group_tests + num_of_individual_tests
        print("Total Computation: {:.1f} TeraMACs".format(TeraMACs_total), "Total Tests:", tests_total, "Relative Cost", TeraMACs_total/805.2)

        result_dict = {
            'gt_score': gt_score, # raw outputs 
            'gt_target': gt_target, # raw outputs 
            'Recall': final_recall*100, # performance metrics 
            'FPR': final_FPR*100, # performance metrics
            'Accuracy': final_acc*100, # performance metrics
            'Group Tests': num_of_group_tests/TOTAL_VAL_IMAGES, # computation cost metrics
            'Total Tests': tests_total/TOTAL_VAL_IMAGES, # computation cost metrics
            'gt_TeraMACs': gt_TeraMACs, # computation cost metrics
            'it_TeraMACs': it_TeraMACs, # computation cost metrics
            'Total GMACs': TeraMACs_total*1000/TOTAL_VAL_IMAGES, # computation cost metrics
        }
        return result_dict


    ##################################
    # Now We Explore Design 2. And group size 8, 16. And potentially Algorithm 2. 
    # Start with Design 2 + Algorithm 1 + Group Size  + LayerGroup 1/2 
    ##################################
    
    

    ##################################
    # Try G2 K=1
    ##################################
    
    results_dict_list = [K0G1_result_dict]
    
    for i,title in enumerate(title_list):
            results_dict_list += [algorithm_1_wrapper(
            pkl_name = pkl_name_list[i], 
            pkl_dir = pkl_dir_list[i],
            pkl_name_k0 = pkl_name_k0, 
            pkl_dir_k0 = pkl_dir_k0,
            exp_title = title, 
            each_GSz_GigaMACs = each_GSz_GigaMACs, 
            group_size=group_size_list[i]
            )]

    
    
    return results_dict_list

def add_result(df_full,results_dict_list,SNR_name, Schemes_list,ch_use_funcs_list):
    
#     if df_full == None:
#         print(SNR_name,Schemes_list)
    df_keys = list(df_full.keys())
    ch_use_itit = 224*224*3
    TMACs_itit = 16.5/1000
    
    for i,results_dict in enumerate(results_dict_list):   
        for key in results_dict.keys():
            if key in df_keys:
                df_full[key] += [results_dict[key]]
        
        df_full["Noise level"] += [SNR_name]
        df_full["Scheme"] += [Schemes_list[i]]
        
        n_gt = df_full["Group Tests"][-1]
        n_it = df_full["Total Tests"][-1] - n_gt 
        df_full["Total Channel Uses"] += [ch_use_funcs_list[i](n_gt,n_it)]
        
        if i == 0:
            ch_use_itit = df_full["Total Channel Uses"][-1]
            TMACs_itit = df_full["Total GMACs"][-1]
        
        df_full["Channel Uses Relative to ITIT (%) "] += [(df_full["Total Channel Uses"][-1]*100)/ch_use_itit]
        df_full["GMACs Relative to ITIT (%)"] += [(df_full['Total GMACs'][-1]*100)/TMACs_itit]
    return df_full
        

if __name__ == '__main__':


    #####################################
    # Fixed Group Size (4) , All Schemes
    #####################################
    
    K = 7 #group_size-1
    
    SNR_Names_list = ["No Noise","SNR 0","SNR 1","SNR 2","SNR 3","SNR 5"]
    SNR_list =[None,0,1,2,3,5]
    
    Schemes_list = ["ITIT","ITGT-FM","GTGT-SM","GTGT-FM"]
    A_list = [1,3,4,2]
    
    base_name = "Trained_Models/ResNeXt_K{}_A{}/"
    base_name_noisy = "Trained_Models/ResNeXt_K{}_A{}_SNR{}/"
    base_name_noisy2 = "Trained_Models/ResNeXt_K{}_A{}_SNR{}_2/"
    
    # Channel use computation functions
    # Ns = 224*224*3
    # Nf = 512*28*28
    
    ch_use_A2 = lambda n_gt,n_it: (512*28*28)*(n_gt+n_it)
    ch_use_A4 = lambda n_gt,n_it: ((224*224*3)*n_gt + (512*28*28)*n_it)
    ch_use_A1 = lambda n_gt,n_it: (224*224*3)+0*(n_gt+n_it) 
    
    ch_use_funcs_list = [ch_use_A1,ch_use_A1,ch_use_A4,ch_use_A2]
    
    df_full_alg = {"Noise level":[],"Scheme":[],"Group Tests":[],"Total Tests":[],"Total Channel Uses":[],"Channel Uses Relative to ITIT (%) ":[],"Total GMACs":[],"GMACs Relative to ITIT (%)":[],"Recall":[],"FPR":[],"Accuracy":[]}
    
    print(df_full_alg.keys())
    ITIT_results_list = []
    for i_s,SNR in enumerate(SNR_list):
        
        # ************ Baseline Schemes ************
        
        # Individual Testing Pickles
        pkl_name_k0 = "model_validate_dump.pkl"
        if SNR is None:
            pkl_dir_k0 = base_name.format(0,1) 
        else:
            pkl_dir_k0 = base_name_noisy.format(0,1,SNR)

        # Scheme Specific Pickles
        
        
        if SNR is None:
            pkl_dir_list = [base_name.format(K,2)] # No noise ITGT-FM == No noise GTGT-FM
        else:
            pkl_dir_list = [base_name_noisy.format(K,A_list[1],SNR)]
        
        pkl_name_list = ["model_validate_dump.pkl"]
        group_size_list = [K+1]
        title_list = [Schemes_list[1]+" - "+SNR_Names_list[i_s]]
        
        results_dict_list = main_analysis(pkl_name_k0,pkl_dir_k0,pkl_name_list,pkl_dir_list,group_size_list,title_list)
        ITIT_results_list += [results_dict_list[0]]
        
        # ************ Proposed Schemes ************
        
        # Individual Testing Pickles A2
        pkl_name_k0 = "model_validate_dump.pkl"
        if SNR is None:
            pkl_dir_k0 = base_name.format(0,2)
        else:
            pkl_dir_k0 = base_name_noisy.format(0,2,SNR)

        # Scheme Specific Pickles
        if SNR is None:
            pkl_dir_list = [base_name.format(K,A) for A in A_list[2:]]
        else:
            pkl_dir_list = [base_name_noisy2.format(K,A,SNR) for A in A_list[2:]]
        
        pkl_name_list = ["model_validate_dump.pkl" for A in A_list[2:]]
        group_size_list = [K+1 for A in A_list[:2]]
        title_list = [scheme+" - "+SNR_Names_list[i_s] for scheme in Schemes_list[2:] ]
        
        results_dict_list2 = main_analysis(pkl_name_k0,pkl_dir_k0,pkl_name_list,pkl_dir_list,group_size_list,title_list)
        
        results_dict_list += results_dict_list2[1:]
        
        df_full_alg = add_result(df_full_alg,results_dict_list,SNR_Names_list[i_s], Schemes_list,ch_use_funcs_list)
    
        print("********************************************")
    print(df_full_alg)
    df_full_alg = pd.DataFrame(df_full_alg)
    
    df_full_alg.to_csv("Final_results_Algs2.csv")
    
    ##########################################
    # Fixed Scheme (GTGT-FM) , All Group Sizes
    ##########################################
    
    df_full_gsize = {"Noise level":[],"Scheme":[],"Group Tests":[],"Total Tests":[],"Total Channel Uses":[],"Channel Uses Relative to ITIT (%) ":[],"Total GMACs":[],"GMACs Relative to ITIT (%)":[],"Recall":[],"FPR":[],"Accuracy":[]}
    
    Schemes_list = ["ITIT","GTGT-FM (K=4)","GTGT-FM (K=8)","GTGT-FM (K=16)"]
    K_list = [3,7,15]
    ch_use_funcs_list = [ch_use_A1,ch_use_A2,ch_use_A2,ch_use_A2]
    A = 2
    for i_s,SNR in enumerate(SNR_list):
        
        results_dict_list = [ITIT_results_list[i_s]]
        
        # ************ Proposed Schemes ************
        
        # Individual Testing Pickles A2
        pkl_name_k0 = "model_validate_dump.pkl"
        if SNR is None:
            pkl_dir_k0 = base_name.format(0,2)
        else:
            pkl_dir_k0 = base_name_noisy.format(0,2,SNR)
            
        # Group Size Specific Pickles

        if SNR is None:
            pkl_dir_list = [base_name.format(K,A) for K in K_list]
        else:
            pkl_dir_list = [base_name_noisy2.format(K,A,SNR) for K in K_list]
        
        pkl_name_list = ["model_validate_dump.pkl" for K in K_list]
        group_size_list = [K+1 for K in K_list]
        title_list = [scheme+" - "+SNR_Names_list[i_s] for scheme in Schemes_list[1:]]
        
        
        results_dict_list2 = main_analysis(pkl_name_k0,pkl_dir_k0,pkl_name_list,pkl_dir_list,group_size_list,title_list)
        
        results_dict_list += results_dict_list2[1:]
        
        df_full_gsize = add_result(df_full_gsize,results_dict_list,SNR_Names_list[i_s], Schemes_list,ch_use_funcs_list)
        
    df_full_gsize = pd.DataFrame(df_full_gsize)
    
    df_full_gsize.to_csv("Final_results_Gsize2.csv")
        
            
        
        
        

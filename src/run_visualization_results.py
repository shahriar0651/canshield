from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from dataset.load_dataset import *
from hydra.utils import get_original_cwd
from testing import *
from training import *
from visualize import *

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from sklearn.metrics import roc_auc_score, roc_curve, auc

from os.path import exists as file_exists  
from sklearn.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import auc

def get_conf_matric(y_true, y_pred):
    try:
        CM = confusion_matrix(y_true, y_pred, normalize = 'true')
        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]
        FPR = FP/(FP+TN)
        TPR = TP/(TP+FN)
        return TPR, FPR
    except:
        return 0.00, 0.00


def plot_auroc_curve(args, fprs, tprs, fig_name):
    # Create ROC curve
    root_dir = args.root_dir
    dataset_name = args.dataset_name
    auc_score = auc(fprs, tprs)
    attack_name = fig_name.split("_")[-1]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # Plot AUROC curve
    ax.plot(fprs, tprs, label=f"AUROC = {auc_score:.3f}", color='blue', linewidth=2)

    # Plot random baseline
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Baseline")

    # Labels and title
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f"{attack_name} Attack")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6, linewidth=0.5)

    plt.tight_layout()

    # Save figure
    fig_dir = Path(f"{root_dir}/../plots/{dataset_name}/auroc_curves/{fig_name}")
    fig_dir.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(f"{fig_dir}.jpg", dpi=250)
    plt.savefig(f"{fig_dir}.pdf", dpi=250)
    plt.close(fig)


def get_canshield_confusion_mat(args, heatmap_auc_dict, prediction_df_final):
    sampling_periods = args.sampling_periods
    attacks_dict = args.attacks_dict
    time_step =args.time_step
    loss_factor = args.loss_factor
    time_factor = args.time_factor
    dataset_name = args.dataset_name
    root_dir = args.root_dir
    eval_type = args.eval_type
    per_of_samples = args.per_of_samples


    try:
        baseline_cm_df = pd.read_csv(f"{root_dir}/../data/results/{dataset_name}/baseline_cm.csv")
        baseline_cm_df.index = baseline_cm_df['Model']
    except FileNotFoundError as e:
        print(e)
        print("Add baseline data to")
        print(f"{root_dir}/../data/results/{dataset_name}/baseline_cm.csv")
        columns = ['Model','Plateau_tpr','Plateau_fpr','Continuous_tpr','Continuous_fpr','Playback_tpr','Playback_fpr','Flooding_tpr','Flooding_fpr','Suppress_tpr','Suppress_fpr']
        baseline_cm_df = pd.DataFrame([['CANet',0.955,0.025,0.765,0.006,0.905,0.004,0.901,0.004,0.613,0.004],
            ['Predictive',0.33,0.026,0.015,0.006,0.02,0.004,0.644,0.006,0.003,0.007],
            ['Autoencoder',0.361,0.074,0.016,0.025,0.029,0.005,0.688,0.005,0.001,0.007]], columns = columns)


    max_fpr_th = 0.01
    # sampling_periods = [1, 5, 10]
    
    canshield_cm = {}
    for sampling_period in sampling_periods:
        canshield_cm[f"CANShield-{sampling_period}"] = {}
        canshield_cm["CANShield-Base"] = {}
        canshield_cm["CANShield-Ens"] = {}
    
    for file_name in attacks_dict.keys():
        ensemble_data = 0
        prediction_df_final_file = prediction_df_final[prediction_df_final['Attack'] == file_name].copy() #[['Label', 'Loss']].values

        for sampling_period in sampling_periods:
            prediction_df_final_ind = prediction_df_final_file[prediction_df_final_file['Sampling Period'] == sampling_period].copy() #[['Label', 'Loss']].values
            
            if sampling_period == 1:
                y_true = prediction_df_final_ind['Label']
                y_pred = prediction_df_final_ind['Prediction']
                fprs, tprs, thresholds = roc_curve(y_true, y_pred)
                optimal_threshold = thresholds[fprs<max_fpr_th][-1]
                y_pred = (y_pred >= optimal_threshold).astype(int)
                tpr, fpr = get_conf_matric(y_true, y_pred)
                canshield_cm["CANShield-Base"][f'{file_name}_tpr'] = tpr
                canshield_cm["CANShield-Base"][f'{file_name}_fpr'] = fpr
                plot_auroc_curve(args, fprs, tprs, f"CANShield-Base_{dataset_name}_{sampling_period}_{file_name}")

            y_true = prediction_df_final_ind['Label']
            y_pred = prediction_df_final_ind['Loss']
            fprs, tprs, thresholds = roc_curve(y_true, y_pred)
            optimal_threshold = thresholds[fprs<max_fpr_th][-1]
            y_pred = (y_pred >= optimal_threshold).astype(int)
            tpr, fpr = get_conf_matric(y_true, y_pred)
            canshield_cm[f"CANShield-{sampling_period}"][f'{file_name}_tpr'] = tpr
            canshield_cm[f"CANShield-{sampling_period}"][f'{file_name}_fpr'] = fpr
            plot_auroc_curve(args, fprs, tprs, f"CANShield-{sampling_period}_{dataset_name}_{sampling_period}_{file_name}")
            # Get canshield ensemble data
            # ensemble_data += prediction_df_final_ind[['Label', 'Loss']].values
            samp_data = prediction_df_final_ind[['Label', 'Loss']].values
            try:
                ensemble_data = ensemble_data + samp_data
            except:
                a = len(ensemble_data)
                b = len(samp_data)
                if a > b:
                    ensemble_data = ensemble_data[:b] + samp_data
                else:
                    ensemble_data = ensemble_data + samp_data[:a]

        ensemble_data /= len(sampling_periods)
        y_true = (ensemble_data[:,0]>=0.50).astype(int)
        y_pred = ensemble_data[:,1]
        fprs, tprs, thresholds = roc_curve(y_true, y_pred)
        optimal_threshold = thresholds[fprs<max_fpr_th][-1]
        y_pred = (y_pred >= optimal_threshold).astype(int)
        tpr, fpr = get_conf_matric(y_true, y_pred)
        canshield_cm["CANShield-Ens"][f'{file_name}_tpr'] = tpr
        canshield_cm["CANShield-Ens"][f'{file_name}_fpr'] = fpr
        plot_auroc_curve(args, fprs, tprs, f"CANShield-Ens_{dataset_name}_{file_name}")


    canshield_cm_df = np.round(pd.DataFrame(canshield_cm).T,3)
    canshield_cm_df['Model'] = canshield_cm_df.index
    canshield_cm_total = pd.concat([canshield_cm_df, baseline_cm_df], ignore_index= True, axis = 0)
    # canshield_cm_total.to_csv(f"{root_dir}/../data/results/{dataset_name}/canshield_cm_total.csv")

    for file_name in attacks_dict.keys():
        tpr_data = canshield_cm_total[f"{file_name}_tpr"].values
        fpr_data = canshield_cm_total[f"{file_name}_fpr"].values
        canshield_cm_total[f"{file_name}"] = [f"{np.round(tpr,3)} / {np.round(fpr,3)}" for tpr, fpr in zip(tpr_data, fpr_data)]
    canshield_cm_total.to_csv(f"{root_dir}/../data/results/{dataset_name}/canshield_{dataset_name}_{eval_type}_{per_of_samples}_cm_total_merged.csv")
    
def get_canshield_baselines(args, heatmap_auc_dict, prediction_df_final):
    sampling_periods = args.sampling_periods
    attacks_dict = args.attacks_dict
    time_step =args.time_step
    loss_factor = args.loss_factor
    time_factor = args.time_factor
    dataset_name = args.dataset_name
    root_dir = args.root_dir
    eval_type = args.eval_type
    label_list = np.array(['Benign', 'Attack'])
    per_of_samples = args.per_of_samples


    # Get autoencoder base data
    base_df = prediction_df_final[prediction_df_final['Sampling Period'] == 1]
    base_df['Model'] = "Mean-Absolute"
    base_df["Label Name"] = label_list[base_df['Label']]
    # base_df['Prediction'] = base_df['Prediction']/base_df['Prediction'].max()

    # Get canshield ensemble data
    ensemble_data = 0
    for sampling_period in sampling_periods:
        samp_data = prediction_df_final[prediction_df_final['Sampling Period'] == sampling_period][['Label', 'Loss']].values
        try:
            ensemble_data = ensemble_data + samp_data
        except:
            a = len(ensemble_data)
            b = len(samp_data)
            if a > b:
                ensemble_data = ensemble_data[:b] + samp_data
            else:
                ensemble_data = ensemble_data + samp_data[:a]

    ensemble_data /= len(sampling_periods)

    canshield_df = base_df.iloc[:ensemble_data.shape[0], :].copy()
    canshield_df['Label'] = (ensemble_data[:,0]>=0.50).astype(int)
    canshield_df['Prediction'] = ensemble_data[:,1]
    canshield_df['Loss'] = ensemble_data[:,1]
    canshield_df['Model'] = "CANShield-Ensemble"

    # Get the auc scores...
    baseline_autoencoder_auc = {}
    baseline_canshield_auc = {}

    # try:
    #     baseline_cm = pd.read_csv(f"{root_dir}/../data/results/{dataset_name}/baseline_cm.csv")
    #     baseline_cm.index = baseline_cm['Model']
    # except FileNotFoundError as e:
    #     print(e)
    #     print("Add baseline data to")
    #     print(f"{root_dir}/../data/results/{dataset_name}/baseline_cm.csv")

    for file_name in attacks_dict.keys():

        # Baseline autoencoder
        base_df_cut = base_df[base_df["Attack"] == file_name]
        fprs, tprs, thresholds = roc_curve(base_df_cut['Label'], base_df_cut['Prediction'])
        roc_auc = auc(fprs, tprs)
        baseline_autoencoder_auc[file_name] = roc_auc
        
        # Baseline canshields
        base_df_cut = canshield_df[canshield_df["Attack"] == file_name]
        fprs, tprs, thresholds = roc_curve(base_df_cut['Label'], base_df_cut['Prediction'])

        roc_auc = auc(fprs, tprs)
        baseline_canshield_auc[file_name] = roc_auc

    baseline_canshield_auc_df = pd.DataFrame(pd.Series(baseline_canshield_auc)).T
    baseline_canshield_auc_df.index = ['CANSheild-Ens']
    baseline_autoencoder_auc_df = pd.DataFrame(pd.Series(baseline_autoencoder_auc)).T
    baseline_autoencoder_auc_df.index = ['CANShield-Base']

    # Get canshield-i data
    canshield_each_dict = {}
    for file_name in attacks_dict.keys():
        canshield_each ={}
        for sampling_period in sampling_periods:
            
            key = f"{file_name}_{time_step}_{sampling_period}"
            heatmap_auc = heatmap_auc_dict[key][0]
            auc_avg = heatmap_auc_dict[key][1]
            found = False
            for i, indx in enumerate(heatmap_auc.index):
                for j, col in enumerate(heatmap_auc.columns):
                    if float(indx) == float(loss_factor) and float(col) == float(time_factor):
                        found = True
                        auc_score = heatmap_auc.iloc[i,j] + auc_avg
                        break
                if found == True:
                    break
            canshield_each[f"CANShield-{sampling_period}"] = auc_score
        canshield_each_dict[f"{file_name}"] = canshield_each
    baseline_comparison_auc_df = pd.DataFrame(canshield_each_dict)
    
    # Get existing baselines
    base_auc_others = pd.DataFrame([], columns = ['Flooding' ,'Suppress', 'Plateau', 'Continuous', 'Playback'])
    base_auc_others.loc['CANet'] = [0.979 ,0.882, 0.983, 0.936, 0.974]
    base_auc_others.loc['Autoencoder'] = [0.755  ,0.563 ,0.530 ,0.874 ,0.489]   
    base_auc_others.loc['Predictive']   = [0.722 ,0.561 ,0.530, 0.874, 0.489]   
    
    # Merge all the data...
    baseline_auc_final_df = np.round(pd.concat([baseline_autoencoder_auc_df, baseline_comparison_auc_df, baseline_canshield_auc_df, base_auc_others], ignore_index=False), 3)
    baseline_dir = Path(f'{root_dir}/../data/results/{dataset_name}/canshield_{dataset_name}_{eval_type}_{per_of_samples}_baseline_auc_final_df.csv')
    baseline_dir.parent.mkdir(exist_ok=True, parents=True)
    baseline_auc_final_df.to_csv(baseline_dir, header = True, index = True)

@hydra.main(version_base=None, config_path="../config", config_name="syncan")
def visualize_results(args : DictConfig) -> None:
    args.root_dir = get_original_cwd()
    print("Current working dir: ", args.root_dir)

    heatmap_auc_dict = None
    prediction_df_final = None
    
    # Calculate the AUCROC for different configuration... 
    # Increase in the AUC scores in different attacks with three-steps loss analysis 
    heatmap_auc_dict = create_heatmap_auc_dict(args)
    # plot_auc_improvement_with_three_steps(args, heatmap_auc_dict)
    
    # Plot the impact of different sampling periods
    prediction_df_final = anomalyScores_for_diff_canshields(args)
    # plot_anomalyScores_for_diff_canshields(args, prediction_df_final)
    

    get_canshield_baselines(args, heatmap_auc_dict, prediction_df_final)
    get_canshield_confusion_mat(args, heatmap_auc_dict, prediction_df_final)
    
             
if __name__ == "__main__":
    visualize_results()
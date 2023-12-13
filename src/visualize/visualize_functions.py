from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from dataset.load_dataset import *
from hydra.utils import get_original_cwd
from testing import *
import json
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from sklearn.metrics import roc_auc_score

from os.path import exists as file_exists  
from pathlib import Path

def generate_auroc_matrix(args, file_name):

    time_step = args.time_step
    sampling_period = args.sampling_period
    per_of_samples = args.per_of_samples
    root_dir = args.root_dir
    eval_type = args.eval_type
    dataset_name = args.dataset_name
    loss_factors = args.loss_factors
    time_factors = args.time_factors

    y_pred_avg = pd.Series([], dtype = np.float64)
    y_true_bin = pd.Series([], dtype = np.float64)
    heatmap_auc = pd.DataFrame([])

    label_file_name = f"label_{file_name}_{time_step}_{sampling_period}_{per_of_samples}"
    label_file_dir = f"{root_dir}/../data/label/{dataset_name}/{label_file_name}.csv"
    y_true_comb = pd.read_csv(label_file_dir, index_col=False)

    y_pred_avg = pd.concat([y_pred_avg, y_true_comb['Prediction']])
    y_true_bin = pd.concat([y_true_bin, y_true_comb['Label']])

    try:
        auc_avg = roc_auc_score(y_true_bin, y_pred_avg)
    except ValueError as e:
        print(e)
        print("Setting auc score as 0.00")
        auc_avg = 0

    for loss_factor in tqdm(loss_factors, leave=False):
        for time_factor in tqdm(time_factors, leave=False):
            y_pred_score = pd.Series([], dtype = np.float64)

            pred_file_name = f"prediction_{file_name}_{time_step}_{sampling_period}_{loss_factor}_{time_factor}_{per_of_samples}"
            pred_file_dir = f"{root_dir}/../data/prediction/{dataset_name}_{eval_type}/{pred_file_name}.csv"
            y_pred_score = pd.concat([y_pred_score, pd.read_csv(pred_file_dir, index_col=False)['0']])
            y_pred_score = y_pred_score.replace('>','.', regex=True).astype(np.float64)

            try:
                auroc = roc_auc_score(y_true_bin, y_pred_score)
            except ValueError as e:
                print(e)
                print("Setting auroc as 0.00")
                auroc = 0.00

            heatmap_auc.loc[loss_factor, time_factor] = auroc - auc_avg

    heatmap_auc_file_name = f"heatmap_auc_{file_name}_{time_step}_{sampling_period}_{per_of_samples}"
    heatmap_auc_file_dir = Path(f"{root_dir}/../data/visualization/{dataset_name}/{heatmap_auc_file_name}.csv")
    heatmap_auc_file_dir.parent.mkdir(parents=True, exist_ok=True)
    heatmap_auc.to_csv(heatmap_auc_file_dir, index = True)


    avg_auc_Dict = {
        'file_name' : file_name,
        'time_step' : time_step,
        'sampling_period' : sampling_period,
        'per_of_samples' : per_of_samples,
        'auc_avg' : auc_avg
        }
    avg_auc_file_name = f"avg_auc_{file_name}_{time_step}_{sampling_period}_{per_of_samples}"
    avg_auc_file_dir = f"{root_dir}/../data/visualization/{dataset_name}/{avg_auc_file_name}.json"
    with open(avg_auc_file_dir, 'w') as json_file:
        json.dump(avg_auc_Dict, json_file, indent=4)
    print("Data saved...!")
    return heatmap_auc, auc_avg

def create_heatmap_auc_dict(args):

    dataset_name = args.dataset_name
    time_steps = args.time_steps
    num_signals = args.num_signals
    sampling_periods = args.sampling_periods
    loss_factors = args.loss_factors
    time_factors = args.time_factors
    signal_factors = args.signal_factors
    attacks_dict = args.attacks_dict
    root_dir = args.root_dir
    per_of_samples = 1

    heatmap_auc_dict = {}

    for attack_name, file_name in attacks_dict.items():
        print("attack_name :", attack_name)
        for time_step in time_steps:
            print('time_step', time_step)
            for sampling_period in sampling_periods:
                print('sampling_period', sampling_period)
                
                try:
                    heatmap_auc_file_name = f"heatmap_auc_{file_name}_{time_step}_{sampling_period}_{per_of_samples}"
                    heatmap_auc_file_dir = f"{root_dir}/../data/visualization/{dataset_name}/{heatmap_auc_file_name}.csv"
                    heatmap_auc = pd.read_csv(heatmap_auc_file_dir, index_col=0)

                    avg_auc_file_name = f"avg_auc_{file_name}_{time_step}_{sampling_period}_{per_of_samples}"
                    avg_auc_file_dir = f"{root_dir}/../data/visualization/{dataset_name}/{avg_auc_file_name}.json"
                    with open(avg_auc_file_dir, 'r') as json_file:
                        auc_avg = json.load(json_file)['auc_avg']
                except:
                    args.sampling_period = sampling_period
                    args.time_step = time_step
                    heatmap_auc, auc_avg = generate_auroc_matrix(args, file_name)
                
                heatmap_auc_dict[f"{attack_name}_{time_step}_{sampling_period}"] = [heatmap_auc, auc_avg]

    return heatmap_auc_dict

def plot_auc_improvement_with_three_steps(args, heatmap_auc_dict):
    sampling_periods = args.sampling_periods
    attacks_dict = args.attacks_dict
    dataset_name = args.dataset_name
    time_step = args.time_step
    root_dir = args.root_dir
    testing_files = list(attacks_dict.keys())

    for sampling_period in sampling_periods:
        fig, axs = plt.subplots(1, len(attacks_dict), figsize = (7,2), sharex=True, sharey=True)
        for ax, file_name in zip(axs, testing_files):
            value = heatmap_auc_dict[f"{file_name}_{time_step}_{sampling_period}"][0].iloc[:-2,:-2][::-1]
            value.index = np.ceil(np.array(value.index)).astype(int)
            # print(np.array(value.columns).astype(float))
            value.columns = np.array(value.columns).astype(float) #np.ceil(np.array(value.columns)).astype(int)
            base = round(heatmap_auc_dict[f"{file_name}_{time_step}_{sampling_period}"][1],3)
            cbar = True if ax == axs[-1] else False
            sns.heatmap(value,  cmap="PiYG", vmin = -0.03, vmax = 0.03, square = False, cbar = cbar, xticklabels = 3, yticklabels = 3, ax = ax)
            ax.set_title(f"{file_name[:].capitalize()}\n $[{base}]$", weight='bold', fontsize = 12)

            if file_name == 'Flooding':
                ax.set_ylabel("Loss Threshold, \n$R^{{Loss}}(\%)$", fontsize = 12)
            if file_name == 'Plateau':
                ax.set_xlabel("Time Threshold, $R^{Time}(\%)$", fontsize = 12)
        
        plt.tight_layout()
        fig_dir = Path(f"{root_dir}/../plots/{dataset_name}/improve-in-auc-tsp-analysis_{sampling_period}")
        fig_dir.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{fig_dir}.jpg", dpi = 250)
        plt.savefig(f"{fig_dir}.pdf", dpi = 250)
        # plt.show()

def anomalyScores_for_diff_canshields(args):
    loss_factor = args.loss_factor
    time_factor = args.time_factor
    time_step = args.time_step
    attacks_dict = args.attacks_dict
    sampling_periods = args.sampling_periods
    dataset_name = args.dataset_name
    per_of_samples = args.per_of_samples
    root_dir = args.root_dir
    eval_type = args.eval_type

    prediction_df_final = pd.DataFrame([])

    for attack_name, file_name in tqdm(attacks_dict.items()):    
        for sampling_period in sampling_periods:

            prediction_df = pd.DataFrame([])
            y_pred_score = pd.DataFrame([])

            try:
                label_file_name = f"label_{file_name}_{time_step}_{sampling_period}_{per_of_samples}"
                label_file_dir = f"{root_dir}/../data/label/{dataset_name}/{label_file_name}.csv"
                prediction_df = pd.concat([prediction_df, pd.read_csv(label_file_dir, index_col=False)], ignore_index=True)
            except FileNotFoundError: #TODO : Fix genereated discripancies
                label_file_name = f"label_{file_name}_generated_{time_step}_{sampling_period}_{per_of_samples}"
                label_file_dir = f"{root_dir}/../data/label/{dataset_name}/{label_file_name}.csv"
                prediction_df = pd.concat([prediction_df, pd.read_csv(label_file_dir, index_col=False)], ignore_index=True)
            

            try:
                pred_file_name = f"prediction_{file_name}_{time_step}_{sampling_period}_{loss_factor}_{time_factor}_{per_of_samples}"
                pred_file_dir = f"{root_dir}/../data/prediction/{dataset_name}_{eval_type}/{pred_file_name}.csv"
                y_pred_score = pd.concat([y_pred_score, pd.read_csv(pred_file_dir, index_col=False)], ignore_index=True)
            except FileNotFoundError:
                pred_file_name = f"prediction_{file_name}_generated_{time_step}_{sampling_period}_{loss_factor}_{time_factor}_{per_of_samples}"
                pred_file_dir = f"{root_dir}/../data/prediction/{dataset_name}_{eval_type}/{pred_file_name}.csv"
                y_pred_score = pd.concat([y_pred_score, pd.read_csv(pred_file_dir, index_col=False)], ignore_index=True)
            

            prediction_df = pd.concat([prediction_df, y_pred_score], ignore_index= False, axis = 1)
            prediction_df['Attack'] = attack_name
            prediction_df['Sampling Period'] = sampling_period
            
            print(prediction_df_final.shape, prediction_df.shape, sampling_period)
            prediction_df_final = pd.concat([prediction_df_final, prediction_df], ignore_index= False, axis = 0)
    prediction_df_final = prediction_df_final.rename(columns = {'0': "Loss"})
    return prediction_df_final

def plot_anomalyScores_for_diff_canshields(args, prediction_df_final):
    
    loss_factor = args.loss_factor
    time_factor = args.time_factor
    time_step = args.time_step
    attacks_dict = args.attacks_dict
    sampling_periods = args.sampling_periods
    dataset_name = args.dataset_name
    per_of_samples = args.per_of_samples
    root_dir = args.root_dir

    fig, axes = plt.subplots(1, 5, figsize = (12,2.5), sharey = True)
    for ax, (attack_name, file_name) in tqdm(zip(axes, attacks_dict.items())):    
        print("prediction_df_final: ", prediction_df_final)
        print( prediction_df_final["Attack"].unique())
        print("file_name", file_name)

        filter1 = prediction_df_final["Attack"] == attack_name
        filter2 = prediction_df_final["Label"] == 1
        prediction_df_final_cut = prediction_df_final.where(filter1 & filter2).dropna()
        prediction_df_final_cut['Sampling Period'] = prediction_df_final_cut['Sampling Period'].astype(int)
        print("prediction_df_final_cut: ", prediction_df_final_cut)

        sns.boxplot(x="Sampling Period", y="Loss",  data= prediction_df_final_cut, ax = ax) #hue="Sampling Period", hue = 'Label',
        ax.legend().set_visible(False)
        # adding transparency to colors
        for patch in ax.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .3))
        # sns.violinplot(x="Sampling Period", y="Loss",  data= prediction_df_final_cut, alpha = 0.5, ax = ax) #hue="Sampling Period",
        if ax != axes[0]:
            ax.set(ylabel=None)
        else:
            ax.set_ylabel("Anomaly Score", fontsize = 12)
        
        if ax != axes[2]:
            ax.set(xlabel=None)
        else:
            ax.set_xlabel("Sampling Period", fontsize = 12)

        ax.set_title(f"{attack_name}", fontweight='bold', fontsize = 12) #    axs[file_index].set_title(f"{file_name} attack", fontweight='bold')
        
    plt.tight_layout()
    fig_dir = Path(f"{root_dir}/../plots/{dataset_name}/attack-wise-loss-for-diff-sp_box")
    fig_dir.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{fig_dir}.jpg", dpi = 300)
    plt.savefig(f"{fig_dir}.pdf", dpi = 300)
    plt.show()

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

#pred_missing_df = find_missing_files (args, test_file_dir_dict)
def find_missing_files (args, test_file_dir_dict):
    dataset_name = args.dataset_name
    time_steps = args.time_steps
    sampling_periods = args.sampling_periods
    loss_factors = args.loss_factors
    time_factors = args.time_factors
    root_dir = args.root_dir
    per_of_samples = args.per_of_samples
    eval_type = args.eval_type
    
    pred_missing = []


    for file_name in test_file_dir_dict.keys():
        print(file_name)
        file_name = file_name.replace("_generated", "")
        for time_step in time_steps:
            for sampling_period in sampling_periods:
                for loss_factor in loss_factors:
                    for time_factor in time_factors:
                        pred_file_name = f"prediction_{file_name}_{time_step}_{sampling_period}_{loss_factor}_{time_factor}_{per_of_samples}"
                        pred_file_dir = f"{root_dir}/../data/prediction/{dataset_name}_{eval_type}/{pred_file_name}.csv"

                        if not file_exists(pred_file_dir):
                            pred_missing.append([file_name, time_step, sampling_period, loss_factor
                                              , time_factor, per_of_samples])
    feat_cols = ["file_name", "time_step", "sampling_period", "loss_factor", "time_factor", 'file_size']
    pred_missing_df = pd.DataFrame(pred_missing, columns=feat_cols)
    print(pred_missing_df)
    return pred_missing_df

def generate_testing_predictions(args, file_dir_dict, pred_missing_df):

    dataset_name = args.dataset_name
    time_steps = args.time_steps
    sampling_periods = args.sampling_periods
    loss_factors = args.loss_factors
    time_factors = args.time_factors
    root_dir = args.root_dir
    num_signals = args.num_signals
    per_of_samples = args.per_of_samples
    eval_type = args.eval_type

    for file_name, file_path in tqdm(file_dir_dict.items()):        
        filter_repeat_file = (pred_missing_df['file_name'] == file_name)
        
        for time_step in time_steps:
            print('time_step', time_step)
            
            filter_repeat_time = (filter_repeat_file & pred_missing_df['time_step'] == time_step)

            for sampling_period in sampling_periods:
                print('sampling_period', sampling_period)

                args.time_step = time_step
                args.sampling_period = sampling_period

                filter_repeat_sampling = (filter_repeat_time & pred_missing_df['sampling_period'] == sampling_period)

                if len(filter_repeat_sampling) == 0:
                    continue

                print("Generating prediction data")
                y_test_prob_org, y_test_seq = generate_and_save_prediction_loss_per_file(
                    args, file_name, file_path, eval_type)
                                
                label_file_name = f"label_{file_name}_{time_step}_{sampling_period}_{per_of_samples}"
                label_file_dir = Path(f"{root_dir}/../data/label/{dataset_name}/{label_file_name}.csv")
                label_file_dir.parent.mkdir(parents=True, exist_ok=True)
                y_test_seq_label = pd.DataFrame([])
                y_test_seq_label["Label"] = y_test_seq
                y_test_seq_label["Prediction"] = np.mean(np.mean(y_test_prob_org,1),1)
                y_test_seq_label.to_csv(label_file_dir, header=True, index=False)

                for loss_factor in loss_factors:
                    print(f"loss_factor : {loss_factor}")

                    filter_repeat_loss = (filter_repeat_sampling & pred_missing_df['loss_factor'] == loss_factor)
                    if len(filter_repeat_loss) == 0:
                        continue
                    print("Load loss_factor...", loss_factor)
                    loss_df = pd.read_csv(f'{root_dir}/../data/thresholds/{dataset_name}/thresholds_loss_{dataset_name}_{num_signals}_{time_step}_{sampling_period}.csv')
                    ths_loss_image = np.squeeze(loss_df[loss_df['loss_factor'] == loss_factor]['th'].values)
                    y_test_prob_org_bin = (y_test_prob_org> ths_loss_image).astype(int).copy()
                    y_test_prob_org_bin_count = np.sum(y_test_prob_org_bin, 1)/time_step


                    for time_factor in time_factors:
                        print(f"time_factor : {time_factor}")
                        filter_repeat_time = (filter_repeat_loss & pred_missing_df['time_factor'] == time_factor)
                        if len(filter_repeat_time) == 0:
                            continue

                        print("Load time_factor...", time_factor)
                        pred_file_name = f"prediction_{file_name}_{time_step}_{sampling_period}_{loss_factor}_{time_factor}_{per_of_samples}"
                        pred_file_dir = Path(f"{root_dir}/../data/prediction/{dataset_name}_{eval_type}/{pred_file_name}.csv")
                        pred_file_dir.parent.mkdir(parents=True, exist_ok=True)

                        time_df = pd.read_csv(f'{root_dir}/../data/thresholds/{dataset_name}/thresholds_time_{dataset_name}_{num_signals}_{time_step}_{sampling_period}_{loss_factor}.csv')
                        ths_time_image = np.squeeze(time_df[time_df['time_factor'] == time_factor]['th'].values)
                        y_test_prob_org_sig_count = (y_test_prob_org_bin_count> ths_time_image).astype(int)
                        y_test_prob_org_sig_count = pd.DataFrame(np.sum(y_test_prob_org_sig_count, 1)/num_signals)
                        y_test_prob_org_sig_count.to_csv(pred_file_dir, index = False)


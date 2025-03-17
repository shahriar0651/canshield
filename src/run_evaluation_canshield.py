from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from dataset.load_dataset import *
from hydra.utils import get_original_cwd
from testing import *

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from sklearn.metrics import roc_auc_score

from os.path import exists as file_exists

from testing.load_predictions import find_missing_files, generate_testing_predictions    


@hydra.main(version_base=None, config_path="../config", config_name="syncan")
def evaluate_canshield(args : DictConfig) -> None:
    root_dir = Path(__file__).resolve().parent
    print("root_dir: ", root_dir)
    args.root_dir = root_dir
    args.data_type = "testing"
    print("Current working dir: ", args.root_dir)      

    # Generated all the thresholds for different factors....  
    args.window_step = args.window_step_valid
    print(f"Starting thresholding with args.window_step: {args.window_step}")
    args.data_dir = args.train_data_dir # target data
    train_file_dir_dict = get_list_of_files(args)
    loss_dict = get_existing_threshold_data(args)
    generate_remaining_threshold_data(args, loss_dict, train_file_dir_dict)
    print("Generated all the thresholds data...")

    # Generated all the prediction for different factors....  
    args.window_step = args.window_step_test
    print(f"Starting testing with args.window_step: {args.window_step}")
    args.data_dir = args.test_data_dir # target test data
    test_file_dir_dict = get_list_of_files(args)
    pred_missing_df = find_missing_files (args, test_file_dir_dict)
    generate_testing_predictions(args, test_file_dir_dict, pred_missing_df)              

if __name__ == "__main__":
    evaluate_canshield()
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from dataset.load_dataset import *
# from hydra.utils import get_original_cwd
from training import *


@hydra.main(version_base=None, config_path="../config", config_name="syncan")
def develop_canshield(args : DictConfig) -> None:
    root_dir = Path(__file__).resolve().parent
    print("root_dir: ", root_dir)
    args.root_dir = root_dir
    args.data_type = "training"
    args.data_dir = args.train_data_dir
    print("Current working dir: ", args.root_dir)
    dataset_name = args.dataset_name
    num_signals = args.num_signals

    for time_step in args.time_steps:
        for sampling_period in args.sampling_periods:
            # Sep-up variable to define the AE model
            args.time_step = time_step
            args.sampling_period = sampling_period
            args.window_step = args.window_step_train
            print(f"Starting thresholding with args.window_step: {args.window_step}")

            # Train individual AE for each combination
            autoencoder, retrain = get_autoencoder(args)

            # if retrain == False:
            #     print("Model already trained.")
            #     return None
            
            file_dir_dict = get_list_of_files(args)
            print("file_dir_dict: ", file_dir_dict)
            assert len(file_dir_dict) > 0, "No files found in the specified directory."
            for file_index, (file_name, file_path) in tqdm(enumerate(file_dir_dict.items())):
                try:
                    print("Starting loading", file_index, file_name)
                    x_train_seq, _ = load_data_create_images(args, file_name, file_path)
                
                    print("Starting trainin with", file_name)
                    autoencoder = train_autoencoder(args, file_index, autoencoder, x_train_seq)
                except Exception as error:
                    print(error)
                    print(f"Skipping dataset {file_name}")

            model_dir = f"{root_dir}/../artifacts/models/{dataset_name}/autoendoer_canshield_{dataset_name}_{time_step}_{num_signals}_{sampling_period}.h5"
            autoencoder.save(model_dir)
            print("Model saved.......!")

print("Training AE Models for CANShield is complete!")

if __name__ == "__main__":
    develop_canshield()
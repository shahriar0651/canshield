import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
import json

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0

def visualize_autoencoder(autoencoder, input_image, history_dir, save):
    # Visualizing the training performance...........
    output_image = autoencoder.predict(input_image)
    recon_loss = np.abs(output_image - input_image)

    #------- Ploting the data -------------------
    fig, axes = plt.subplots(1, 3, figsize = (12, 4))

    sns.heatmap(input_image[0,:,:,0], ax = axes [0], vmin = 0, vmax = 1)
    axes[0].set_title("input_image")
    sns.heatmap(output_image[0,:,:,0], ax = axes[1], vmin = 0, vmax = 1)
    axes[1].set_title("output_image")
    sns.heatmap(recon_loss[0,:,:,0], ax = axes[2], vmin = 0, vmax = 1)
    axes[2].set_title("recon_loss")
    plt.tight_layout()
    if save:
        plt.savefig(history_dir, dpi = 150)
    plt.show()

# def train_autoencoder(args, file_index, autoencoder, x_train_seq):
    
#     dataset_name = args.dataset_name
#     time_step = args.time_step
#     num_signals = args.num_signals
#     sampling_period = args.sampling_period
#     max_epoch = args.max_epoch
#     root_dir = args.root_dir
    

#     checkpoint_path = f"{root_dir}/../artifacts/model_ckpts/{dataset_name}/autoendoer_canshield_{dataset_name}_{time_step}_{num_signals}_{sampling_period}.h5"
#     keras_callbacks   = [
#             EarlyStopping(monitor='val_loss', patience=10, mode='min'),
#             ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only = True, mode='auto', verbose=1)
#             ]
    
#     history = autoencoder.fit(
#         x_train_seq,
#         x_train_seq,
#         epochs=max_epoch ,
#         batch_size=128,
#         validation_split=0.1,
#         callbacks=keras_callbacks
#     )

#     visualize_dir = Path(f"{root_dir}/../artifacts/visualize/{dataset_name}/performance_canshield_{dataset_name}_{time_step}_{num_signals}_{sampling_period}_{file_index+1}.jpg")
#     history_dir = Path(f"{root_dir}/../artifacts/histories/{dataset_name}/history_canshield_{dataset_name}_{time_step}_{num_signals}_{sampling_period}_{file_index+1}.json")
#     visualize_dir.parent.mkdir(exist_ok=True, parents=True)
#     history_dir.parent.mkdir(exist_ok=True, parents=True)
#     visualize_autoencoder(autoencoder, x_train_seq[0:1], visualize_dir, True)
#     with open(history_dir, "w") as fp:
#         json.dump(history.history, fp)

#     return autoencoder



def train_autoencoder(args, file_index, autoencoder, x_train_seq):



    # Ensure the model is using GPU
    if tf.config.experimental.list_physical_devices('GPU'):
        print("Training on GPU")
    else:
        print("Training on CPU")

    # Your existing code...
    dataset_name = args.dataset_name
    time_step = args.time_step
    num_signals = args.num_signals
    sampling_period = args.sampling_period
    max_epoch = args.max_epoch
    root_dir = args.root_dir

    checkpoint_path = f"{root_dir}/../artifacts/model_ckpts/{dataset_name}/autoendoer_canshield_{dataset_name}_{time_step}_{num_signals}_{sampling_period}.h5"
    keras_callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, mode='min'),
        ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='auto', verbose=1)
    ]
    
    # Train the model
    history = autoencoder.fit(
        x_train_seq,
        x_train_seq,
        epochs=max_epoch,
        batch_size=128,
        validation_split=0.1,
        callbacks=keras_callbacks
    )

    # Save visualization and history
    visualize_dir = Path(f"{root_dir}/../artifacts/visualize/{dataset_name}/performance_canshield_{dataset_name}_{time_step}_{num_signals}_{sampling_period}_{file_index+1}.jpg")
    history_dir = Path(f"{root_dir}/../artifacts/histories/{dataset_name}/history_canshield_{dataset_name}_{time_step}_{num_signals}_{sampling_period}_{file_index+1}.json")
    visualize_dir.parent.mkdir(exist_ok=True, parents=True)
    history_dir.parent.mkdir(exist_ok=True, parents=True)
    
    visualize_autoencoder(autoencoder, x_train_seq[0:1], visualize_dir, True)

    with open(history_dir, "w") as fp:
        json.dump(history.history, fp)

    return autoencoder

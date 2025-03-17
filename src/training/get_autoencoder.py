# from tensorflow import keras
# from keras import Sequential
# from keras.layers import Conv2D
# from keras.layers import LeakyReLU, MaxPooling2D, UpSampling2D 
# from keras.layers import ZeroPadding2D, Cropping2D
# from keras.optimizers import Adam
# from keras.models import load_model
# from keras.losses import MeanSquaredError

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, MaxPooling2D, UpSampling2D, ZeroPadding2D, Cropping2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError


import glob
from pathlib import Path


def get_new_autoencoder(time_step, num_signals):
    in_shape = (time_step, num_signals, 1)
    autoencoder = Sequential()
    #------------------- Encoder -------------------------#
    autoencoder.add(ZeroPadding2D((2, 2),input_shape=in_shape))
    autoencoder.add(Conv2D(32, (5,5), strides=(1, 1), padding='same'))
    autoencoder.add(LeakyReLU(alpha=0.2))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))

    autoencoder.add(Conv2D(16, (5,5), strides=(1, 1), padding='same'))
    autoencoder.add(LeakyReLU(alpha=0.2))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))

    autoencoder.add(Conv2D(16, (3,3), strides=(1, 1), padding='same'))
    autoencoder.add(LeakyReLU(alpha=0.2))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    # #------------------- Decoder -------------------------#
    autoencoder.add(Conv2D(16, (3,3), strides=(1, 1), padding='same')) 
    autoencoder.add(LeakyReLU(alpha=0.2))
    autoencoder.add(UpSampling2D((2, 2)))

    autoencoder.add(Conv2D(16, (5,5), strides=(1, 1), padding='same')) 
    autoencoder.add(LeakyReLU(alpha=0.2))
    autoencoder.add(UpSampling2D((2, 2)))

    autoencoder.add(Conv2D(32, (5,5), strides=(1, 1), padding='same')) 
    autoencoder.add(LeakyReLU(alpha=0.2))
    autoencoder.add(UpSampling2D((2, 2))) 

    #--------------------- Crop ---------------------------#
    temp_shape = autoencoder.output_shape
    diff = temp_shape[1] - in_shape[0]
    top = int(diff/2)
    bottom = diff - top
    diff = temp_shape[2] - in_shape[1]
    left = int(diff/2)
    right = diff - left
    #-------------------------------------------------------#
    autoencoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
    autoencoder.add(Cropping2D(cropping=((top, bottom), (left, right))))
    return autoencoder


def get_autoencoder(args):
    root_dir = args.root_dir
    time_step = args.time_step
    num_signals = args.num_signals
    dataset_name = args.dataset_name
    sampling_period = args.sampling_period

    # Search for existin model.....
    model_list = glob.glob(f"{root_dir}/../artifacts/models/{dataset_name}/autoendoer_canshield_{dataset_name}_{time_step}_{num_signals}_*.h5")    
    sp_best = 0
    for model_dir in model_list: 
        file_name = Path(model_dir).name.split(".")[0]
        print(file_name)
        sp_existing = int(file_name.split("_")[-1])
        if sp_existing > sp_best and sp_existing <= sampling_period: 
            sp_best = sp_existing

    retrain = True
    if sp_best == 0:
        # If not found, create a new one
        autoencoder = get_new_autoencoder(time_step, num_signals)
        print(f"Model created...")
        
    if sp_best > 0:
        model_dir = f"{root_dir}/../artifacts/models/{dataset_name}/autoendoer_canshield_{dataset_name}_{time_step}_{num_signals}_{sp_best}.h5"
        autoencoder = load_model(model_dir)
        if sp_best == sampling_period:
            retrain = False
        print(f"Model loaded from {model_dir}")

    # compile autoencoder
    opt = Adam(learning_rate = 0.0002, beta_1=0.5, beta_2=0.99)
    autoencoder.compile(loss=MeanSquaredError(), optimizer=opt, metrics=['accuracy']) #loss='binary_crossentropy'
    autoencoder.summary()

    return autoencoder, retrain
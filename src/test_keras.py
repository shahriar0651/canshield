import tensorflow as tf
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential
from keras.layers import Conv2D
from keras.layers import LeakyReLU, MaxPooling2D, UpSampling2D, ZeroPadding2D, Cropping2D
from keras.optimizers import Adam
from keras.models import load_model
from omegaconf import DictConfig, OmegaConf
import glob
from pathlib import Path
attacks_dict = {'Flooding': 'test_flooding',
        'Suppress': 'test_suppress',
        'Plateau': 'test_plateau',
        'Continuous': 'test_continuous',
        'Playback': 'test_playback'}

df = pd.read_csv("data/results/syncan/baseline_cm.csv")
# df = df.astype(float)
print(df.describe())

for file_name in attacks_dict.keys():
    df = df[df['Model'] == 'CANet']
    print(df[f'{file_name}_fpr'].values[0])


# autoencoder_org = tf.keras.models.load_model('/Users/hshahriar/Desktop/Workspace/canshield-updated/src/../artifacts/models/syncan/Autoencoder_Final_50_1_5_True.h5')
# # print(model.summary())

# # for layers in autoencoder.layers:
# #     try:
# #         print(layers.name)
# #         print(layers.activation)
# #     except:
# #         pass

# time_step = 50
# num_signals = 20
# in_shape = (time_step, num_signals, 1)
# autoencoder = Sequential()
# #------------------- Encoder -------------------------#
# autoencoder.add(ZeroPadding2D((2, 2),input_shape=in_shape))
# autoencoder.add(Conv2D(32, (5,5), strides=(1, 1), padding='same'))
# autoencoder.add(LeakyReLU(alpha=0.2))
# autoencoder.add(MaxPooling2D((2, 2), padding='same'))

# autoencoder.add(Conv2D(16, (5,5), strides=(1, 1), padding='same'))
# autoencoder.add(LeakyReLU(alpha=0.2))
# autoencoder.add(MaxPooling2D((2, 2), padding='same'))

# autoencoder.add(Conv2D(16, (3,3), strides=(1, 1), padding='same'))
# autoencoder.add(LeakyReLU(alpha=0.2))
# autoencoder.add(MaxPooling2D((2, 2), padding='same'))
# # #------------------- Decoder -------------------------#
# autoencoder.add(Conv2D(16, (3,3), strides=(1, 1), padding='same')) 
# autoencoder.add(LeakyReLU(alpha=0.2))
# autoencoder.add(UpSampling2D((2, 2)))

# autoencoder.add(Conv2D(16, (5,5), strides=(1, 1), padding='same')) 
# autoencoder.add(LeakyReLU(alpha=0.2))
# autoencoder.add(UpSampling2D((2, 2)))

# autoencoder.add(Conv2D(32, (5,5), strides=(1, 1), padding='same')) 
# autoencoder.add(LeakyReLU(alpha=0.2))
# autoencoder.add(UpSampling2D((2, 2))) 

# #---------------------------------------------------------
# temp_shape = autoencoder.output_shape
# # print("output_shape", temp_shape)
# # print("in_shape", in_shape)

# diff = temp_shape[1] - in_shape[0]
# top = int(diff/2)
# bottom = diff - top
# # print(diff, top, bottom)

# diff = temp_shape[2] - in_shape[1]
# left = int(diff/2)
# right = diff - left
# # print(diff, left, right)
# #------------------------------------------------------------

# autoencoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
# autoencoder.add(Cropping2D(cropping=((top, bottom), (left, right))))

# print(autoencoder.summary())

# for layers_org, layers_ in zip(autoencoder_org.layers, autoencoder.layers):
#     try:
#         print("-----org-----")
#         # print(layers_org.name)
#         print(layers_org.activation)
#         print("-----new-------")
#         # print(layers.name)
#         print(layers_.activation)
#         print("==================\n\n")
#     except:
#         pass

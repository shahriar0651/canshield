import os
import numpy as np
import pandas as pd

def sigmoid(x):
    return 1/(1+np.exp(-x))

#--------------- Create Folders -------------
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Detectory created!\n{directory}")
        return False
    else:
        return True

#--------------------------------------
def calc_loss(x_org, x_recon):
    recon_loss = np.abs(x_org - x_recon)
    return recon_loss #sigmoid(np.mean(np.mean(recon_loss, axis = 1), axis = 1))
# #--------------------------------------
# def calc_loss(x_org, autoencoder):
#     x_recon = autoencoder.predict(x_org)
#     recon_loss = np.abs(x_org - x_recon)
#     return recon_loss #sigmoid(np.mean(np.mean(recon_loss, axis = 1), axis = 1))
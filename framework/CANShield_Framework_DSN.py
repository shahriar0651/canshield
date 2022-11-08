#!/usr/bin/env python
# coding: utf-8

# In[6]:


# importing libraries.......
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from tensorflow.keras.models import load_model
#import matplotlib.pyplot as plt
import datetime


# In[7]:


# Loading parameters and dataset
platform = "PC"

if platform != "PC":
    
    #import libraries for led blinking
    from gpiozero import LED

    # Defining three LEDs
    dataLED = LED(23)
    attackLED = LED(24)
    detectionLED = LED(25)


# Defining IDs and number of associated signals
list_of_ids = list(range(1, 11))
num_of_id = [2, 3, 2, 1, 2, 2, 2, 1, 1, 4]
num_sigs_per_id = {}
for can_id, n_id in zip(list_of_ids, num_of_id):
    num_sigs_per_id[can_id] = n_id


# #Loading Raw Data Frame
folder_dir = 'SynCAN Data Cut/'
target_file = "test_flooding_cut"
target_file = "new_data"

data_dir = folder_dir+target_file+'.csv'

print("Loading dataset....")

with open(data_dir,'r') as dest_f:
    data_iter = csv.reader(dest_f,
                           delimiter = ',',
                           quotechar = '"')
    data = [data for data in data_iter]
df_raw = pd.DataFrame(data)
df_raw.columns = ['Index', 'Label', 'Time', 'ID','Signal1', 'Signal2', 'Signal3', 'Signal4']
df_raw = df_raw.drop(0)
df_raw = df_raw.iloc[:,1:]

# In[8]:


# collected signals...
signals_in_cluster = ['Sig_2_of_ID_7',
 'Sig_2_of_ID_1',
 'Sig_1_of_ID_3',
 'Sig_2_of_ID_10',
 'Sig_2_of_ID_6',
 'Sig_1_of_ID_8',
 'Sig_3_of_ID_2',
 'Sig_1_of_ID_5',
 'Sig_1_of_ID_4',
 'Sig_1_of_ID_6',
 'Sig_2_of_ID_5',
 'Sig_3_of_ID_10',
 'Sig_2_of_ID_3',
 'Sig_1_of_ID_2',
 'Sig_1_of_ID_7',
 'Sig_2_of_ID_2',
 'Sig_1_of_ID_1',
 'Sig_4_of_ID_10',
 'Sig_1_of_ID_10',
 'Sig_1_of_ID_9']

print("clusters dataset....")
# In[9]:


# # Design hyper-parameters....
# T_xs = [1]
# w = 20
# q = max(T_xs)*w
# n = len(signals_in_cluster)

# # Loading autoencoders....
# autoencoders = {}
# for T_x in T_xs:
#     filename = f'AE_Model//Final_{w}//Autoencoder_Final_{w}_1_{T_x}_True.h5'
#     autoencoders[T_x] = load_model(filename)
    

# #Setting up thresholds....
# R_loss = np.random.rand(n)
# R_time = (np.random.rand(n)*n).astype(int)
# R_signal = np.random.randint(n)


# In[ ]:





# In[ ]:





# In[11]:


# # Prediction results...
# predictions = {}
# for T_x in T_xs:
#     predictions[T_x] =[]
# predictions['Ens'] =[]
# predictions


# In[19]:


time_data = []


# In[20]:


dataset = 'syncan'


# In[21]:


for w in [20, 50, 100]:
#for w in [25, 50, 75]:
    # Design hyper-parameters....
    T_xs = [1, 5, 10]
    q = max(T_xs)*w
    n = len(signals_in_cluster)

    print("windowsize....", w)

    # Loading autoencoders....
    autoencoders = {}
    for T_x in T_xs:
        filename = f'AE_Model//Final_{w}//Autoencoder_Final_{w}_1_{T_x}_True.h5'
        #filename = f'AE_Model//{dataset}//autoendoer_canshield_{dataset}_{w}_1_{T_x}_1.h5'

        autoencoders[T_x] = load_model(filename)


    #Setting up thresholds....
    R_loss = np.random.rand(n)
    R_time = (np.random.rand(n)*n).astype(int)
    R_signal = np.random.randint(n)


    # Initiating the dataQ
    dataQ = pd.DataFrame(np.zeros((q, n), dtype = float), columns = signals_in_cluster)
    dataQ.index = dataQ.index + 1

    #print("Initiating analysis.... ", w)
    for index, data in enumerate(df_raw.values):   
        
        
        #print(index)
        
        begin_time = datetime.datetime.now()
        #Reading data from the CAN bus in decoded foramt
        can_id = int(data[2][2:])
        signals_reported = [f'Sig_{i+1}_of_ID_{can_id}' for i in range(num_sigs_per_id[can_id])]
        signals_missing = set(signals_in_cluster) - set(signals_reported)

        #print("Updating dataQ")

        # Updating dataQ    
        new_data = [float(x) for x in data[3:3+num_sigs_per_id[can_id]]].copy()
        dataQ.loc[2:q] = dataQ.loc[1:q-1].values.copy()
        dataQ.loc[1, signals_reported] = new_data.copy()
        dataQ.loc[1, signals_missing] = dataQ.loc[2, signals_missing].copy()
        time_elapsed = datetime.datetime.now() - begin_time
        time_needed_dataQ = float(time_elapsed.total_seconds())


        #Creating different views and predicting reconstrcuted image
        anomaly_S_ens = 0
    #     prediction_time = 0
    #     analysis_time = 0


        for T_x in T_xs:       

            #print("Time", T_x)
            prediction_time = 0
            analysis_time = 0

            # Creating different views.....
            #starting timer
            begin_time = datetime.datetime.now()

            dataV_org = dataQ.loc[1:(w*T_x):T_x].values.copy()
            dataV_recon = autoencoders[T_x].predict(dataV_org.reshape(1, w, n, 1))[0,:,:,0]

            #stopping timer
            time_elapsed = datetime.datetime.now() - begin_time
            prediction_time += float(time_elapsed.total_seconds())

            #starting timer
            begin_time = datetime.datetime.now()

            dataV_loss = abs(dataV_org - dataV_recon).copy()
            #finding anomaly scores for individual models...
            anomaly_B = (dataV_loss > R_loss).astype(int)
            anomaly_C = np.sum(anomaly_B,0)
            anomaly_S = np.sum(anomaly_C > R_time)
            anomaly_S_ens += anomaly_S

            #stopping timer
            time_elapsed = datetime.datetime.now() - begin_time
            analysis_time  += float(time_elapsed.total_seconds())

            time_data.append([w, T_x, time_needed_dataQ, prediction_time, analysis_time])



    #         if anomaly_S > R_signal:
    #             predictions[T_x].append(1)
    #         else:
    #             predictions[T_x].append(0)    


        # Checking the final ensemble score..
    #     if anomaly_S_ens > R_signal_ens:
    #         predictions['Ens'].append(1)
    #     else:
    #         predictions['Ens'].append(0)   

    #     time_data.append([w, T_x, time_needed_dataQ, prediction_time, analysis_time])

        if index == 20:
            break


# In[22]:


time_df = pd.DataFrame(time_data, columns = ['w', 'Tx', 'Data Queue', 'Prediction', 'Analysis'])


# In[23]:


time_df.to_csv('time_df.csv', header = True, index = True)


# In[24]:


import seaborn as sns
sns.boxplot(data = time_df, x = 'w', y = 'Prediction', hue = 'Tx')
sns.boxplot(data = time_df, x = 'w', y = 'Data Queue', hue = 'Tx')
sns.boxplot(data = time_df, x = 'w', y = 'Analysis', hue = 'Tx')


# In[25]:


#autoencoders[1].summary()


# In[ ]:





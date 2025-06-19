import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import OrderedDict
from tqdm import tqdm
from pathlib import Path


def generate_dataset(file_name, file_path, org_columns):
  
    print("Generating dataset: ", file_name)
    extd_file_dir = Path(f"{file_path.parent}/generated/{file_name}_generated.csv")   
    # Load original dataset
    df_original = pd.read_csv(file_path, skiprows=1, names=org_columns)
    # Replace 'id' from the ID number only in SynCAN dataset
    df_original['ID'].replace("id", "", regex=True, inplace=True) 

    # Start extending the dataset with "Signal_X_of_ID_Y" format
    df_extended = pd.DataFrame([])
    # Starting with 'ID','Label','Time' data from the original file
    df_extended[['ID','Label','Time']] = df_original[['ID','Label','Time']]
    
    # Repeating for all the unique CAN IDs
    for target_id in tqdm(df_original['ID'].unique()):
        # print("Targeted ID: ", target_id, end = ' ')
        # selecting the rows only with the a specific CAN ID and valid signals
        df_perID = df_original[df_original['ID'] == target_id].T.dropna().T

        # Appending ID number at the end of the signal column
        column_rename_dict = {}
        for org_column_name in set(df_perID.columns) - set(['ID','Label','Time']):
            new_column_name = org_column_name.replace("Signal", "Sig_")
            if "_of_ID" not in new_column_name:
                new_column_name = new_column_name + "_of_ID"
            new_column_name = new_column_name + "_" + str(target_id)
            # print(org_column_name, new_column_name)
            column_rename_dict[org_column_name] = new_column_name
        # renaming the columns
        df_perID = df_perID.rename(column_rename_dict, axis=1)
        new_column_names = list(column_rename_dict.values())
        
        #merging ID wise data frame in a singel combined one
        df_extended = pd.concat([df_extended, df_perID[new_column_names]], axis = 1)

    # Saving the dataset for future use
    extd_file_dir.parent.mkdir(parents=True, exist_ok=True)
    df_extended.to_csv(extd_file_dir, header = True, index = True)
    print(f"File saved to {extd_file_dir}")
    return df_extended

def get_minmax_scaler(features, dataset_name, scaler_dir):
    df_min_max = pd.read_csv(f"{scaler_dir}/min_max_values_{dataset_name}.csv"
                             , index_col=0)[features]
    scaler = MinMaxScaler()
    scaler.fit(df_min_max.values)
    print("scaler loaded...!")
    return scaler

def get_list_of_files(args): #data_type: str, clean_data_dir: str):
    data_type = args.data_type
    data_dir = args.data_dir

    file_dir_dict = {}
    file_paths = glob.glob(f"{data_dir}/*.csv")
    print(f"Found {len(file_paths)} files in {data_dir} for {data_type} data.")
    for file_path in file_paths:
        file_name = file_path.split("/")[-1].split(".")[0]
        file_dir_dict[file_name] = file_path
    file_dir_dict = OrderedDict(sorted(file_dir_dict.items()))
    return file_dir_dict
       
def load_data(dataset_name, file_name, file_path, features, org_columns, per_of_samples):
    
    # Load dataset
    file_path = Path(file_path)
    generated_data = Path(f"{file_path.parent}/generated/{file_name}_generated.csv")

    if generated_data.exists():
        print(f"Loading {generated_data}...")
        X = pd.read_csv(generated_data, index_col=0)
    else:
        print(f"{generated_data} does not exists!")
        X = generate_dataset(file_name, file_path, org_columns)


    print(f"{file_name} loaded..")
    # Defining the number of samples 
    num_of_samples = int(X.shape[0]*per_of_samples)
    y = X['Label'].iloc[0:num_of_samples]
    X = X[features].iloc[:num_of_samples].astype(float)
    print("Forward filling...")
    X = X.ffill().copy()
    X = X.bfill().dropna()   
    print("X_train.shape", X.shape)
    print("Done data treatment..")
    return X, y

def scale_dataset(X, dataset_name, features, scaler_dir):
    X = X.values.copy()  
    scaler_train = get_minmax_scaler(features, dataset_name, scaler_dir)
    X = scaler_train.transform(X)
    print("Dataset scalled!")
    return X

def create_x_sequences(X, time_step, window_step, num_signals, sampling_period):
    X_output = []
    for i in range(0, (len(X) - sampling_period*time_step), window_step):
        X_output.append(X[i : (i + sampling_period*time_step) : sampling_period])
    print("X sequence created!")
    return np.stack(X_output).reshape(-1, time_step, num_signals, 1)

def create_y_sequences(y, time_step, window_step, sampling_period):
    y_output = []
    for i in range(0, (len(y) - sampling_period*time_step), window_step):
        if y[i : (i + sampling_period*time_step)].sum() > 0:
            y_output.append(1)
        else:
            y_output.append(0)
    print("y sequence created!")
    return np.stack(y_output)

def load_data_create_images(args, file_name, file_path):

    dataset_name = args.dataset_name
    org_columns = args.org_columns
    features = args.features
    per_of_samples = args.per_of_samples
    scaler_dir = args.scaler_dir
    time_step = args.time_step
    window_step = args.window_step
    num_signals = args.num_signals
    sampling_period = args.sampling_period
    
    X, y = load_data(dataset_name, file_name, file_path, features, org_columns, per_of_samples)
    X = scale_dataset(X, dataset_name, features, scaler_dir)
    print("getting x seq")
    x_seq = create_x_sequences(X, time_step, window_step, num_signals, sampling_period)
    print("getting y seq")
    y_seq = create_y_sequences(y, time_step, window_step, sampling_period)
    return x_seq, y_seq
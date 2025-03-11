import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from training import *
from dataset import *
from testing.helper import *

class node:
    def __init__(self, th_name, th_exist, th_data, child_list):
        self.th_name = th_name
        self.th_exist = th_exist
        self.th_data = th_data
        self.child_list = child_list   

def get_existing_threshold_data(args):
    dataset_name = args.dataset_name
    time_steps = args.time_steps
    num_signals = args.num_signals
    sampling_periods = args.sampling_periods
    root_dir = args.root_dir

    loss_factors = args.loss_factors
    time_factors = args.time_factors
    signal_factors = args.signal_factors


    loss_dict = {}

    for time_step in time_steps:
        for sampling_period in sampling_periods:
                
            loss_th = f'thresholds_loss_{dataset_name}_{num_signals}_{time_step}_{sampling_period}'
            loss_th_dir = f'{root_dir}/../data/thresholds/{dataset_name}/{loss_th}'
            
            try:
                loss_th_data = pd.read_csv(loss_th_dir+".csv")
                loss_th_exist = True 
            except FileNotFoundError:
                loss_th_data = None
                loss_th_exist = False
                print("missing: ", loss_th_dir)


            loss_node = node(loss_th, loss_th_exist, loss_th_data, {})
            loss_dict[loss_th] = loss_node
            
            for loss_factor in loss_factors:
                
                time_th = f'thresholds_time_{dataset_name}_{num_signals}_{time_step}_{sampling_period}_{loss_factor}'
                time_th_dir = f'{root_dir}/../data/thresholds/{dataset_name}/{time_th}'
                try:
                    time_th_data = pd.read_csv(time_th_dir+".csv")
                    time_th_exist = True 
                except FileNotFoundError:
                    time_th_data = None
                    time_th_exist = False
                    print("missing: ", time_th_dir)


                time_node = node(time_th, time_th_exist, time_th_data, {})
                loss_node.child_list[time_th] = time_node

                for time_factor in time_factors:
                    signal_th = f'thresholds_signal_{dataset_name}_{num_signals}_{time_step}_{sampling_period}_{loss_factor}_{time_factor}'
                    signal_th_dir = f'{root_dir}/../data/thresholds/{dataset_name}/{signal_th}'

                    try:
                        signal_th_data = pd.read_csv(signal_th_dir+".csv")
                        signal_th_exist = True 
                    except FileNotFoundError:
                        signal_th_data = None
                        signal_th_exist = False
                        print("missing: ", signal_th_dir)
                    signal_node = node(signal_th, signal_th_exist, signal_th_data, {})
                    time_node.child_list[signal_th] = signal_node

    return loss_dict

def generate_and_save_prediction_loss_per_file(args, file_name, file_path, run_type):
    
    dataset_name = args.dataset_name
    root_dir = args.root_dir
    num_signals = args.num_signals
    time_step = args.time_step
    sampling_period = args.sampling_period

    print("Loading dataset: ",file_name)
    x_seq, y_seq = load_data_create_images(args, file_name, file_path)
    
    print("Training input shape: ", x_seq.shape)

    if dataset_name =='syncan':
        model_dir = f"{root_dir}/../artifacts/models/{dataset_name}/autoendoer_canshield_{dataset_name}_{time_step}_{num_signals}_{sampling_period}.h5"
    elif dataset_name =='road':
        model_dir = f"{root_dir}/../artifacts/models/{dataset_name}/autoendoer_canshield_{dataset_name}_{time_step}_{num_signals}_{sampling_period}.h5"
    
    autoencoder = load_model(model_dir)

    if run_type == 'original' or run_type == 'org':
        x_recon = autoencoder.predict(x_seq)

    elif run_type == 'lite':
        model_dir = f"{root_dir}/../artifacts/models/{dataset_name}/autoendoer_canshield_org_{dataset_name}_{time_step}_{num_signals}_{sampling_period}"
        autoencoder.save(model_dir)
        converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)
        tflite_model = converter.convert()
        lite_model_dir = f"{root_dir}/../artifacts/models/{dataset_name}/autoendoer_canshield_lite_{dataset_name}_{time_step}_{num_signals}_{sampling_period}.tflite"
        open(lite_model_dir , "wb") .write(tflite_model)
        interpreter = tf.lite.Interpreter(model_path = lite_model_dir)
        # Get lite model

        # Get input/output information
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Adjust graph input to handle batch tensor
        interpreter.resize_tensor_input(input_details[0]['index'], x_seq.shape) #(batch_size, 512, 512, 3)

        # Adjust output #1 in graph to handle batch tensor
        interpreter.resize_tensor_input(output_details[0]['index'], x_seq.shape) #(batch_size, 512, 512, 3)

        # # Adjust output #2 in graph to handle batch tensor
        # output_shape = output_details[1]['shape']
        # interpreter.resize_tensor_input(output_details[1]['index'], (batch_input.shape[0], output_shape[1], output_shape[2], output_shape[3])) #(batch_size, 512, 512, 18)
        # Allocate for the resizing operations
        interpreter.allocate_tensors()
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], x_seq.astype(np.float32))
        # Run
        interpreter.invoke()
        # Output
        output_details = interpreter.get_output_details()
        x_recon = interpreter.get_tensor(output_details[0]['index'])

        # # Run inference
        # # print("interpreters", interpreters)
        # interpreter.allocate_tensors() 
        # #get input and output tensors
        # input_details = interpreter.get_input_details()
        # output_details = interpreter.get_output_details()

        # #set the tensor to point to the input data to be inferred
        # input_index = input_details[0]["index"]
        # interpreter.set_tensor(input_index, x_seq.astype(np.float32))
        # #Run the inference
        # interpreter.invoke()
        # output_details = interpreter.get_output_details()
        # # Output
        # x_recon = interpreter.get_tensor(output_details[0]['index'])
    
    y_train_prob_org = calc_loss(x_seq, x_recon)[:,:,:,0]
    return y_train_prob_org, y_seq

def generate_and_save_prediction_loss(args, file_dir_dict, y_prob_org_dict):

    # args.time_step = int(node.th_name.split("_")[2])
    # args.sampling_period = int(node.th_name.split("_")[3])

    time_step =args.time_step
    sampling_period = args.sampling_period
    dataset_name = args.dataset_name
    num_signals = args.num_signals
    root_dir = args.root_dir

    for file_name, file_path in tqdm(file_dir_dict.items()):         
        
        try: 
            y_prob_org_dict[f"{file_name}_{time_step}_{sampling_period}"]
            print(f"{file_name}_{time_step}_{sampling_period} exists.....")
        except:
            y_prob_org, _ = generate_and_save_prediction_loss_per_file(args, file_name, file_path, 'original')

            y_prob_org_dict[f"{file_name}_{time_step}_{sampling_period}"] = y_prob_org
    
    return y_prob_org_dict
    
def generate_and_save_loss_data(node, file_dir_dict, y_train_prob_org_dict, loss_factors):

    print("Starting : generate_and_save_loss_data")
    #do something
    args = node.th_name.split("_")
    th_type = args[1]
    dataset_name = args[2]
    num_signals = int(args[3])
    time_step = int(args[4])
    sampling_period = int(args[5])
    th_loss_df = pd.DataFrame([])

    print(node.th_name)
    for file_index, (file_name, file_path) in tqdm(enumerate(file_dir_dict.items())):         

        try:
            y_train_prob_org = y_train_prob_org_dict[f"{file_name}_{time_step}_{sampling_period}"].copy()
            y_train_prob = y_train_prob_org.reshape(-1,num_signals)
        except:
            print("Prediction data does not exist!!!")
    

        print(".....loss_factors.....")
        th_values = {}
        for Signal, y_train_prob_signal in enumerate(y_train_prob.T):

            for loss_factor in loss_factors:
                th = np.percentile(y_train_prob_signal, loss_factor)
                # th_values['file'] = file_name
                th_values['sampling_period'] = sampling_period
                th_values['time_step'] = time_step
                th_values['loss_factor'] = loss_factor
                th_values['th'] = th 
                th_values['Signal'] = Signal
                th_loss_df = pd.concat([th_loss_df, pd.DataFrame(th_values, index = [0])], ignore_index= True)
                #-------------------------------------------------------------------------------------------
    ths_loss = th_loss_df.groupby(['time_step', 'sampling_period', 'loss_factor', 'Signal']).mean().reset_index()
    file_dir_ths_loss = Path(f'../data/thresholds/{dataset_name}/{node.th_name}.csv')
    file_dir_ths_loss.parent.mkdir(exist_ok=True, parents=True)
    ths_loss.to_csv(file_dir_ths_loss, index = True, header = True)
    node.th_data = ths_loss
    node.th_exist = True
    return node

def generate_and_save_time_data(node, file_dir_dict, loss_df, y_train_prob_org_dict, time_factors):

    print("Starting : generate_and_save_loss_data")
    #do something
    args = node.th_name.split("_")
    th_type = args[1]
    dataset_name = args[2]
    num_signals = int(args[3])
    time_step = int(args[4])
    sampling_period = int(args[5])
    loss_factor = float(args[6])

    th_time_df = pd.DataFrame([])

    print(node.th_name)
    for file_index, (file_name, file_path) in tqdm(enumerate(file_dir_dict.items())):         

        try:
            y_train_prob_org = y_train_prob_org_dict[f"{file_name}_{time_step}_{sampling_period}"].copy()
        except:
            print("Prediction data does not exist!!!")
    
        print(".....time_factors.....")

        ths_loss_image = np.squeeze(loss_df[loss_df['loss_factor'] == loss_factor]['th'].values)

        y_train_prob_org_bin = (y_train_prob_org> ths_loss_image).astype(int).copy()
        y_train_prob_org_bin_count = np.sum(y_train_prob_org_bin, 1)/time_step

        print("y_train_prob_org_bin_count.shape", y_train_prob_org_bin_count.shape)

        th_values = {}
        for Signal, y_train_prob_org_bin_count_each in enumerate(y_train_prob_org_bin_count.T):
            for time_factor in time_factors:
                th = np.percentile(y_train_prob_org_bin_count_each, time_factor)
                # th_values['file'] = file_name
                th_values['time_step'] = time_step
                th_values['sampling_period'] = sampling_period
                th_values['loss_factor'] = loss_factor
                th_values['time_factor'] = time_factor
                th_values['th'] = th 
                th_values['Signal'] = Signal
                th_time_df = pd.concat([th_time_df, pd.DataFrame(th_values, index = [0])], ignore_index= True)
                #-----------

    ths_time = th_time_df.groupby(['time_step', 'sampling_period', 'loss_factor', 'time_factor', 'Signal']).mean().reset_index()
    file_dir_ths_time = Path(f'../data/thresholds/{dataset_name}/{node.th_name}.csv')
    file_dir_ths_time.parent.mkdir(exist_ok=True, parents=True)    
    ths_time.to_csv(file_dir_ths_time, index = True, header = True)
    node.th_data = ths_time
    node.th_exist = True
    return node

def generate_and_save_signal_data(node, file_dir_dict, loss_df, time_df, y_train_prob_org_dict, signal_factors):

    print("Starting : generate_and_save_signal_data")
    #do something
    args = node.th_name.split("_")
    th_type = args[1]
    dataset_name = args[2]
    num_signals = int(args[3])
    time_step = int(args[4])
    sampling_period = int(args[5])
    loss_factor = float(args[6])
    time_factor = float(args[7])

    th_signal_df = pd.DataFrame([])

    print(node.th_name)

    for file_index, (file_name, file_path) in tqdm(enumerate(file_dir_dict.items())):         

        try:
            y_train_prob_org = y_train_prob_org_dict[f"{file_name}_{time_step}_{sampling_period}"].copy()
        except:
            print("Prediction data does not exist!!!")
        #-------------  Training -----------------
        ths_loss_image = np.squeeze(loss_df[loss_df['loss_factor'] == loss_factor]['th'].values)
        y_train_prob_org_bin = (y_train_prob_org> ths_loss_image).astype(int).copy()
        y_train_prob_org_bin_count = np.sum(y_train_prob_org_bin, 1)/time_step

        ths_time_image = np.squeeze(time_df[time_df['time_factor'] == time_factor]['th'].values)

        y_train_prob_org_sig_count = (y_train_prob_org_bin_count> ths_time_image).astype(int).copy()
        y_train_prob_org_sig_count = np.sum(y_train_prob_org_sig_count, 1)/num_signals


        for signal_factor in signal_factors:
            th = np.percentile(y_train_prob_org_sig_count, signal_factor)
            th_values = {}
            # th_values['file'] = file_name
            th_values['time_step'] = time_step
            th_values['sampling_period'] = sampling_period
            th_values['loss_factor'] = loss_factor
            th_values['time_factor'] = time_factor
            th_values['signal_factor'] = signal_factor
            th_values['th'] = th 
            th_signal_df = pd.concat([th_signal_df, pd.DataFrame(th_values, index = [0])], ignore_index= True)

    th_signal = th_signal_df.groupby(['time_step', 'sampling_period', 'loss_factor', 'time_factor', 'signal_factor']).mean().reset_index()
    
    file_dir_th_signal = Path(f'../data/thresholds/{dataset_name}/{node.th_name}.csv')
    file_dir_th_signal.parent.mkdir(exist_ok=True, parents=True)    
    th_signal.to_csv(file_dir_th_signal, index = True, header = True)
    node.th_data = th_signal
    node.th_exist = True
    return node

def generate_remaining_threshold_data(args, loss_dict, file_dir_dict):
    
    y_train_prob_org_dict = {}
    
    time_steps = args.time_steps
    sampling_periods = args.sampling_periods
    loss_factors = args.loss_factors
    time_factors = args.time_factors
    signal_factors = args.signal_factors
    dataset_name = args.dataset_name
    num_signals = args.num_signals
    
    for time_step in time_steps:
        for sampling_period in sampling_periods:

            args.time_step = time_step
            args.sampling_period = sampling_period

            print(f"sampling_period: {sampling_period}")
            loss_th = f'thresholds_loss_{dataset_name}_{num_signals}_{time_step}_{sampling_period}'
            loss_node = loss_dict[loss_th]

            if loss_node.th_exist == False:
                print("Loss node doesn't exits..")
                y_train_prob_org_dict = generate_and_save_prediction_loss(args, file_dir_dict, y_train_prob_org_dict)
                loss_node = generate_and_save_loss_data(loss_node, file_dir_dict, y_train_prob_org_dict, loss_factors)

            for time_th, time_node in loss_node.child_list.items(): 
                if time_node.th_exist == False:
                    print("Time node doesn't exits..")
                    y_train_prob_org_dict = generate_and_save_prediction_loss(args, file_dir_dict, y_train_prob_org_dict)
                    time_node = generate_and_save_time_data(time_node, file_dir_dict, loss_node.th_data, y_train_prob_org_dict, time_factors)
                                
                for signal_th, signal_node in time_node.child_list.items():
                    if signal_node.th_exist == False:
                        print("Signal node doesn't exits..") 
                        y_train_prob_org_dict = generate_and_save_prediction_loss(args, file_dir_dict, y_train_prob_org_dict)
                        signal_node = generate_and_save_signal_data(signal_node, file_dir_dict, loss_node.th_data, time_node.th_data, y_train_prob_org_dict, signal_factors)

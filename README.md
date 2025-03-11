# CANShield
This repository provides a deep learning-based signal-level intrusion detection framework for the CAN bus. CANShield consists of three modules: 1) a data preprocessing module that handles the high-dimensional CAN data stream at the signal level and parses them into time series suitable for a deep learning model; 2) a data analyzer module consisting of multiple deep autoencoder (AE) networks, each analyzing the time-series data from a different temporal scale and granularity; and 3) finally an attack detection module that uses an ensemble method to make the final decision.

![CANShield Workflow](doc/canshield_workflow.jpg)


## Clone CANShield

```
git clone https://github.com/shahriar0651/canshield.git
cd canshield
```

## Install Mambaforge
### Download and Install Mambaforge
```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
chmod +x Mambaforge-$(uname)-$(uname -m).sh
./Mambaforge-$(uname)-$(uname -m).sh
```

### Create Environment

We provide two environment configurations ([environment_v1](dependency/environment_v1.yaml) & [environment_v2](dependency/environment_v2.yaml)) which are optimized for different compuation setup. Create `canshield` environment using anyone of these (or customize as needed).

**Environment Comparison Table**  

| **Component**     | **Environment V1**            | **Environment V2**            |  
|------------------|----------------------------|----------------------------|  
| **Python**       | 3.9                        | 3.9                        |  
| **TensorFlow**   | 2.10                       | 2.15                       |  
| **Keras**        | 2.10                       | 2.15                       |  
| **CUDA**        | 11.7                        | 12.8                        |  
| **cuDNN**        | 8.5.0.96                    | 8.9.6.50                    |  
| **NumPy**        | 1.26.4                      | 1.26.4                      |  
| **GPU Tested**   | RTX 2080 Ti                 | Tesla P40                   |  

```
conda env create --file dependency/environment_v1.yaml
```
Or update the existing env
```
conda env update --file dependency/environment_v1.yaml --prune
```

### Activate Environment
```
conda activate canshield
```

## Download Dataset

### Download SynCAN Dataset

```
cd src
chmod +x download_syncan_dataset.sh
./download_syncan_dataset.sh
```

Here is the tree file structure after downloading the synCAN dataset:
```
datasets/
└── can-ids/
    └── syncan
        ├── ambients
        │   ├── train_1.csv
        │   ├── train_2.csv
        │   ├── train_3.csv
        │   └── train_4.csv
        ├── attacks
        │   ├── test_continuous.csv
        │   ├── test_flooding.csv
        │   ├── test_plateau.csv
        │   ├── test_playback.csv
        │   └── test_suppress.csv
        ├── License terms.txt
        └── README.md
```

## Building CANShield

### Training multiple autoencoders
```
python run_development_canshield.py
```

## Evaluating CANShield

### Testing on the test dataset
```
python run_evaluation_canshield.py
```

## Visualizing Results

### Visualize results on the test dataset
```
python run_visualization_results.py
```

## Citation
```bibtex
@article{shahriar2023canshield,
  title={CANShield: Deep-Learning-Based Intrusion Detection Framework for Controller Area Networks at the Signal Level}, 
  author={Shahriar, Md Hasan and Xiao, Yang and Moriano, Pablo and Lou, Wenjing and Hou, Y. Thomas},
  journal={IEEE Internet of Things Journal}, 
  year={2023},
  volume={10},
  number={24},
  pages={22111-22127},
  doi={10.1109/JIOT.2023.3303271}
}
```

# CANShield
CAN IDS

# Clone CANShield

```
git clone https://github.com/shahriar0651/canshield.git
cd canshield
```

# Install Mambaforge
### Download and Install Mambaforge
```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
chmod +x Mambaforge-$(uname)-$(uname -m).sh
./Mambaforge-$(uname)-$(uname -m).sh
```
### Create Environement
```
mamba env create --file dependancy/environment.yaml
```
Or update the existing env
```
mamba env update --file dependancy/environment.yaml --prune
```

### Activate Environment
```
mamba activate canshield
```

# Download Dataset

### Download SynCAN Dataset

```
cd src
chmod +x download_syncan_dataset.sh
./download_syncan_dataset.sh
```

Here are the tree file structure after downloading the synCAN dataset:
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

# Building CANShield

### Training multiple autoencoders
```
python run_development_canshield.py
```

# Evaluating CANShield

### Testing on test dataset
```
python run_evaluation_canshield.py
```

# Visualizing Results

### Visualize results on test dataset
```
python run_visualization_results.py
```

# Citation
```
@article{shahriar2023canshield,
  title={CANShield: Deep Learning-Based Intrusion Detection Framework for Controller Area Networks at the Signal-Level},
  author={Shahriar, Md Hasan and Xiao, Yang and Moriano, Pablo and Lou, Wenjing and Hou, Y Thomas},
  journal={IEEE Internet of Things Journal},
  year={2023},
  publisher={IEEE}
}
```
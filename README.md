# canshield
This repo contains the data and code for the paper *Deep Learing-based Intrusion Detection System for Controller Area Networks at the Signal Level.*

1. Download the folder "datasets" from [here](https://drive.google.com/drive/folders/1U0lHjx99EQz47aNxWLKnfPMcjQD2MVGd?usp=sharing) and copy it within the project folder.
2. Download the folder "models" from [here](https://drive.google.com/drive/folders/1MTp7lsMhBPbWfw86kZ8DRszsYPeYfGwX?usp=share_link) and copy it within the project folder.
3. Run the file [CANShiled_Train_SynCAN.ipynb](CANShiled_Train_SynCAN.ipynb) to train new autoencoder with different sampling periods.
4. Run the file [CANShiled_Test_SynCAN.ipynb](CANShiled_Test_SynCAN.ipynb) to evaluate the performance of CANShield with different combinations of AEs and different hyperparameters.

The arxiv version of the paper can be found [here](https://arxiv.org/abs/2205.01306). 
Please cite the paper: 
```
@article{shahriar2022canshield,
  title={CANShield: Signal-based Intrusion Detection for Controller Area Networks},
  author={Shahriar, Md Hasan and Xiao, Yang and Moriano, Pablo and Lou, Wenjing and Hou, Y Thomas},
  journal={arXiv preprint arXiv:2205.01306},
  year={2022}
}
```

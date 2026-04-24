
# PFDL-SC: A Privacy-Preserving Federated Deep Learning Framework for scRNA-seq Clustering Analysis
PFDL-SC is a privacy-preserving federated deep learning clustering method for scRNA-seq clustering, designed to balance privacy requirements with model clustering performance.


## Project contents in /Application
+ base_SC.py: A foundational deep learning model for scRNA-seq clustering
+ DL_SC.py: Enhancing the base_SC model with federated learning training design
+ PFDL_SC.py: Implementing the privacy-preserving final scRNA-seq clustering model
+ client.py: Distributing the data to multiple clients for federated learning
+ metrics.py: Functions for computing model evaluation metrics
+ utils.py: Contains utility functions supporting the solution implementation


## System Requirements
python==3.9.18  
h5py==3.14.0
networkx==3.0 
numpy==1.26.4  
pandas==2.1.3  
scikit-learn==1.6.1  
scipy==1.10.0  
torch==2.1.1+cu118  
torch-geometric==2.4.0  
torch-scatter==2.1.2  
torchaudio==2.1.1  
torchvision==0.16.1  

## Datasets
A total of 15 real-world datasets, the dataset used for privacy attacks, and the dataset for batch effect correction will be provided after the paper is accepted.

## Files Description
+ `/Application`: examples of applying PPML-Omics, users can eaisy modify based on it to meet their own requirements
+ `/Attack`: source codes for attack experiments
+ `/Batch_effect_correction`: source codes for Batch effect correction experiments
+ `/Data`: Datasets used in all experiments


##### Example of centrally trained method (base_SC)

```
python base_SC.py --dataset Quake_10x_Limb_Muscle
```

##### Example of introducing federated learning and distributed training (FL_SC)

```
python DL_SC.py --com-time 100 --client 5 --epochs 10 --dataset Quake_10x_Limb_Muscle
```

##### Example of the final model incorporating differential privacy mechanisms (PFDL_SC)

```
python PFDL_SC.py --com-time 100 --client 5 --epochs 10 --delta 0.001 --l2_clip 2 --dataset Quake_10x_Limb_Muscle --epsilon 10
```


Options:
- `--com-time`: Number of federated training sessions
- `--client`: Number of clients 
- `--epochs`: Number of training epochs
- `--delta`: Privacy leakage probability for differential privacy
- `--l2_clip`: Upper limit of L2 global sensitivity
- `--dataset`: Name of dataset file
- `--epsilon`: Privacy budget for differential privacy



## Authors
Author email: guic.sun@gmail.com


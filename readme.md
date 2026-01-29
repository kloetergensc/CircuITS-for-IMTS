# CircuITS: Probabilistic Circuits for Irregular Multivariate Time Series Forecasting

This repository contains the official PyTorch implementation of the paper **"Probabilistic Circuits for Irregular Multivariate Time Series Forecasting"**.
### Prerequisites

The code is implemented in Python 3.11 using PyTorch.

Bash

```
pip install -r requirements.txt
```

### Datasets

The paper evaluates the model on four real-world datasets:

1. **USHCN:** Climate data (Precipitation, Snow Fall, Snow Depth, Max Temperature, Min Temperature ).
    
2. **PhysioNet-2012:** ICU mortality prediction data (37 vital signs).
    
3. **MIMIC-III:** Clinical database (96 variables).
    
4. **MIMIC-IV:** Updated clinical database (102 variables).
    

### 1. Synthetic Bifurcation Experiment

To reproduce the "Independent Multi-channel Bifurcation" experiment, which demonstrates CircuITS' ability to capture distinct modes where baselines like MOSES fail:

CircuITS:

```
python train_circuits.py --dataset bifurcation -ct 12 -ft 36 --plot
```

MOSES:
```
python src/baselines/moses/train_moses --dataset bifurcation -ct 12 -ft 36 --plot n_gaussians 10
```

### 2. Real-World Benchmarks

To train and evaluate CircuITS on the benchmark datasets (Table 1).

To reproduce the USHCN 36-12 dataset for 1st fold run: 

```
python train_circuits.py --components 2 hidden_dim 32  --flow_layers 2 --flow_hid_dim 10 --n_heads 2 --dataset ushcn -ct 36 -ft 12 -bs 64 --fold 0
```

Respectively for Physionet-2012, MIMIC-III and MIMIC-IV:

```
python train_circuits.py --components 4  --flow_layers 2 --flow_hid_dim 10 --hidden_dim 64 --n_heads 2 --dataset physionet2012 -ct 36 -ft 12 -bs 64 --fold 0
```

```
python train_circuits.py --components 4 --flow_layers 2 --flow_hid_dim 10 --hidden_dim 64 --n_heads 2 --dataset mimiciii  -ct 36 -ft 12 -bs 64 --fold 0
```

```
python train_circuits.py --components 2 --flow_layers 2 --flow_hid_dim 10 --hidden_dim 32 --n_heads 2 --dataset mimiciv  -ct 36 -ft 12 -bs 64 --fold 0 
```


**Hint**: Setting the parameter -ft (forecasting time) as 0 denotes the next-three-steps forecasting  scenario. -ct 36 -ft 3 would result in a different scenario (where the task is to predict the next three hours/months).  

## Citation

If you find this repository useful, please cite our paper (BibTeX will be updated upon de-anonymization):

Code snippet

```
@inproceedings{anonymous2025circuits,
  title={Probabilistic Circuits for Irregular Multivariate Time Series Forecasting},
  author={Anonymous Authors},
  booktitle={Under Review},
  year={2025}
}

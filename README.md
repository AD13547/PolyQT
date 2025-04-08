## PolyQT ##

Harbin Normal University </br>
[Pipeline](pipeline.pdf)
## Getting Started

### Installation

```
# create a new environment
$ conda create --name PolyQT python=3.9
$ conda activate PolyQT

# install requirements
$ conda install pytorch==1.12.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch
$ pip install PyYAML==6.0
$ pip install fairscale==0.4.6
$ conda install -c conda-forge transformers=4.20.1
$ conda install -c conda-forge rdkit=2022.3.4
$ conda install -c conda-forge scikit-learn==0.24.2
$ conda install -c conda-forge tensorboard==2.9.1
$ conda install -c conda-forge torchmetrics==0.9.2
$ conda install -c conda-forge packaging==21.0
$ conda install scipy=1.13.0 numpy=1.23.5 numpy-base=1.23.5
$ python -m pip install pennylane
```

### Dataset
We used six publicly available datasets for training quantum models. The sequences of these datasets are related to polymers only, with the inputs being the SMILES of the polymers, which are then labeled by a polymer tokenizer as the inputs to PolyQT. Specifically, data for Glass Transition Temperature and Polymer Density were obtained from the PolyInfo database and are based on experimental measurements (Exp), while the other four properties—Ionization Energy, Dielectric Constant, Refractive Index, and Crystallization Tendency—were adopted from the work of Kuenneth et al. (2021) and are derived from density functional theory (DFT) calculations. This diversity in data sources enables us to assess the model’s performance across both experimentally measured and computationally simulated properties.All datasets are provided in the data folder.

### Quantum Component
The model uses a quantum component with 8 qubits. In the experiments, different numbers of qubits (4, 5, 6, 7, and 8) were tested to evaluate the impact on performance.

Location in the code: `model_quantum.py` (Line 86)

## Run the Model

### PolyQT
To train PolyQT, where the configurations and detailed explaination for each variable can be found in `config_PolyQT.yaml`.
```
$ python PolyQT.py
```

### Transformer
To train the baseline Transformer model for validating PolyQT's effectiveness in addressing data sparsity, the configurations and detailed explanations for each variable can be found in `config_Transformer.yaml`.
```
$ python Transformer.py
```

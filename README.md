# Social Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation </br><sub><sub> Bruno Deprez, Toon Vanderschueren, Tim Verdonck, Bart Baesens, Wouter Verbeke [[Preprint]](https://arxiv.org/abs/2405.19383)</sub></sub>

[![License: MIT](https://img.shields.io/badge/License-MIT-orange.svg)](https://opensource.org/licenses/MIT)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/network-analytics-for-anti-money-laundering-a/fraud-detection-on-elliptic-dataset)](https://paperswithcode.com/sota/fraud-detection-on-elliptic-dataset?p=network-analytics-for-anti-money-laundering-a)

The source code of the experimental evaluation of the paper *Social Network Analytics for Anti-Money Laundering -- A Systematic Literature Review and Experimental Evaluation*. A preprint version of the work is available on arXiv at https://arxiv.org/abs/2405.19383.

This repository provides an implementation of different network learning techniques in a uniform manner.

## Data
The main experiment is done on the elliptic dataset. Additional code is present that was used to verify the correct working of the code on the Cora dataset. 

This repository does not provide any data, due to size constraints. The data can be found online using the following links:
- [Cora](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html#torch_geometric.datasets.Planetoid)
- [Elliptic](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.EllipticBitcoinDataset.html#torch_geometric.datasets.EllipticBitcoinDataset)
- [IBM-AML](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)

## Repository structure
The structure includes folders and scripts/notebooks containing code. The files with results, i.e., csv-files, are note shown.
This repository is organised as follows:
```bash
|-data
|-notebooks
    |-AnalysisResults.ipynb
|-res
|-scripts
    |-train_supervides.py
    |-test_supervised.py
    |-train_unsupervides.py
    |-test_unsupervised.py
|-src
    |-data
        |-DatasetConstruction.py
    |-methods
        |-utils
            |-decoder.py
            |-functionsNetworKit.py
            |-functionsNetworkX.py
            |-functionsTorch.py
            |-GNN.py
            |-isolation_forest.py
        |-evaluation.py
        |-experiments_supervised.py
        |-experiments_unsupervised.py
|-utils
    |-Network.py
```

## Installing
We have provided a `requirements.txt` file:
```bash
pip install -r requirements.txt
```

To create a new environment, we advise to use the following steps. This helps avoid dependency issues when loading and running node2vec. 
```bash
conda create -n benchmark_AML python=3.10.16 jupyter numpy scipy matplotlib
conda install networkx
pip install networkit
pip install scikit-learn
conda install pandas
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
pip install torch-geometric==2.6.1
pip install optuna
```

## Citing
Please cite our paper and/or code as follows:
*Use the BibTeX citation*

```tex

@article{deprez2024networkevaluation,
      title={Network Analytics for Anti-Money Laundering -- A Systematic Literature Review and Experimental Evaluation}, 
      author={Bruno Deprez and Toon Vanderschueren and Bart Baesens and Tim Verdonck and Wouter Verbeke},
      year={2024},
      journal={arXiv preprint arXiv:2405.19383},
      eprint={2405.19383},
      archivePrefix={arXiv},
      primaryClass={cs.SI},
      url={https://arxiv.org/abs/2405.19383}, 
}

```

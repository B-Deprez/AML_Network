</br><sub><sub> Bruno Deprez, Toon Vanderschueren, Wouter Verbeke, Bart Baesens, Tim Verdonck </sub></sub>

The source code of the experimental evaluation of the paper *Social Network Analytics for Anti-Money Laundering -- A Systematic Literature Review and Experimental Evaluation*.

It provides an implementation of different network learning techniques in a uniform manner.

## Data
The main experiment is done on the elliptic dataset. Additional code is present that was used to verify the correct working of the code on the Cora dataset. 

This repository does not provide any data, due to size constraints. The data can be found online using the following links:
- [Cora](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html#torch_geometric.datasets.Planetoid)
- [Elliptic](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.EllipticBitcoinDataset.html#torch_geometric.datasets.EllipticBitcoinDataset)

## Repository structure
The structure includes folders and scripts/notebooks containing code. The files with results, i.e., csv-files, are note shown.
This repository is organised as follows:
```bash
|-data
|-notebooks
    |-AnalysisResults.ipynb
|-res
|-scripts
    |-test.py
    |-train.py
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
        |-evaluation.py
        |-experiments.py
|-utils
    |-Network.py
```

## Installing
We have provided a `requirements.txt` file:
```bash
pip install -r requirements.txt
```
Please use the above in a newly created virtual environment to avoid clashing dependencies.

## Citing
Please cite our paper and/or code as follows:
*Use the BibTeX citation*

```tex

@article{}

```

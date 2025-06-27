# DualCellChat


This is the code page for the paper "Reconstructing cell-cell interaction network in single-cell spatial transcriptomics via directed heterogeneous graph autoencoder".

Taking advantage of spatial transcriptomic data and graph autoencoders, we present a directed heterogeneous graph autoencoder-based approach DualCellChat to learn from incomplete spatial organization to reconstruct a complete and accurate CCI network in single cell spatial transcriptomics. Through application to five single-cell spatial datasets from four different technologies, we demonstrate that DualCellChat outperforms existing deep learning-based methods on single-cell spatial transcriptomics datasets and can infer the direction of cellular interactions. Furthermore, we introduce downstream analysis to infer signature genes involved in short- and long-range cellular interactions from the reconstructed CCI network and infer significant ligand-receptor pairs for specific cell types.

![Image text](https://github.com/JinxianHu/DualCellChat/blob/main/figures/fig1.png)

## Install Dependencies
The following Python packages are required in addition to standard `tensorflow` and `pytorch` machine learning frameworks:
- `torch-geometric`: https://github.com/rusty1s/pytorch_geometric
- `gravity-gae`: https://github.com/deezer/gravity_graph_autoencoders
- See dependency versions in `requirements.txt`.

## Run Experiments
You can run a demo from the command line:

#### Training of homogeneous graph autoencoder models
```bash
python train.py --dataset=HDST_ob --model=digae --alpha=0.0 --beta=0.2 --epochs=40 --nb_run=5 --logfile=digae_HDST_ob_grid_search.json --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True
```

#### Training of heterogeneous graph autoencoder models
```bash
python train_HEAT.py --dataset=StereoSeq --model=HEAT_digae --alpha=0.0 --beta=0.2 --epochs=40 --nb_run=5 --logfile=HEAT_digae_StereoSeq_grid_search.json --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True
```

train.sh provides a complete training script for running DualCellChat.

## Datasets

The datasets used in this study are available from the following websites: 

(1) seqFISH dataset: https://bitbucket.org/qzhu/smfish-hmrf/src/master/; 

(2) MERFISH dataset: https://datadryad.org/stash/dataset/doi:10.5061/dryad.8t8s248;

(3) Two HDST datasets: https://portals.broadinstitute.org/single_cell/study/SCP420;

(4) The processed Stereo-seq dataset: https://github.com/JinmiaoChenLab/SEDR_analyses. 

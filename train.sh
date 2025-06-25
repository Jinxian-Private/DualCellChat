####################### MERFISH data
##### missing edges
# python train.py --dataset=MERFISH --model=digae --alpha=0.0 --beta=0.2 --epochs=40 --nb_run=5 --logfile=digae_merfish_grid_search.json --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True --test_ratio=0.9

##### noise edges
# python train.py --dataset=MERFISH --model=digae --alpha=0.0 --beta=0.2 --epochs=40 --nb_run=5 --logfile=digae_merfish_grid_search.json --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True --test_ratio=0.1 --noise=True --noise_disp=1.1

##### enrichment analysis
# python train.py --dataset=MERFISH --model=digae --alpha=0.0 --beta=0.2 --epochs=40 --nb_run=5 --logfile=digae_merfish_grid_search.json --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True --test_ratio=0.1 --enrichment=True

##### sensitivity analysis, genes
# python train.py --dataset=MERFISH --model=digae --alpha=0.0 --beta=0.2 --epochs=40 --nb_run=5 --logfile=digae_merfish_grid_search.json --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True --test_ratio=0.1 --sensitivity=True

##### sensitivity analysis, cell type specific LR pairs 
# python train.py --dataset=MERFISH --model=digae --alpha=0.0 --beta=0.2 --epochs=40 --nb_run=5 --logfile=digae_merfish_grid_search.json --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True --test_ratio=0.1 --sensitivity_celltype_LR=True


####################### seqFISH data
# python train.py --dataset=seqFISH --model=digae --alpha=0.0 --beta=0.2 --epochs=40 --nb_run=5 --logfile=digae_seqFISH_grid_search.json --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True --test_ratio=0.1

##### missing edges
# python train.py --dataset=seqFISH --model=digae --alpha=0.0 --beta=0.2 --epochs=40 --nb_run=5 --logfile=digae_seqFISH_grid_search.json --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True --test_ratio=0.9

##### noise edges
# python train.py --dataset=seqFISH --model=digae --alpha=0.0 --beta=0.2 --epochs=40 --nb_run=5 --logfile=digae_seqFISH_grid_search.json --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True --test_ratio=0.1 --noise=True --noise_disp=1.1

##### enrichment analysis
# python train.py --dataset=seqFISH --model=digae --alpha=0.0 --beta=0.2 --epochs=40 --nb_run=5 --logfile=digae_seqFISH_grid_search.json --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True --test_ratio=0.1 --enrichment=True

##### sensitivity analysis
# python train.py --dataset=seqFISH --model=digae --alpha=0.0 --beta=0.2 --epochs=40 --nb_run=5 --logfile=digae_HDST_cancer_grid_search.json --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True --test_ratio=0.1 --sensitivity=True


####################### HDST_ob data
# python train.py --dataset=HDST_ob --model=digae --alpha=0.0 --beta=0.2 --epochs=40 --nb_run=5 --logfile=digae_HDST_ob_grid_search.json --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True --test_ratio=0.1 

##### missing edges
# python train.py --dataset=HDST_ob --model=digae --alpha=0.0 --beta=0.2 --epochs=40 --nb_run=5 --logfile=digae_HDST_ob_grid_search.json --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True --test_ratio=0.5

#### noise edges
# python train.py --dataset=HDST_ob --model=digae --alpha=0.0 --beta=0.2 --epochs=40 --nb_run=5 --logfile=digae_HDST_ob_grid_search.json --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True --test_ratio=0.1 --noise=True --noise_disp=1.0

###### fake edges
# python train.py --dataset=HDST_ob --model=digae --alpha=0.0 --beta=0.2 --epochs=40 --nb_run=5 --logfile=digae_HDST_ob_grid_search.json --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True --task=fake

###### dropout1: delete different ratio genes
# python train.py --dataset=HDST_ob --model=digae --alpha=0.0 --beta=0.2 --epochs=40 --nb_run=5 --logfile=digae_HDST_ob_grid_search.json --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True --test_ratio=0.1 --dropout_gene=True --dropout_gene_ratio=0.1

##### dropout2: transfer different ratio of non-zero value to zero  
# python train.py --dataset=HDST_ob --model=digae --alpha=0.0 --beta=0.2 --epochs=40 --nb_run=5 --logfile=digae_HDST_ob_grid_search.json --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True --test_ratio=0.1 --dropout_value=True --dropout_value_ratio=0.1

#### enrichment analysis
# python train.py --dataset=HDST_ob --model=digae --alpha=0.0 --beta=0.2 --epochs=40 --nb_run=5 --logfile=digae_HDST_ob_grid_search.json --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True --test_ratio=0.1 --enrichment=True

#### sensitivity analysis, randomly select edges as testing edges
# python train.py --dataset=HDST_ob --model=digae --alpha=0.0 --beta=0.2 --epochs=40 --nb_run=5 --logfile=digae_HDST_ob_grid_search.json --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True --test_ratio=0.1 --sensitivity=True

#### sensitivity analysis, select long range edges as tesing edges
# python train.py --dataset=HDST_ob --model=digae --alpha=0.0 --beta=0.2 --epochs=40 --nb_run=5 --logfile=digae_HDST_ob_grid_search.json --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True --test_ratio=0.1 --sensitivity_long=True

#### visulize analysis
# python train.py --dataset=HDST_ob --model=digae --alpha=0.0 --beta=0.2 --epochs=40 --nb_run=5 --logfile=digae_HDST_ob_grid_search.json --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True --test_ratio=0.1 --visualize=True

####################### HDST_cancer data
#### missing edges
# python train.py --dataset=HDST_cancer --model=digae --alpha=0.0 --beta=0.2 --epochs=40 --nb_run=5 --logfile=digae_HDST_cancer_grid_search.json --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True --test_ratio=0.9

#### noise edges
# python train.py --dataset=HDST_cancer --model=digae --alpha=0.0 --beta=0.2 --epochs=40 --nb_run=5 --logfile=digae_HDST_cancer_grid_search.json --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True --test_ratio=0.1 --noise=True --noise_disp=1.0

#### enrichment analysis
# python train.py --dataset=HDST_cancer --model=digae --alpha=0.0 --beta=0.2 --epochs=40 --nb_run=5 --logfile=digae_HDST_cancer_grid_search.json --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True --test_ratio=0.1 --enrichment=True

##### sensitivity analysis
# python train.py --dataset=HDST_cancer --model=digae --alpha=0.0 --beta=0.2 --epochs=40 --nb_run=5 --logfile=digae_HDST_cancer_grid_search.json --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True --test_ratio=0.1 --sensitivity=True

#### dropout1: delete different ratio genes
# python train.py --dataset=HDST_cancer --model=digae --alpha=0.0 --beta=0.2 --epochs=40 --nb_run=5 --logfile=digae_HDST_cancer_grid_search.json --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True --test_ratio=0.1 --dropout_gene=True --dropout_gene_ratio=0.9

#### dropout2: transfer different ratio of non-zero value to zero  
# python train.py --dataset=HDST_cancer --model=digae --alpha=0.0 --beta=0.2 --epochs=40 --nb_run=5 --logfile=digae_HDST_cancer_grid_search.json --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True --test_ratio=0.1 --dropout_value=True --dropout_value_ratio=0.9

####################### Stereo-seq data
##### Stereo-seq, all gene PCA, digae model, load_data_from_csv
# python train.py --dataset=StereoSeq --model=digae --alpha=0.0 --beta=0.2 --epochs=40 --nb_run=5 --logfile=digae_StereoSeq_grid_search.json --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True --seed=0

##### Stereo-seq, all gene PCA, HEAT_digae, load_data_from_csv
# python train_HEAT.py --dataset=StereoSeq --model=HEAT_digae --alpha=0.0 --beta=0.2 --epochs=40 --nb_run=5 --logfile=digae_StereoSeq_grid_search.json --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True

##### Stereo-seq, all gene PCA, HEAT_digae, load_data_from_csv, cell_type_random_initialize  
# python train_HEAT.py --dataset=StereoSeq --model=HEAT_digae --alpha=0.0 --beta=0.2 --epochs=40 --nb_run=5 --logfile=digae_StereoSeq_grid_search.json --learning_rate=0.005 --hidden=64 --dimension=32 --celltype_nosie=random_generate --validate=True --seed=2 

##### Stereo-seq, LR genes, load_data_from_anndata
# python train.py --dataset=StereoSeq --model=digae --alpha=0.0 --beta=0.2 --epochs=120 --nb_run=5 --logfile=digae_StereoSeq_grid_search.json --learning_rate=0.005 --hidden=64 --dimension=32 --validate=True --feature_vector_type=LR

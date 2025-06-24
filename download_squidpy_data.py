import scanpy as sc
import squidpy as sq

import numpy as np

adata = sq.datasets.merfish()
# adata = sq.datasets.seqfish()
print(adata)
print(adata.obs['Cell_class'])
print(adata.obs['Bregma'].values)

Bregma = adata.obs['Bregma'].values
Bregma_list = list(set(Bregma.tolist()))
for Bregma_i in Bregma_list:
	print('Bregma_i: ',Bregma_i)
	test = adata[adata.obs.Bregma==Bregma_i]
	print(test)

# Animal_ID = adata.obs['Animal_ID'].values
# print(list(set(Animal_ID.tolist())))
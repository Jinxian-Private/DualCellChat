import pandas as pd 
import numpy as np

input_LR = 'data/LR/CellChatDB.human/interaction.csv'
LR_df = pd.read_csv(open(input_LR)) 
LR_df = LR_df[LR_df['annotation']=='Cell-Cell Contact']
# LR_df = LR_df[LR_df['annotation']=='Secreted Signaling']
# LR_df = LR_df[LR_df['annotation']=='ECM-Receptor']

interaction = LR_df['interaction_name'].values
Ligand_list = []
Receptor_list = []
Receptor_all = []
for interaction_i in interaction:
    interaction_i = interaction_i.split('_')
    Ligand_list.append(interaction_i[0])
    Receptor_list.extend(interaction_i[1:]) 
print(Ligand_list)
print(Receptor_list)
for gene in Ligand_list:
	print(gene)
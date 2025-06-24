import numpy as np 
import pandas as pd 

def read_LR(input_LR,LR_type,MERFISH_genes):
    LR_df = pd.read_csv(open(input_LR)) 
    if LR_type:
        LR_df = LR_df[LR_df['annotation']==LR_type]
    # LR_df = LR_df[LR_df['annotation']=='Cell-Cell Contact']
    # LR_df = LR_df[LR_df['annotation']=='Secreted Signaling']
    # LR_df = LR_df[LR_df['annotation']!='ECM-Receptor']

    print(LR_df)
    interaction = LR_df['interaction_name'].values

    LR_list = []
    for interaction_i in interaction:
        interaction_i = interaction_i.split('_')
        print(interaction_i)
        ligand = interaction_i[0]
        receptors = interaction_i[1:]
        if ligand in MERFISH_genes:
        	for receptor in receptors:
        		if receptor in MERFISH_genes:
        			LR_list.append(ligand.title()+'-'+receptor.title())
    return LR_list

df_MERFISH = pd.read_csv('data/MERFISH/counts.csv')
MERFISH_genes = df_MERFISH.columns.values
MERFISH_genes = [gene.upper() for gene in MERFISH_genes]

#############  load ligand-Receptor, and generate sender_reciever_matrix 
LR_file = 'data/LR/CellChatDB.mouse/interaction.csv'
LR_type = False #'Cell-Cell Contact', 'Secreted Signaling',False, 'ECM-Receptor'
LR_list = read_LR(LR_file,LR_type,MERFISH_genes)
LR_list.append('Pnoc-Oprl1')
print(np.array(LR_list))
np.save('data/MERFISH/CellChatDB_LR.npy',np.array(LR_list))
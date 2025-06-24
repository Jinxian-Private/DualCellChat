import numpy as np
import ot
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import random
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import os
import getopt
import sys

import warnings
warnings.filterwarnings('ignore')

def generate_gene_distribution(gene, count_df):
    '''
    input:
        - gene : gene (type : str)(e.g. 'gene', 'gene1+gene2')
        - count_df : normalized count matrix (gene*spot) (type : pd.DataFrame)
    output:
        - flag : {0,1} : 0 -> count_df 中有该gene；1 --》 count_df 无该gene
        - gene_df : Non-zero gene expression value dataframe with spot name as its index (e.g. 'spot1')
                (type : pd.DataFrame)
                
    spot num < min_spot --> trimed
    '''
    
    spot_list = list(count_df.columns)
    
    # check single/multi genes
    if '+' not in gene:
        # simple
        if gene in list(count_df.index):
            gene_df = count_df.loc[gene,:]
            gene_df = gene_df.loc[gene_df > 0]
            if gene_df.shape[0] == 0:
                return 1, None
            return 0, gene_df
        else:
#             print('{} not in counts df'.format(gene))
            return 1, None
    else:
        # complex
        gene = gene.split('+')
        for g in gene:
            if g not in list(count_df.index):
#                 print('{} not in counts df'.format(g))
                return 1, None
        gene_df = count_df.loc[gene[0],:]
        sub_num = len(gene)
        for g in gene[1:]:
            gene_df = gene_df*count_df.loc[g,:]
        gene_df = gene_df.apply(lambda x: np.power(x, 1/sub_num))
        gene_df = gene_df.loc[gene_df > 0]
        if gene_df.shape[0] == 0:
            return 1, None
        return 0, gene_df


def cal_gene_distance(ip, count_df, spot_pos_df, ip_type, reg = 0.001, iternum = 200, shuffle = False):
    '''
    input:
        - gene_1/2 : genes to be calculated
        - spot_dis_df : spot distance matrix, returned by generate_distance_matrix(spot_loc_path)
        - celltype_percent_1/2 : celltype percentage matrix calculated by spatialDWLS for celltype 
    output:
        - gene_distance : 如果有基因不在count里面返回的都是None
        
    '''
    gene_a, gene_b = ip.split(' - ')
    gene_a = gene_a.strip().replace('(','').replace(')','')
    gene_b = gene_b.strip().replace('(','').replace(')','')
    
    flag_a, gene_df_a = generate_gene_distribution(gene_a,count_df)
    flag_b, gene_df_b = generate_gene_distribution(gene_b,count_df)
    if flag_a + flag_b > 0:
        return None

    # gene_pos_a = np.array(spot_pos_df.loc[[spot in list(gene_df_a.index) for spot in list(spot_pos_df.index)],['X','Y']])
    # gene_pos_b = np.array(spot_pos_df.loc[[spot in list(gene_df_b.index) for spot in list(spot_pos_df.index)],['X','Y']])
    gene_pos_a = np.array(spot_pos_df.loc[[spot in list(gene_df_a.index) for spot in list(spot_pos_df.index)],['row','col']])
    gene_pos_b = np.array(spot_pos_df.loc[[spot in list(gene_df_b.index) for spot in list(spot_pos_df.index)],['row','col']])

    # if shuffle == False:
    #     gene_pos_a = np.array(spot_pos_df.loc[[spot in list(gene_df_a.index) for spot in list(spot_pos_df.index)],['row','col']])
    #     gene_pos_b = np.array(spot_pos_df.loc[[spot in list(gene_df_b.index) for spot in list(spot_pos_df.index)],['row','col']])
    # else:
    #     joined_spot_list = list(spot_pos_df.index)
        
    #     r_select_spot_a = random.sample(joined_spot_list, gene_df_a.shape[0])
    #     r_select_spot_b = random.sample(joined_spot_list, gene_df_b.shape[0])
    #     gene_pos_a = np.array(spot_pos_df.loc[[spot in r_select_spot_a for spot in list(spot_pos_df.index)],['row','col']])
    #     gene_pos_b = np.array(spot_pos_df.loc[[spot in r_select_spot_b for spot in list(spot_pos_df.index)],['row','col']])

    cost_matrix = ot.dist(gene_pos_a,gene_pos_b,metric='euclidean')
    x = cost_matrix.flatten()
    if ip_type == 'Secreted Signaling':
        color = 'green'
        y = np.random.uniform(0,100,np.shape(x))
    if ip_type == 'Cell-Cell Contact':
        color = 'blue'
        y = np.random.uniform(110,210,np.shape(x))
    if ip_type == 'ECM-Receptor':
        color = 'red'
        y = np.random.uniform(220,320,np.shape(x))

    print(x)
    print(y)
    # plt.hist(cost)
    plt.scatter(x,y,s=0.01,c=color)
    # plt.show()

    gene_distance_dir = ot.sinkhorn2(np.array(gene_df_a),np.array(gene_df_b), cost_matrix/cost_matrix.max(), reg = reg, numItermax=iternum)

    ### the R-L dir
    cost_matrix = ot.dist(gene_pos_b,gene_pos_a,metric='euclidean')
    gene_distance_rev = ot.sinkhorn2(np.array(gene_df_b),np.array(gene_df_a), cost_matrix/cost_matrix.max(), reg = reg, numItermax=iternum)
    
    gene_distance = (gene_distance_dir+gene_distance_rev)/2

    return gene_distance


def data_prepare(count_path,gene_name_list,min_spot=5):
    ### filter genes
    # count_df = pd.read_csv(count_path) #, sep = '\t',index_col = 0
    # colums_name = count_df.columns.values.tolist()
    # colums_name_upper = [column_i.upper() for column_i in colums_name]
    # count_df.columns = colums_name_upper
    # count_df = count_df.T
    count_df = pd.read_csv(count_path, sep = '\t',index_col = 0)

    select_index = [gene in gene_name_list for gene in list(count_df.index)]
    count_df = count_df.loc[select_index,:]

    select_index = list((count_df != 0).sum(axis=1) >= min_spot)
    count_df = count_df.loc[select_index,:]

    return count_df


def processing(idx_path,
               count_path,
               output_path,
               cc_ip_df,
               cc_gene_name,
               min_spot=5,
               reg=0.001,
                    iternum=500,
                    shuffle_num=500
              ):

    ## spot postion & spatial counts
    cc_ip_dic = {}
    for i in range(cc_ip_df.shape[0]):
        cc_ip_dic.update({cc_ip_df.iloc[i,2]:{'pathway':cc_ip_df.iloc[i,0],'ip_type':cc_ip_df.iloc[i,1]}})

    # pos_df = pd.read_csv(idx_path) #,index_col = 0
    pos_df = pd.read_csv(idx_path,sep='\t',index_col = 0)
    print(pos_df)
    count_df = data_prepare(count_path,cc_gene_name, min_spot=min_spot)
    print(count_df)
    
    # filter avi ips (all l/r have expression in st data)
    def select_expressed_lr(df_line, all_genes):
        '''
        find out whether both ligand and recepter genes can be found in sc data
        '''
        flag_l = df_line['ligand'] in all_genes
        if df_line['receptor'].count(',') == 0: # receptor with no subunit
            flag_r = df_line['receptor'] in all_genes
        else:  # receptor with multi-subunits
            flag_r = True
            for tmp_r in df_line['receptor'].split(','):
                tmp_flag = tmp_r in all_genes
                flag_r = flag_r and tmp_flag
        return flag_l and flag_r

    all_genes = list(count_df.index)
    select_index = cc_ip_df.apply(lambda x: select_expressed_lr(x, all_genes), axis = 1)
    cc_ip_df = cc_ip_df.loc[select_index,:]
    cc_ip_list = list(cc_ip_df['interaction_name_2'])
    print(cc_ip_list)

    dis_ori_list = []
    for ip in cc_ip_list:
        print(ip)
        print(cc_ip_dic[ip])
        ip_type = cc_ip_dic[ip]['ip_type']
        print(ip_type)
        dis_ori = cal_gene_distance(ip,count_df,pos_df,ip_type,reg=reg, iternum=iternum)
        dis_ori_list.append(dis_ori)
    plt.show()
    print(dis_ori_list)
    

if __name__ == '__main__':
    dataname='ST_A3_GSM4797918' #'HDST_ob'

    if dataname=='HBC':
        file_prefix = 'generated_data/V1_Breast_Cancer_Block_A_Section_1/'
    if dataname=='HDST_ob':
        file_prefix = 'data/HDST_ob/'
    if dataname=='ST_A3_GSM4797918':
        file_prefix = 'example_data/ST_A3_GSM4797918/data/processed/'


    st_coord_path = file_prefix + 'st_coord.tsv' #'coord.csv'
    st_count_path = file_prefix + 'st_counts.tsv' #'counts.csv'
    output_path = 'ip_dist/'+dataname+'/'

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # cc_ip_path = '/fs/home/liuzhaoyang/data/cc_ip/cc_ip_all_multi_split_deduplicates.tsv'
    cc_ip_path = 'cci_database/cc_ip_multi_split.tsv'

    cc_ip_df = pd.read_csv(cc_ip_path,sep='\t',index_col=0)
    print(cc_ip_df)

    all_ips = list(cc_ip_df.index)
    cc_gene_name = []
    for ip in all_ips:
        cc_gene_name += ip.split('_')
    cc_gene_name = list(set(cc_gene_name))
    print(cc_gene_name)

    processing(st_coord_path,st_count_path,output_path,cc_ip_df,cc_gene_name,shuffle_num=500, reg=0.001)


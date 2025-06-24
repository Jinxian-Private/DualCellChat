import os
os.environ['PYTHONHASHSEED'] = '0'

import matplotlib
havedisplay = "DISPLAY" in os.environ
if havedisplay:  #if you have a display use a plotting backend
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
import networkx as nx
import pandas as pd 


def polyfit_metrics(metrics_list):
    # figure, ax = plt.subplots(figsize=figsize, dpi=100)

    # ax = plt.subplot(111, facecolor='linen')

    # 多项式拟合
    f1 = np.polyfit(range(1,len(metrics_list)+1), np.array(metrics_list), 6)
    p1 = np.poly1d(f1)
    yvals1 = p1(range(1,len(metrics_list)+1))
    return yvals1

# "epoch",    "score",    ["AUPRC"], "AUPRC"
# fig_xlabel, fig_ylabel, fig_legend, filename, figsize=(13,9), color='orange'

current_dir = os.getcwd()
print("当前工作目录:", current_dir)
# os.chdir('C:\Users\huo\Desktop\细胞通讯工作修改\4.CCI\DiGAE-CCI')

line_compare = False

Box_noise = False
line_noise = True

Box_missing = False

line_dropout_gene = False
line_dropout_value = False

line_LR = False
Box_LR = False

line_HEAT = False
Box_HEAT = False

dataname = 'HDST_ob' #'HDST_ob','HDST_cancer','MERFISH','seqFISH'; 'HBC'; 'StereoSeq'

if line_HEAT == True:
	color_list = ['cornflowerblue','limegreen','palevioletred','orange']

	for iter_num in range(5):
		figure, ax = plt.subplots(figsize= (13,9), dpi=100)

		df_auc = pd.read_csv('results/AUC/'+'StereoSeq_HEAT'+'/'+dataname+'_missingratio0.1_iter'+str(iter_num)+'.csv') #missingratio, testratio
		AUC_list = df_auc['auc'].values.tolist()
		epochs_list = df_auc['epoch'].values.tolist()
		yvals = polyfit_metrics(AUC_list)
		ax.plot(range(1,len(epochs_list)+1), yvals, color='cornflowerblue', linestyle='-', linewidth=6,label='Heterogeneous GCN')

		df_auc = pd.read_csv('results/AUC/'+'StereoSeq_digae'+'/'+dataname+'_missingratio0.1_iter'+str(iter_num)+'.csv') #missingratio, testratio
		AUC_list = df_auc['auc'].values.tolist()
		epochs_list = df_auc['epoch'].values.tolist()
		yvals = polyfit_metrics(AUC_list)
		ax.plot(range(1,len(epochs_list)+1), yvals, color='orange', linestyle='-', linewidth=6,label='Homogeneous GCN')

		# plt.legend(handles=[],labels=['1','2','3','4'])

		plt.tick_params(labelsize=20)
		labels = ax.get_xticklabels() + ax.get_yticklabels()
		[label.set_fontname('Arial') for label in labels]

		font1 = {'family':'Arial','weight':'normal','size':25,}
		plt.xlabel("epoch", font1)
		plt.ylabel("AUROC", font1)

		# font2 = {'family':'Arial','size':25,}
		# leg = plt.legend(["AUPRC"], bbox_to_anchor=(1.02, 0), loc='lower right', borderaxespad=0, prop=font2)
		# leg.get_frame().set_linewidth(0.0)

		font2 = {'family':'Arial','size':15,}
		plt.legend(frameon=False,prop=font2)
		# plt.show()
		file_name = 'plot/AUC/'+dataname+'/AUPRC_HEAT_iter'+str(iter_num)+'.png'
		plt.savefig(file_name)
		plt.close()

if Box_HEAT == True:
	color_list = ['cornflowerblue','limegreen','palevioletred','orange']

	for iter_num in range(5):
		ratio_list = ['Heterogeneous GCN','Homogeneous GCN']
		df_box = pd.DataFrame(columns=ratio_list)
		for noise_ratio in ratio_list:
			AUC_list = []
			for iter_num in range(5):
				if noise_ratio == 'Heterogeneous GCN':
					df_auc = pd.read_csv('results/AUC/'+'StereoSeq_HEAT'+'/'+dataname+'_missingratio0.1_iter'+str(iter_num)+'.csv')
					AUC_list.append(df_auc['auc'].values[-1])
				elif noise_ratio == 'Homogeneous GCN':
					df_auc = pd.read_csv('results/AUC/'+'StereoSeq_digae'+'/'+dataname+'_missingratio0.1_iter'+str(iter_num)+'.csv')
					AUC_list.append(df_auc['auc'].values[-1])

			df_box[noise_ratio] = AUC_list
			# ax.plot(range(1,len(epochs_list)+1), AUC_list, color='orange', linestyle='-', linewidth=6)

	print(df_box)
	sns.boxplot(data=df_box)
	# plt.show()
	font1 = {'family':'Arial','weight':'normal','size':12}
	plt.xlabel('missing edges ratio',font1)
	plt.ylabel('AUROC',font1)
	plt.savefig('plot/AUC/'+dataname+'/HEAT_boxplot.png')
	plt.close()

if Box_LR == True:
	color_list = ['cornflowerblue','limegreen','palevioletred','orange']

	for iter_num in range(5):
		ratio_list = ['All gene','Cell-cell contact','Secreted signaling','ECM']
		df_box = pd.DataFrame(columns=ratio_list)
		for noise_ratio in ratio_list:
			AUC_list = []
			for iter_num in range(5):
				if noise_ratio == 'All gene':
					df_auc = pd.read_csv('results/AUC/'+'StereoSeq_digae_PCA300'+'/'+dataname+'_missingratio0.1_iter'+str(iter_num)+'.csv')
					AUC_list.append(df_auc['auc'].values[19]) #[59]
				elif noise_ratio == 'Cell-cell contact':
					df_auc = pd.read_csv('results/AUC/'+'StereoSeq_cellcellLR'+'/'+dataname+'_missingratio0.1_iter'+str(iter_num)+'.csv')
					AUC_list.append(df_auc['auc'].values[19]) #[59]
				elif noise_ratio == 'Secreted signaling':
					df_auc = pd.read_csv('results/AUC/'+'StereoSeq_LR_60epoch/StereoSeq_Secreted'+'/'+dataname+'_missingratio0.1_iter'+str(iter_num)+'.csv')
					AUC_list.append(df_auc['auc'].values[19]) #[-1]
				elif noise_ratio == 'ECM':
					df_auc = pd.read_csv('results/AUC/'+'StereoSeq_LR_60epoch/StereoSeq_ECM/'+'/'+dataname+'_missingratio0.1_iter'+str(iter_num)+'.csv')
					AUC_list.append(df_auc['auc'].values[19]) #[-1]

			df_box[noise_ratio] = AUC_list
			# ax.plot(range(1,len(epochs_list)+1), AUC_list, color='orange', linestyle='-', linewidth=6)

	print(df_box)
	sns.boxplot(data=df_box)
	# plt.show()
	font1 = {'family':'Arial','weight':'normal','size':12}
	# plt.xlabel('missing edges ratio',font1)
	plt.ylabel('AUROC',font1)
	plt.savefig('plot/AUC/'+dataname+'/LR_boxplot_PCA300.png')
	plt.close()


if line_LR == True:
	color_list = ['cornflowerblue','limegreen','palevioletred','orange']
	# ratio_list = ['HBC_cellcellLR','HBC_Secreted','HBC_ECM']
	ratio_list = ['StereoSeq_cellcellLR'] #['StereoSeq_cellcellLR','StereoSeq_Secreted','StereoSeq_ECM']

	df_box = pd.DataFrame(columns=ratio_list)
	figure, ax = plt.subplots(figsize= (13,9), dpi=100)
	for i in range(len(ratio_list)):
		ratio = ratio_list[i]
		iter_num = 4
		df_auc = pd.read_csv('results/AUC/'+ratio+'/'+dataname+'_missingratio0.1_iter'+str(iter_num)+'.csv')
		AUC_list = df_auc['auc'].values.tolist()
		epochs_list = df_auc['epoch'].values.tolist()
		yvals = polyfit_metrics(AUC_list)
		ax.plot(range(1,len(epochs_list)+1), yvals, color=color_list[i], linestyle='-', linewidth=6,label=ratio)

	df_auc = pd.read_csv('results/AUC/'+'StereoSeq_digae_PCA300'+'/'+dataname+'_missingratio0.1_iter'+str(iter_num)+'.csv') #missingratio, testratio
	AUC_list = df_auc['auc'].values.tolist()
	epochs_list = df_auc['epoch'].values.tolist()
	yvals = polyfit_metrics(AUC_list)
	ax.plot(range(1,len(epochs_list)+1), yvals, color='orange', linestyle='-', linewidth=6,label='All gene')

	# plt.legend(handles=[],labels=['1','2','3','4'])

	plt.tick_params(labelsize=20)
	labels = ax.get_xticklabels() + ax.get_yticklabels()
	[label.set_fontname('Arial') for label in labels]

	font1 = {'family':'Arial','weight':'normal','size':25,}
	plt.xlabel("epoch", font1)
	plt.ylabel("AUROC", font1)

	# font2 = {'family':'Arial','size':25,}
	# leg = plt.legend(["AUPRC"], bbox_to_anchor=(1.02, 0), loc='lower right', borderaxespad=0, prop=font2)
	# leg.get_frame().set_linewidth(0.0)

	font2 = {'family':'Arial','size':15,}
	plt.legend(frameon=False,prop=font2)
	# plt.show()
	file_name = 'plot/AUC/'+dataname+'/AUPRC_LR_PCA300.png'
	plt.savefig(file_name)
	plt.close()


if line_dropout_gene == True:
	# color_list = ['cornflowerblue','limegreen','palevioletred']
	color_list = ['cornflowerblue','limegreen','palevioletred','green','blue','yellow','grey','grey','grey','orange']
	if dataname == 'HDST_ob':
		# ratio_list = ['0.1','0.2','0.3']
		ratio_list = ['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9']
	if dataname == 'HDST_cancer':
		# ratio_list = ['0.1','0.2','0.3']
		ratio_list = ['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9']

	df_box = pd.DataFrame(columns=ratio_list)
	figure, ax = plt.subplots(figsize= (13,9), dpi=100)
	for i in range(len(ratio_list)):
		iter_num = 1
		df_auc = pd.read_csv('results/AUC/'+dataname+'/'+dataname+'_dropout_gene_'+ratio_list[i]+'_iter'+str(iter_num)+'.csv')
		AUC_list = df_auc['auc'].values.tolist()
		epochs_list = df_auc['epoch'].values.tolist()
		yvals = polyfit_metrics(AUC_list)
		ax.plot(range(1,len(epochs_list)+1), yvals, color=color_list[i], linestyle='-', linewidth=6,label=ratio_list[i])

	df_auc = pd.read_csv('results/AUC/'+dataname+'/'+dataname+'_testratio0.1_iter'+str(iter_num)+'.csv') #missingratio, testratio
	AUC_list = df_auc['auc'].values.tolist()
	epochs_list = df_auc['epoch'].values.tolist()
	yvals = polyfit_metrics(AUC_list)
	ax.plot(range(1,len(epochs_list)+1), yvals, color='orange', linestyle='-', linewidth=6,label='normal')

	# plt.legend(handles=[],labels=['1','2','3','4'])

	plt.tick_params(labelsize=20)
	labels = ax.get_xticklabels() + ax.get_yticklabels()
	[label.set_fontname('Arial') for label in labels]

	font1 = {'family':'Arial','weight':'normal','size':25,}
	plt.xlabel("epoch", font1)
	plt.ylabel("AUROC", font1)

	# font2 = {'family':'Arial','size':25,}
	# leg = plt.legend(["AUPRC"], bbox_to_anchor=(1.02, 0), loc='lower right', borderaxespad=0, prop=font2)
	# leg.get_frame().set_linewidth(0.0)

	font2 = {'family':'Arial','size':15,}
	plt.legend(frameon=False,prop=font2)
	# plt.show()
	file_name = 'plot/AUC/'+dataname+'/AUPRC_dropoutgene.png'
	plt.savefig(file_name)
	plt.close()


if line_dropout_value == True:
	color_list = ['cornflowerblue','limegreen','palevioletred','orange']
	color_list = ['cornflowerblue','limegreen','palevioletred','green','blue','yellow','grey','grey','grey','orange']
	if dataname == 'HDST_ob':
		# ratio_list = ['0.1','0.2','0.3']
		ratio_list = ['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9']
	if dataname == 'HDST_cancer':
		# ratio_list = ['0.1','0.2','0.3']
		ratio_list = ['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9']

	df_box = pd.DataFrame(columns=ratio_list)
	figure, ax = plt.subplots(figsize= (13,9), dpi=100)
	for i in range(len(ratio_list)):
		iter_num = 1
		df_auc = pd.read_csv('results/AUC/'+dataname+'/'+dataname+'_dropout_value_'+ratio_list[i]+'_iter'+str(iter_num)+'.csv')
		AUC_list = df_auc['auc'].values.tolist()
		epochs_list = df_auc['epoch'].values.tolist()
		yvals = polyfit_metrics(AUC_list)
		ax.plot(range(1,len(epochs_list)+1), yvals, color=color_list[i], linestyle='-', linewidth=6,label=ratio_list[i])

	df_auc = pd.read_csv('results/AUC/'+dataname+'/'+dataname+'_testratio0.1_iter'+str(iter_num)+'.csv') #missingratio, testratio
	AUC_list = df_auc['auc'].values.tolist()
	epochs_list = df_auc['epoch'].values.tolist()
	yvals = polyfit_metrics(AUC_list)
	ax.plot(range(1,len(epochs_list)+1), yvals, color='orange', linestyle='-', linewidth=6,label='normal')

	# plt.legend(handles=[],labels=['1','2','3','4'])

	plt.tick_params(labelsize=20)
	labels = ax.get_xticklabels() + ax.get_yticklabels()
	[label.set_fontname('Arial') for label in labels]

	font1 = {'family':'Arial','weight':'normal','size':25,}
	plt.xlabel("epoch", font1)
	plt.ylabel("AUROC", font1)

	# font2 = {'family':'Arial','size':25,}
	# leg = plt.legend(["AUPRC"], bbox_to_anchor=(1.02, 0), loc='lower right', borderaxespad=0, prop=font2)
	# leg.get_frame().set_linewidth(0.0)

	font2 = {'family':'Arial','size':15,}
	plt.legend(frameon=False,prop=font2)
	# plt.show()
	file_name = 'plot/AUC/'+dataname+'/AUPRC_dropoutvalue.png'
	plt.savefig(file_name)
	plt.close()


##### compare to DeepLinc
if line_compare == True:
	for iter_num in range(5):
		df_DiGAE = pd.read_csv('results/AUC/'+dataname+'/'+dataname+'_testratio0.1_iter'+str(iter_num)+'.csv')
		epochs_list = df_DiGAE['epoch'].values.tolist()
		DiGAE_auc = df_DiGAE['auc'].values.tolist()

		df_DeepLink = pd.read_csv('../DeepLinc-main/results/AUC/'+dataname+'/'+dataname+'_testratio0.1_iter'+str(iter_num)+'.csv')
		DeepLink_auc = df_DeepLink['auc'].values.tolist()

		yvals_DiGAE = polyfit_metrics(DiGAE_auc)
		yvals_DeepLink = polyfit_metrics(DeepLink_auc)

		figure, ax = plt.subplots(figsize= (13,9), dpi=100)

		ax.plot(range(1,len(epochs_list)+1), yvals_DiGAE, color='orange', linestyle='-', linewidth=6)
		ax.plot(range(1,len(epochs_list)+1), yvals_DeepLink, color='cornflowerblue', linestyle='-', linewidth=6)
		# ax.plot(range(1,len(metrics_list)+1), yvals_3, color='limegreen', linestyle='-', linewidth=6)
		# ax.plot(range(1,len(metrics_list)+1), yvals_4, color='palevioletred', linestyle='-', linewidth=6)

		plt.tick_params(labelsize=20)
		labels = ax.get_xticklabels() + ax.get_yticklabels()
		[label.set_fontname('Arial') for label in labels]

		font1 = {'family':'Arial','weight':'normal','size':30,}
		plt.xlabel("epoch", font1)
		plt.ylabel("AUROC", font1)

		font2 = {'family':'Arial','size':15,}
		leg = plt.legend(["DiGAE","DeepLink"],  loc='upper left', prop=font2) #'lower right' bbox_to_anchor=(1.02, 0), borderaxespad=0,
		leg.get_frame().set_linewidth(0.0)

		# plt.show()
		# plt.savefig('{}.tif'.format("AUPRC"))
		file_name = 'plot/AUC/'+dataname+'/AUPRC_testratio0.1_iter_'+str(iter_num)+'.png'
		plt.savefig(file_name)
		plt.close()

if line_noise == True:
	# color_list = ['cornflowerblue','limegreen','palevioletred','orange']
	# color_list = ['palevioletred','limegreen','cornflowerblue','orange']
	color_list = ['palevioletred','limegreen','cornflowerblue','green','orange']
	if dataname == 'HDST_ob':
		# ratio_list = ['2.5','2.7','3.0']
		ratio_list = ['2.0','2.5','2.7','3.0']
	if dataname == 'HDST_cancer':
		ratio_list = ['2.0','2.6','3.0']
	if dataname == 'seqFISH':
		ratio_list = ['1.0','1.7','2.0']
	if dataname == 'MERFISH':
		ratio_list = ['1.0','1.8','2.0']

	df_box = pd.DataFrame(columns=ratio_list)
	figure, ax = plt.subplots(figsize= (13,9), dpi=100)
	for i in range(len(ratio_list)):
		iter_num = 0
		df_auc = pd.read_csv('results/AUC/'+dataname+'/'+dataname+'_noise_'+ratio_list[i]+'_iter'+str(iter_num)+'.csv')
		AUC_list = df_auc['auc'].values.tolist()
		epochs_list = df_auc['epoch'].values.tolist()
		yvals = polyfit_metrics(AUC_list)
		ax.plot(range(1,len(epochs_list)+1), yvals, color=color_list[i], linestyle='-', linewidth=6,label=ratio_list[i])

	df_auc = pd.read_csv('results/AUC/'+dataname+'/'+dataname+'_testratio0.1_iter'+str(iter_num)+'.csv')
	AUC_list = df_auc['auc'].values.tolist()
	epochs_list = df_auc['epoch'].values.tolist()
	yvals = polyfit_metrics(AUC_list)
	ax.plot(range(1,len(epochs_list)+1), yvals, color='orange', linestyle='-', linewidth=6,label='normal')

	# plt.legend(handles=[],labels=['1','2','3','4'])

	plt.tick_params(labelsize=20)
	labels = ax.get_xticklabels() + ax.get_yticklabels()
	[label.set_fontname('Arial') for label in labels]

	font1 = {'family':'Arial','weight':'normal','size':25,}
	plt.xlabel("epoch", font1)
	plt.ylabel("AUROC", font1)

	# font2 = {'family':'Arial','size':25,}
	# leg = plt.legend(["AUPRC"], bbox_to_anchor=(1.02, 0), loc='lower right', borderaxespad=0, prop=font2)
	# leg.get_frame().set_linewidth(0.0)

	font2 = {'family':'Arial','size':15,}
	plt.legend(frameon=False,prop=font2)
	# plt.show()
	file_name = 'plot/AUC/'+dataname+'/AUPRC_noise.png'
	plt.savefig(file_name)
	plt.close()

if Box_missing == True:
	ratio_list = ['0.1','0.2','0.4','0.5','0.6','0.7','0.8','0.9']
	df_box = pd.DataFrame(columns=ratio_list)
	for noise_ratio in ratio_list:
		AUC_list = []
		for iter_num in range(5):
			df_auc = pd.read_csv('results/AUC/'+dataname+'/'+dataname+'_testratio'+noise_ratio+'_iter'+str(iter_num)+'.csv')
			AUC_list.append(df_auc['auc'].values[-1])
		df_box[noise_ratio] = AUC_list
		# ax.plot(range(1,len(epochs_list)+1), AUC_list, color='orange', linestyle='-', linewidth=6)

	print(df_box)
	sns.boxplot(data=df_box)
	# plt.show()
	font1 = {'family':'Arial','weight':'normal','size':12}
	plt.xlabel('missing edges ratio',font1)
	plt.ylabel('AUROC',font1)
	plt.savefig('plot/AUC/'+dataname+'/missingratio_boxplot.png')
	plt.close()


if Box_noise == True:
	ratio_list = ['2.5','2.7','3.0']
	df_box = pd.DataFrame(columns=ratio_list)
	for noise_ratio in ratio_list:
		AUC_list = []
		for iter_num in range(5):
			df_auc = pd.read_csv('results/AUC/'+dataname+'/'+dataname+'_noise_'+noise_ratio+'_iter'+str(iter_num)+'.csv')
			AUC_list.append(df_auc['auc'].values[-1])
		df_box[noise_ratio] = AUC_list
		ax.plot(range(1,len(epochs_list)+1), AUC_list, color='orange', linestyle='-', linewidth=6)

	AUC_list = []
	for iter_num in range(5):
		df_auc = pd.read_csv('results/AUC/'+dataname+'/'+dataname+'_testratio0.1_iter'+str(iter_num)+'.csv')
		AUC_list.append(df_auc['auc'].values[-1])
	df_box['normal'] = AUC_list	

	print(df_box)
	# tips = 
	sns.boxplot(data=df_box)
	plt.savefig('plot/AUC/'+dataname+'/noise_boxplot.png')
	plt.close()

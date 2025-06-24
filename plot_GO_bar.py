import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 

dataname = 'HDST_cancer' # 'HDST_cancer', 'HDST_ob'
df_GO = pd.read_csv('results/GO/GO_'+dataname+'.csv')
print(df_GO)

data = df_GO['logP'].values.tolist()
print(data)
category_colors = plt.get_cmap('Paired')(np.linspace(0, 0.8, len(data)))
category_names = df_GO['GO cellular component complete'].values.tolist()
plt.barh(range(len(data)), data[::-1],tick_label=category_names[::-1],color=category_colors[::-1]) #color=['r', 'g', 'b']
# plt.show()

plt.yticks(rotation=30,fontsize=12)
plt.xticks(np.arange(0,max(data)+1,5),fontsize=12)
plt.xlabel("-log10(p.adjust)",fontsize=12)
plt.tight_layout()

file_name = 'plot/GO/GO_'+dataname+'.png'
plt.savefig(file_name,dpi = 300)


#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_row',15)

# %% Pré-processamento
file_path = r'..\dados\teste_indicium_precificacao.csv'
dados = pd.read_csv(
    file_path, 
    dtype= {
    'id': str,
    'host_id': str
} )
dados.head()

#%%

variaveis_num = dados.select_dtypes(include=np.number).columns.to_list()
variaveis_cat = dados.select_dtypes(exclude= np.number).columns.to_list()

dados[variaveis_num].isna().sum()
# %% Os Nas são basicamente de hostels que nunca tiveram review 

dados.loc[dados['numero_de_reviews'] == 0, 'reviews_por_mes'] = 0
# %%
dados[variaveis_num].isna().sum()
#%%

dados[variaveis_cat].isna().sum()
# %%
dados['nome'].nunique()
# %% O nome que aparece muitas vezes s
dados['host_name'].nunique()
#%%
dados['host_id'].nunique()

# %%
dados['reviews'] =  dados['ultima_review'].notna().astype(int)
dados['reviews'].value_counts()
dados.drop('ultima_review', axis=1, inplace= True)
#%%
variaveis_cat.append('reviews')
variaveis_cat.remove('ultima_review')
# %%
dados[variaveis_cat].isna().sum()
# %%
variaveis_removiveis = ['id', 'nome', 'host_id', 'host_name']

# %%

dados.head()
#%%
dados_limpos = dados.drop(variaveis_removiveis, axis=1).copy()
dados_limpos.info()
# %%
dados_limpos[variaveis_num].describe().T

dados_limpos.to_csv('../dados/dados_limpos.csv', index=False)

# %%

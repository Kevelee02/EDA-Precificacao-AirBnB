#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_row',15)
#%%
file_path = r'..\dados\teste_indicium_precificacao.csv'
dados = pd.read_csv(
    file_path, 
    dtype= {
    'id': str,
    'host_id': str
} )
dados.head()
# %%
dados['bairro_group'].value_counts()
# %%
fig = sns.histplot(dados, x = 'bairro_group')
fig.set_title('Distruição da quantidade de Airbnb por grupos de bairro')
fig.set_xlabel('Grupos de Bairro')
fig.set_ylabel( 'Contagem')
plt.show()
# %%
cores = ['blue','orange','green','red']
room_bairro = pd.crosstab(dados['bairro_group'],dados['room_type'])
room_bairro
#%%
dados['room_type'].value_counts()
# %%
preco_medio = dados.groupby(['bairro_group', 'room_type'])['price'].mean().reset_index()
fig = sns.barplot(
    data= preco_medio,
    x='room_type',
    y='price',
    hue = 'bairro_group'
)
fig.set_title('Média dos valores de Airbnb por bairro')
plt.show()
# %%
dados['price_min'] = dados['price']*dados['minimo_noites']
dados2 = dados.groupby('bairro_group')['price_min'].mean().reset_index()
fig = sns.barplot(data = dados2,
                  x = 'bairro_group', 
                  y = 'price_min' 
                  #barmode = 'group',
             )
fig.set_title('Preço médio mínimo por grupo de bairros')
plt.show()
# %%
bairros_caros = dados.groupby('bairro')['price'].mean().sort_values(ascending = False).head(10)
#renomeando as colunas
bairros_caros.index.name = 'Bairro'
bairros_caros.name = 'Preço Médio'
bairros_caros = pd.DataFrame(bairros_caros)
plt.figure(figsize= (20, 15), dpi=300)
fig = sns.barplot(data = bairros_caros,
                   x = bairros_caros.index, 
                   y = bairros_caros['Preço Médio'])
fig.set_title('Top 10 Bairros mais caros')
plt.show()

# %%
variaveis_num = dados.select_dtypes(include=np.number).columns.to_list()
variaveis_cat = dados.select_dtypes(exclude= np.number).columns.to_list()
#%%
corr = dados[variaveis_num].corr()
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(corr, annot = True, cmap ='coolwarm')
plt.show()
# %%
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data = dados,
    x = 'latitude',
    y = 'longitude',
    alpha = 0.2,
    s = 10,
    edgecolor = None
)
plt.title('Distribuição Geométrica de Pontos em NY')
plt.show()
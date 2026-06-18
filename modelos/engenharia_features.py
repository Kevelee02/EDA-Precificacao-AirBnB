#%%
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
#%%

dados = pd.read_csv('../dados/dados_limpos.csv')
dados.head()
# %%

dados['bairro'].nunique()
#%% Features Geospacial

def criar_variaveis(df):
    
    df['room_type_group'] = df['bairro_group'] + '_' + df['room_type']
    
    
    condicoes = [
        (df['minimo_noites'] <= 3),
        (df['minimo_noites'] > 3) & (df['minimo_noites'] <= 14),
        (df['minimo_noites'] > 14)
    ]
    escolhas = ['curta_duracao', 'media_duracao', 'longa_duracao']
    df['perfil_estadia'] = np.select(condicoes, escolhas, default='outro')
    

    df['dias_ocupados_estimados'] = 365 - df['disponibilidade_365']
    
    return df
#%% Features Geospacial
def calcular_distancia_haversine(lat1, lon1, lat2, lon2):
    """
    Calcula a distância em quilómetros entre dois pontos geográficos
    utilizando a fórmula de Haversine.
    """
    # Converter graus decimais para radianos
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Diferenças das coordenadas
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Fórmula de Haversine
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Raio médio da Terra em quilómetros
    raio_terra_km = 6371
    return raio_terra_km * c

#%% Features Geospacial
def aplicar_engenharia_geoespacial(df, n_clusters=8):
   
    df_geo = df.copy()
    
    # 1. Definição dos Pontos de Interesse (POIs) em Nova Iorque
    pontos_interesse = {
        'distancia_times_square': (40.7580, -73.9855),
        'distancia_aeroporto_jfk': (40.6413, -73.7781),
        'distancia_central_park': (40.7851, -73.9683)
    }
    
    # Cálculo iterativo das distâncias lineares para cada registo
    for nome_feature, (poi_lat, poi_lon) in pontos_interesse.items():
        df_geo[nome_feature] = calcular_distancia_haversine(
            df_geo['latitude'], 
            df_geo['longitude'], 
            poi_lat, 
            poi_lon
        )
        
    # 2. Criação de Micro-Bairros via Clusterização Espacial
    # Isolar apenas as colunas de localização para o treino do algoritmo
    coordenadas = df_geo[['latitude', 'longitude']]
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_geo['micro_bairro_cluster'] = kmeans.fit_predict(coordenadas)
    
    # Conversão do cluster para string (categórica) para evitar que o modelo 
    # o interprete erroneamente como uma variável numérica contínua
    df_geo['micro_bairro_cluster'] = df_geo['micro_bairro_cluster'].astype(str)
    
    return df_geo
# %%

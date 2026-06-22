#%%
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from feature_engine.encoding import OneHotEncoder

def criar_variaveis(df):

    df = df.copy()
    
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

class EngenhariaFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        X = X.copy()

        self.kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
        self.kmeans.fit(X[['latitude', 'longitude']])

        X_transformado = self._aplicar_transformacoes(X)

        colunas_categoricas = X_transformado.select_dtypes(exclude=np.number).columns.tolist()
        self.onehot = OneHotEncoder(variables=colunas_categoricas)
        self.onehot.fit(X_transformado)

        return self

    def _aplicar_transformacoes(self, X):
        X = X.copy()
        X = self._engenharia_geoespacial(X)
        X = criar_variaveis(X)
        return X

    def _engenharia_geoespacial(self, X):
        df_geo = X.copy()

        pontos_interesse = {
            'distancia_times_square': (40.7580, -73.9855),
            'distancia_aeroporto_jfk': (40.6413, -73.7781),
            'distancia_central_park': (40.7851, -73.9683)
        }

        for nome_feature, (poi_lat, poi_lon) in pontos_interesse.items():
            df_geo[nome_feature] = calcular_distancia_haversine(
                lat1=df_geo['latitude'],
                lon1=df_geo['longitude'],
                lat2=poi_lat,
                lon2=poi_lon
            )

        df_geo['micro_bairro_cluster'] = self.kmeans.predict(
            df_geo[['latitude', 'longitude']]
        ).astype(str)

        # Dropa após todos os cálculos
        df_geo = df_geo.drop(columns=['latitude', 'longitude'])

        return df_geo

    def transform(self, X):
        X = X.copy()
        X = self._aplicar_transformacoes(X)
        X = self.onehot.transform(X)
        return X
# %%

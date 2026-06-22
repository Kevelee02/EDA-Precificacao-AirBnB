
from src.engenharia_features import criar_variaveis, calcular_distancia_haversine, aplicar_engenharia_geoespacial
import pandas as pd


def test_criar_variaveis():
    df = pd.DataFrame({
        'bairro_group': ['Manhattan', 'Manhattan', 'Brooklyn', 'Manhattan'],
        'bairro': ['Midtown', 'Harlem', 'Clinton Hill', 'East Harlem'],
        'latitude': [40.75362, 40.80902, 40.68514, 40.79851],
        'longitude': [-73.98377, -73.94190, -73.95976, -73.94399],
        'room_type': ['Entire home/apt', 'Private room', 'Entire home/apt', 'Entire home/apt'],
        'price': [225, 150, 89, 80],
        'minimo_noites': [1, 3, 10, 15],
        'disponibilidade_365': [355, 365, 194, 0]
    })
    df_esperado = pd.DataFrame({
        'room_type_group': ['Manhattan_Entire home/apt', 
        'Manhattan_Private room', 
        'Brooklyn_Entire home/apt', 
        'Manhattan_Entire home/apt' ],


        'perfil_estadia':[ 'curta_duracao', 
        'curta_duracao', 
        'media_duracao', 
        'longa_duracao'],

        'dias_ocupados_estimados' : [10,0,171,365]
    })


    resultado = criar_variaveis(df)

    assert resultado["room_type_group"].tolist() == [
    "Manhattan_Entire home/apt",
    "Manhattan_Private room",
    "Brooklyn_Entire home/apt",
    "Manhattan_Entire home/apt"
    ]

    assert resultado["perfil_estadia"].tolist() == [
    "curta_duracao",
    "curta_duracao",
    "media_duracao",
    "longa_duracao"
    ]

    assert resultado["dias_ocupados_estimados"].tolist() == [
    10,
    0,
    171,
    365
    ]
def test_distancia_haversine():
    import pytest

    dist_zero = calcular_distancia_haversine(40.75, -73.98, 40.75, -73.98)
    assert dist_zero == 0.0, f"A distância para o mesmo ponto deveria ser 0, mas foi {dist_zero}"
    
    # Teste 2: Times Square até Central Park
    # Coordenadas: Times Square (40.7580, -73.9855), Central Park (40.7851, -73.9683)
    # Distância real esperada: ~ 3.34 km
    dist_calculada = calcular_distancia_haversine(40.7580, -73.9855, 40.7851, -73.9683)
    dist_esperada = 3.34
    

    assert dist_calculada == pytest.approx(dist_esperada, abs=0.01)

def test_aplicar_engenharia_geoespacial():
    import pandas as pd

def test_aplicar_engenharia_geoespacial_cria_colunas():

    df = pd.DataFrame({
        "latitude": [40.7580, 40.7851, 40.6413],
        "longitude": [-73.9855, -73.9683, -73.7781]
    })

    resultado = aplicar_engenharia_geoespacial(df, n_clusters=2)

    colunas_esperadas = [
        "distancia_times_square",
        "distancia_aeroporto_jfk",
        "distancia_central_park",
        "micro_bairro_cluster"
    ]

    for coluna in colunas_esperadas:
        assert coluna in resultado.columns
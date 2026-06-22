#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from engenharia_features import EngenhariaFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np


#%%
dados = pd.read_csv('../dados/dados_limpos.csv')
dados.head()
# %%

features = [
    "bairro_group",
    "bairro",
    "latitude",
    "longitude",
    "room_type"	,
    "minimo_noites"	,
    "numero_de_reviews",
    "reviews_por_mes",
    "calculado_host_listings_count",
    "disponibilidade_365",
    "reviews"
]
target = 'price'

X = dados[features]
y = dados[target]


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size= 0.3, random_state= 42)
X_test, X_val2, y_test, y_val2 = train_test_split(X_val, y_val, test_size=0.5, random_state=42)


# %% Mesmo modelo do projeto anterior

rf = RandomForestRegressor(
    n_estimators= 200, #definindo o n de arvores em 100
    max_depth = 15, # deixando as avores mais simples pra tentar evitar overfiting
    min_samples_split = 5,
    min_samples_leaf= 5,
    random_state=42
)

# %%
pipeline = Pipeline([
    ("engenharia_features", EngenhariaFeatures()),
    ("modelo", rf)
])

pipeline.fit(X_train, y_train)
# %%
y_pred_test = pipeline.predict(X_test)
#%%
r_2 = r2_score(y_test, y_pred_test)
msr = mean_absolute_error(y_test, y_pred_test)
print(f' Erro médio absoluto: {msr}')
print(f'R Quadrado: {r_2}')
# %% Aplicando log no target
y_train_log = np.log1p(y_train)

pipeline.fit(X_train, y_train_log)

y_pred = np.expm1(pipeline.predict(X_test))

print(f"\nMAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R²:  {r2_score(y_test, y_pred):.2f}")
# %% Busca de Hyperparametros
y_val2_log = np.log1p(y_val2)

paramns = {
    'modelo__n_estimators' : [100, 200, 300, 500],
    'modelo__max_depth' : [None, 5, 10, 15], 
    'modelo__min_samples_split' : [2 , 5, 10],
    'modelo__min_samples_leaf' : [5, 10],
}
from sklearn.model_selection import GridSearchCV, cross_val_score

gsv = GridSearchCV(pipeline, 
                   paramns,
                   cv = 3, 
                   scoring= 'r2',
                   n_jobs=-1)
gsv.fit(X_val2, y_val2)
gsv.best_params_
#%%
gsv.best_score_

# %%

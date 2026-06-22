# Precificação de Imóveis no Airbnb

## Sobre o Projeto

Este projeto foi desenvolvido como a releitura de um projeto antigo,  e tem como objetivo analisar fatores que influenciam o preço de acomodações anunciadas no Airbnb, além de construir um modelo de Machine Learning capaz de estimar o valor de uma propriedade com base em suas características.

O trabalho envolve desde a análise exploratória dos dados até a construção de um pipeline de engenharia de atributos e treinamento de modelos preditivos.

---

## Objetivos

* Explorar os fatores que impactam o preço dos imóveis.
* Identificar padrões geográficos e comportamentais dos anúncios.
* Desenvolver novas variáveis com potencial explicativo.
* Construir um modelo de regressão para previsão de preços.
* Avaliar a capacidade preditiva do modelo utilizando validação cruzada.

---

## Tecnologias Utilizadas

### Linguagem

* Python

### Bibliotecas

* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Plotly
* Feature Engine
* Pytest

---

## Estrutura do Projeto

```text
├── dados/
├── notebooks/
├── src/
│   ├── engenharia_features.py
│   ├── modelagem.py
│   └── pre_processamento.py
├── testes/
├── README.md
```

---

## Etapas do Projeto

### 1. Análise Exploratória de Dados (EDA)

Durante a etapa exploratória foram analisados:

* Distribuição dos preços;
* Distribuição dos tipos de acomodação;
* Relação entre localização e preço;
* Impacto do número de avaliações;
* Padrões de disponibilidade dos imóveis;
* Presença de outliers.

---

### 2. Engenharia de Features

Foram criadas variáveis adicionais para aumentar o poder explicativo do modelo.

#### Variáveis Geográficas

* Distância até Times Square;
* Distância até Central Park;
* Distância até o Aeroporto JFK.

As distâncias foram calculadas utilizando a fórmula de Haversine.

#### Variáveis de Perfil

* Perfil da estadia;
* Estimativa de ocupação;
* Agrupamentos geográficos por região.

Essas variáveis buscam representar aspectos do comportamento dos hóspedes e da localização dos imóveis.

---

### 3. Modelagem

Foi desenvolvido um pipeline contendo:

* Engenharia de atributos;
* Pré-processamento;
* Modelo de regressão Random Forest.

O modelo foi ajustado por meio de busca de hiperparâmetros utilizando GridSearchCV.

---

### 4. Validação

A avaliação foi realizada utilizando:

* Train/Test Split;
* Validação Cruzada (Cross Validation);
* Coeficiente de Determinação (R²);
* Erro Absoluto Médio (MAE);
* Raiz do Erro Quadrático Médio (RMSE).

A validação cruzada foi utilizada para verificar a estabilidade do modelo e reduzir a dependência de uma única divisão dos dados.

---

## Testes

O projeto possui testes automatizados para validar:

* Criação das features;
* Cálculo das distâncias geográficas;
* Consistência das transformações aplicadas ao dataset.

---

## Principais Conclusões

Os resultados mostram que fatores relacionados à localização e ao tipo de acomodação possuem forte influência sobre o preço dos imóveis.

Apesar da utilização de engenharia de atributos e otimização de hiperparâmetros, o desempenho do modelo apresenta ganhos limitados, indicando que parte significativa da variabilidade dos preços depende de informações não presentes no conjunto de dados, como qualidade do imóvel, avaliações dos hóspedes, características visuais e fatores sazonais.

  pytest


---

# IMPORTS
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


# FUNCAO 1
def carregar_dados():
    iris = load_iris()
    X = iris.data
    y = iris.target
    print("Dataset carregado com sucesso!")
    print(f"X.shape: {X.shape}")
    print(f"y.shape: {y.shape}")
    print(f"Classes: {iris.target_names}")
    return X, y


# FUNCAO 2
def dividir_dados(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42
    )
    print(f"Treino: {X_train.shape[0]} flores")
    print(f"Teste:  {X_test.shape[0]} flores")
    return X_train, X_test, y_train, y_test


# FUNCAO 3
def treinar_knn(X_train, y_train, lista_k=[1, 3, 5, 7, 9]):
    modelos = {}
    for k in lista_k:
        modelo = KNeighborsClassifier(n_neighbors=k)
        modelo.fit(X_train, y_train)
        modelos[k] = modelo
        print(f"Modelo KNN com k={k} treinado!")
    return modelos


# FUNCAO 4
def avaliar_modelos(modelos, X_test, y_test):
    resultados = {}
    for k, modelo in modelos.items():
        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        resultados[k] = acc
        print(f"k={k} -> acuracia: {acc * 100:.2f}%")
    return resultados


# FUNCAO 5
def exibir_resultados(resultados):
    tabela = pd.DataFrame({
        'k (vizinhos)': list(resultados.keys()),
        'Acuracia (%)': [f"{acc * 100:.2f}%" for acc in resultados.values()]
    })
    print("\nTabela Comparativa")
    print(tabela.to_string(index=False))
    melhor_k = max(resultados, key=resultados.get)
    print(f"\nMelhor k: {melhor_k} -> {resultados[melhor_k] * 100:.2f}%")


# CHAMADAS
X, y = carregar_dados()
X_train, X_test, y_train, y_test = dividir_dados(X, y)
modelos = treinar_knn(X_train, y_train)
resultados = avaliar_modelos(modelos, X_test, y_test)
exibir_resultados(resultados)
# 📄 Modelos de Classificação de Qualidade do Ar (AQI)

Este projeto implementa e compara **5 modelos de Machine Learning (ML)** para classificar a Qualidade do Ar (`AQI Category`), conforme exigido na errata do exercício.

---

## 🚀 1. Configuração e Instalação

Siga os passos para configurar o ambiente no VS Code.

### 1.1. Pré-requisitos

* **Python 3.12.15**
* **Visual Studio Code (VS Code)**

### 1.2. Estrutura do Projeto

Certifique-se de que a sua pasta contém o arquivo de dados e todos os scripts de modelo:

/Projeto_ML_AQI/ ├── AQI and Lat Long of Countries.csv

├── knn_classificacao.py ├── naive_bayes_classificacao.py ├── arvore_decisao_classificacao.py ├── regressao_logistica_classificacao.py ├── redes_neurais_classificacao.py └── README.md

### 1.3. Instalação de Dependências

Abra o Terminal no VS Code (**Terminal > New Terminal** ou `Ctrl + '`) e execute o seguinte comando para instalar todas as bibliotecas necessárias (`pandas` e `scikit-learn`):

bash
pip install pandas scikit-learn numpy

▶️ 2. Execução dos ModelosCada script executa o pré-processamento, treinamento e avaliação de um modelo específico. Use o Terminal do VS Code para rodar cada um:
Modelo de Classificação,Comando de Execução
K-Nearest Neighbors (KNN),python knn_classificacao.py
Naive Bayes,python naive_bayes_classificacao.py
Árvore de Decisão,python arvore_decisao_classificacao.py
Regressão Logística,python regressao_logistica_classificacao.py
Redes Neurais (MLPClassifier),python redes_neurais_classificacao.py

📊 3. Análise de Resultados
Variáveis Utilizadas
Dataset: AQI and Lat Long of Countries.csv

Recursos (X): Colunas numéricas como 'AQI Value', 'CO AQI Value', 'lat', etc.

Alvo (y): 'AQI Category' (Categórica).

Desempenho dos Modelos (Acurácia)




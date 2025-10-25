import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB # <--- NOVO MODELO
from sklearn.metrics import accuracy_score, classification_report
import warnings

# Suprime avisos de convergência (apesar de não serem comuns no Naive Bayes, mantemos para consistência)
warnings.filterwarnings('ignore')

# --- 1. CARREGAMENTO DOS DADOS ---
file_name = "AQI and Lat Long of Countries.csv"
try:
    df = pd.read_csv(file_name)
    print("Dados carregados com sucesso.")
except FileNotFoundError:
    print(f"Erro: Arquivo '{file_name}' não encontrado.")
    exit()

# --- 2. DEFINIÇÃO DE RECURSOS (X) E ALVO (Y) ---
features = [
    'AQI Value', 
    'CO AQI Value', 
    'Ozone AQI Value', 
    'NO2 AQI Value', 
    'PM2.5 AQI Value',
    'lat', 
    'lng'
]
target_column = 'AQI Category'

X = df[features]
y = df[target_column]

# --- 3. PRÉ-PROCESSAMENTO: CODIFICAÇÃO E ESCALONAMENTO ---

# Codificação da Variável Alvo (y)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
target_names = le.classes_ 

# Divisão dos Dados em Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# Padronização/Escalonamento dos Recursos 
# IMPORTANTE: O Naive Bayes, em teoria, não precisa de escalonamento, 
# mas o utilizamos aqui para manter a consistência com o pré-processamento
# de todos os modelos.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nPré-processamento concluído.")
print(f"Dados de Treino: {X_train_scaled.shape}")
print(f"Dados de Teste: {X_test_scaled.shape}")

# --- 4. IMPLEMENTAÇÃO E TREINAMENTO DO NAIVE BAYES ---

# Criando o modelo Naive Bayes (GaussianNB para dados contínuos/Gaussianos)
nb_model = GaussianNB()

# Treinamento do modelo
print("\nTreinando o modelo Naive Bayes...")
# Naive Bayes usa os dados escalonados (X_train_scaled)
nb_model.fit(X_train_scaled, y_train)

# --- 5. AVALIAÇÃO DO MODELO ---

# Fazendo previsões no conjunto de teste
y_pred = nb_model.predict(X_test_scaled)

# Calculando a acurácia
accuracy = accuracy_score(y_test, y_pred)
print("\n--- Resultados do Naive Bayes Classifier ---")
print(f"Acurácia no Conjunto de Teste: {accuracy:.4f}")

# Relatório de Classificação (Precision, Recall, F1-Score)
report = classification_report(y_test, y_pred, target_names=target_names)
print("\nRelatório de Classificação Detalhado:")
print(report)

# Remove o filtro de avisos
warnings.filterwarnings('default')
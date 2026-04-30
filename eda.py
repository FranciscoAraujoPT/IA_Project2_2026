import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

engine = create_engine("postgresql://localhost/caseover")

contacts = pd.read_sql("SELECT * FROM contacts", engine)
conversions = pd.read_sql("SELECT * FROM conversions", engine)
customers = pd.read_sql("SELECT * FROM customers", engine)

df = contacts.merge(conversions, on="contact_id", how="left")
df = df.merge(customers[["customer_id", "customer_type"]], on="customer_id", how="left")

print("Shape:", df.shape)
print("\nColunas:", df.columns.tolist())
print("\nTipos:\n", df.dtypes)
print("\nValores nulos:\n", df.isnull().sum())
print("\nEstatísticas:\n", df["response_time_min"].describe())
print("\nConversões:\n", df["converted"].value_counts())

# limpar coluna duplicada
df = df.drop(columns=["order_id_y"]).rename(columns={"order_id_x": "order_id"})

# gráfico 1: distribuição do response_time_min
plt.figure(figsize=(10, 4))
sns.histplot(df["response_time_min"], bins=50, kde=True)
plt.title("Distribuição do tempo de resposta")
plt.xlabel("Minutos")
plt.savefig("graf_response_time.png")
plt.close()

# gráfico 2: taxa de conversão por plataforma
plt.figure(figsize=(8, 4))
conv_platform = df.groupby("platform")["converted"].mean().reset_index()
sns.barplot(data=conv_platform, x="platform", y="converted")
plt.title("Taxa de conversão por plataforma")
plt.ylabel("Taxa de conversão")
plt.savefig("graf_platform.png")
plt.close()

# gráfico 3: taxa de conversão por customer_type
plt.figure(figsize=(8, 4))
conv_type = df.groupby("customer_type")["converted"].mean().reset_index()
sns.barplot(data=conv_type, x="customer_type", y="converted")
plt.title("Taxa de conversão por tipo de cliente")
plt.ylabel("Taxa de conversão")
plt.savefig("graf_customer_type.png")
plt.close()

# gráfico 4: response_time vs converted (boxplot)
plt.figure(figsize=(8, 4))
sns.boxplot(data=df, x="converted", y="response_time_min")
plt.title("Tempo de resposta vs conversão")
plt.savefig("graf_boxplot.png")
plt.close()

print("Gráficos guardados.")



# ─ PREPARAÇÃO PARA ML ─

# 1. selecionar apenas as features relevantes para o modelo
features = df[[
    "response_time_min",
    "platform",
    "contact_reason",
    "customer_type",
    "converted"
]].copy()

features_encoded = pd.get_dummies(features, columns=["platform", "contact_reason", "customer_type"])

features_encoded["converted"] = features_encoded["converted"].astype(int)

print("Shape após encoding:", features_encoded.shape)
print("\nColunas:", features_encoded.columns.tolist())
print("\nPrimeiras linhas:\n", features_encoded.head())
print("\nValores nulos:\n", features_encoded.isnull().sum().sum(), "nulos no total")

features_encoded.to_csv("dataset_ml.csv", index=False)
print("\nFicheiro dataset_ml.csv exportado com sucesso.")
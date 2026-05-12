#!/usr/bin/env python3
"""
06_train_random_forest.py

Treina e avalia um modelo Random Forest usando os ficheiros gerados por
05_prepare_model_data.py.

Usa os ficheiros nao normalizados:
- X_train.csv
- X_test.csv
- y_train.csv
- y_test.csv

Exemplo:
python 06_train_random_forest.py --input-dir outputs_A/model_ready
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# === Secção 1: leitura da variável alvo ===
# Lê y_train/y_test e transforma o DataFrame de uma coluna numa Series de inteiros.
def load_series(path: Path) -> pd.Series:
    return pd.read_csv(path).squeeze("columns").astype(int)


def main() -> None:

    # === Secção 2: definição dos argumentos da linha de comandos ===
    # Permite escolher a pasta de entrada e configurar os hiperparâmetros principais.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory created by 05_prepare_model_data.py, e.g. outputs_A/model_ready",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of trees in the forest",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum depth of each tree",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=5,
        help="Minimum number of samples required at a leaf node",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )

    # === Secção 3: leitura dos argumentos fornecidos pelo utilizador ===
    # A partir daqui, o script usa os valores passados por CLI ou os defaults.
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    # === Secção 4: carregamento dos dados preparados ===
    # Estes ficheiros devem ter sido gerados previamente pelo 05_prepare_model_data.py.
    X_train = pd.read_csv(input_dir / "X_train.csv")
    X_test = pd.read_csv(input_dir / "X_test.csv")
    y_train = load_series(input_dir / "y_train.csv")
    y_test = load_series(input_dir / "y_test.csv")

    # === Secção 5: criação do modelo Random Forest ===
    # Usa os hiperparâmetros escolhidos e mantém class_weight="balanced".
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        class_weight="balanced",
        n_jobs=-1,
    )


    # === Secção 6: resumo inicial da execução ===
    # Mostra o dataset usado e o tamanho dos conjuntos de treino/teste.
    print("\n=== 06. TRAIN RANDOM FOREST ===\n")
    print(f"Input directory: {input_dir}")
    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")

    # === Secção 7: treino do modelo ===
    # O modelo aprende padrões apenas a partir do conjunto de treino.
    model.fit(X_train, y_train)

    # === Secção 8: previsões no conjunto de teste ===
    # O teste é usado para medir generalização, não para treinar o modelo.
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # === Secção 9: cálculo das métricas finais ===
    # Reúne desempenho e configuração do modelo numa única estrutura.
    metrics = {
        "model": "random_forest",
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features": X_train.shape[1],
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth if args.max_depth is not None else "None",
        "min_samples_leaf": args.min_samples_leaf,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
    }


    # === Secção 10: gravação das métricas ===
    # Guarda as métricas finais num CSV dentro da pasta model_ready.
    metrics_path = input_dir / "random_forest_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    
    # === Secção 11: cálculo da importância das variáveis ===
    # A Random Forest estima quanto cada feature contribuiu para as divisões das árvores.
    importance = (
        pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    # === Secção 12: gravação da importância das variáveis ===
    # Ordena e guarda as features mais relevantes para interpretação do modelo.
    importance["importance"] = importance["importance"].round(6)

    importance_path = input_dir / "random_forest_feature_importance.csv"
    importance.to_csv(importance_path, index=False)


    # === Secção 13: gravação do modelo treinado ===
    # Guarda o modelo final para poder ser reutilizado sem treinar novamente.
    model_path = input_dir / "random_forest_model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(model, f)


    # === Secção 14: resumo final no terminal ===
    # Mostra as métricas principais e os caminhos dos ficheiros gerados.
    print("\nMetrics:")
    for key in ["accuracy", "roc_auc", "f1", "precision", "recall"]:
        print(f"- {key}: {metrics[key]}")

    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Feature importance saved to: {importance_path}")
    print(f"Model saved to: {model_path}")
    print("\nRandom Forest training completed.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
08_train_logistic_regression.py

Treina e avalia um modelo Logistic Regression usando os ficheiros gerados por
05_prepare_model_data.py.

Usa os ficheiros normalizados:
- X_train_scaled_for_logistic_regression.csv
- X_test_scaled_for_logistic_regression.csv
- y_train.csv
- y_test.csv

Exemplo:
python 08_train_logistic_regression.py --input-dir outputs_A/model_ready
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
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
        "--c",
        type=float,
        default=1.0,
        help="Inverse of regularization strength",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum number of optimization iterations",
    )
    parser.add_argument(
        "--no-balanced-class-weight",
        action="store_true",
        help="Disable class_weight='balanced' during training",
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
    # Logistic Regression usa os ficheiros X normalizados gerados pelo script 05.
    X_train = pd.read_csv(input_dir / "X_train_scaled_for_logistic_regression.csv")
    X_test = pd.read_csv(input_dir / "X_test_scaled_for_logistic_regression.csv")
    y_train = load_series(input_dir / "y_train.csv")
    y_test = load_series(input_dir / "y_test.csv")

    # === Secção 5: criação do modelo Logistic Regression ===
    # A regularização ajuda a reduzir overfitting, especialmente com one-hot encoding.
    class_weight = None if args.no_balanced_class_weight else "balanced"
    model = LogisticRegression(
        C=args.c,
        max_iter=args.max_iter,
        class_weight=class_weight,
        random_state=args.random_state,
    )

    # === Secção 6: resumo inicial da execução ===
    # Mostra o dataset usado e o tamanho dos conjuntos de treino/teste.
    print("\n=== 08. TRAIN LOGISTIC REGRESSION ===\n")
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
    positive_class_index = list(model.classes_).index(1)
    y_prob = model.predict_proba(X_test)[:, positive_class_index]

    roc_auc = round(roc_auc_score(y_test, y_prob), 4) if y_test.nunique() == 2 else None

    # === Secção 9: cálculo das métricas finais ===
    # Reúne desempenho e configuração do modelo numa única estrutura.
    metrics = {
        "model": "logistic_regression",
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features": X_train.shape[1],
        "c": args.c,
        "max_iter": args.max_iter,
        "balanced_class_weight": not args.no_balanced_class_weight,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "roc_auc": roc_auc,
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
    }

    # === Secção 10: gravação das métricas ===
    # Guarda as métricas finais num CSV dentro da pasta model_ready.
    metrics_path = input_dir / "logistic_regression_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    # === Secção 11: cálculo da importância das variáveis ===
    # Na Logistic Regression, a importância é o valor absoluto do coeficiente.
    importance = (
        pd.DataFrame({
            "feature": X_train.columns,
            "coefficient": model.coef_[0],
        })
        .assign(abs_coefficient=lambda df: df["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )

    # === Secção 12: gravação da importância das variáveis ===
    # Guarda coeficientes com sinal e magnitude absoluta para interpretação.
    importance["coefficient"] = importance["coefficient"].round(6)
    importance["abs_coefficient"] = importance["abs_coefficient"].round(6)

    importance_path = input_dir / "logistic_regression_feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    # === Secção 13: gravação do modelo treinado ===
    # Guarda o modelo final para poder ser reutilizado sem treinar novamente.
    model_path = input_dir / "logistic_regression_model.pkl"
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
    print("\nLogistic Regression training completed.")


if __name__ == "__main__":
    main()

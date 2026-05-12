# Preparação, validação e análise dos datasets

## `01_validate_data.py` — Validação dos dados

Este script valida a estrutura base de cada dataset.

Verifica:

- 500 linhas
- 14 colunas
- `contact_id` único
- sem nulos inesperados
- `response_time_min >= 0`
- datas válidas
- `converted` só com 0/1
- `customer_type` correto
- `contact_source_type` correto

**Objetivo:**  
Garantir que o dataset está completo, consistente e pronto para as próximas etapas.

---

## `02_confirm_derived_variables.py` — Confirmação de variáveis derivadas

Este script confirma se as variáveis calculadas foram geradas corretamente.

Verifica:

- `response_time_min` bate certo com as datas
- `is_payday_period` foi calculado a partir de `contact_datetime`
- `is_peak_season` foi calculado a partir de `contact_datetime`
- `converted` bate certo com a tabela `purchases`

**Objetivo:**  
Confirmar que as variáveis derivadas são coerentes com as regras definidas para o dataset.

---

## `03_exploratory_analysis.py` — Análise exploratória

Este script calcula tabelas e gráficos para perceber os padrões de conversão.

Foram calculadas taxas de conversão por:

- tempo de resposta
- `contact_source_type`
- `customer_type`
- `product_category`
- `product_subcategory`
- `product_collection`
- `is_payday_period`
- `is_peak_season`

**Objetivo:**  
Identificar padrões iniciais nos dados antes de treinar modelos preditivos.

Exemplos de outputs:

- `conversion_by_response_time_bucket.csv`
- `conversion_by_contact_source_type.csv`
- `conversion_by_customer_type.csv`
- gráficos `.png` correspondentes

---

## `04_create_features.py` — Criação de novas variáveis

Este script cria variáveis adicionais a partir de `contact_datetime`.

A partir de `contact_datetime`, foram criadas:

- `month`
- `day_of_month`
- `day_of_week`
- `hour_of_day`

Também foi criada:

- `fast_response_24h`

Onde:

```text
fast_response_24h = 1 se response_time_min <= 1440
fast_response_24h = 0 se response_time_min > 1440
```

**Objetivo:**  
Transformar a data original em variáveis temporais úteis para o modelo.

---

## `05_prepare_model_data.py` — Preparação dos dados para o modelo

Este script prepara os datasets para treino e teste dos modelos preditivos.

### Separação entre X e y

```text
X = variáveis explicativas, ou seja, a informação usada para prever
y = converted, ou seja, aquilo que queremos prever
```

### Variáveis excluídas do modelo

Foram excluídas:

- `contact_id`
- `customer_id`
- `contact_datetime`
- `first_response_datetime`
- `product_id`

Motivos:

`contact_id` e `customer_id` foram excluídos porque são identificadores e poderiam levar o modelo a memorizar casos específicos.

`contact_datetime` foi substituído por variáveis temporais derivadas.

`first_response_datetime` foi excluído porque a informação relevante já está contida em `response_time_min`.

`product_id` foi excluído porque é um identificador de produto. A informação relevante está capturada pelas variáveis categóricas `product_category`, `product_subcategory` e `product_collection`, que generalizam padrões em vez de memorizar produtos específicos.

### Variáveis não usadas por risco de data leakage

As variáveis da tabela `purchases`, como:

- `purchase_datetime`
- `purchase_value`

não foram usadas como input do modelo, pois só são conhecidas depois da compra e poderiam causar data leakage.

### Divisão treino/teste

Os dados foram divididos em:

- 80% para treino
- 20% para teste

Num dataset de 500 linhas:

- 400 linhas para treino
- 100 linhas para teste

O conjunto de treino é usado para o modelo aprender padrões.  
O conjunto de teste é usado para avaliar se o modelo consegue generalizar para dados novos.

### Ficheiros gerados para Logistic Regression

Para Logistic Regression, devem ser usados:

- `X_train_scaled_for_logistic_regression.csv`
- `X_test_scaled_for_logistic_regression.csv`
- `y_train.csv`
- `y_test.csv`

Estes ficheiros X estão normalizados.

### Ficheiros gerados para Random Forest e Gradient Boosting

Para outros algoritmos, como Random Forest e Gradient Boosting, usam-se os ficheiros não normalizados:

- `X_train.csv`
- `X_test.csv`
- `y_train.csv`
- `y_test.csv`

**Nota:**  
Random Forest e Gradient Boosting não precisam de normalização.

---

## `08_train_logistic_regression.py` — Treino de Logistic Regression

Este script treina e avalia um modelo Logistic Regression a partir dos ficheiros
gerados pelo `05_prepare_model_data.py`.

Usa os ficheiros normalizados:

- `X_train_scaled_for_logistic_regression.csv`
- `X_test_scaled_for_logistic_regression.csv`
- `y_train.csv`
- `y_test.csv`

**Objetivo:**  
Treinar um modelo linear de classificação e avaliar o desempenho no conjunto de
teste. A normalização é usada porque Logistic Regression é sensível à escala
das variáveis.

Ficheiros gerados:

- `logistic_regression_metrics.csv`
- `logistic_regression_feature_importance.csv`
- `logistic_regression_model.pkl`

Exemplo:

```bash
python 08_train_logistic_regression.py --input-dir outputs_A/model_ready
```

---

## `06_train_random_forest.py` — Treino de Random Forest

Este script treina e avalia um modelo Random Forest a partir dos ficheiros
gerados pelo `05_prepare_model_data.py`.

Usa os ficheiros não normalizados:

- `X_train.csv`
- `X_test.csv`
- `y_train.csv`
- `y_test.csv`

**Objetivo:**  
Treinar um modelo baseado em árvores e avaliar o desempenho no conjunto de
teste, sem usar os ficheiros normalizados.

Ficheiros gerados:

- `random_forest_metrics.csv`
- `random_forest_feature_importance.csv`
- `random_forest_model.pkl`

Exemplo:

```bash
python 06_train_random_forest.py --input-dir outputs_A/model_ready
```

---

## `07_train_gradient_boosting.py` — Treino de Gradient Boosting

Este script treina e avalia um modelo Gradient Boosting a partir dos ficheiros
gerados pelo `05_prepare_model_data.py`.

Usa os ficheiros não normalizados:

- `X_train.csv`
- `X_test.csv`
- `y_train.csv`
- `y_test.csv`

**Objetivo:**  
Treinar um segundo modelo baseado em árvores, mas com aprendizagem sequencial:
cada nova árvore tenta corrigir erros das árvores anteriores.

Como `GradientBoostingClassifier` não tem `class_weight`, o script usa pesos
de amostra balanceados por defeito para lidar com possíveis diferenças entre
as classes.

Ficheiros gerados:

- `gradient_boosting_metrics.csv`
- `gradient_boosting_feature_importance.csv`
- `gradient_boosting_model.pkl`

Exemplo:

```bash
python 07_train_gradient_boosting.py --input-dir outputs_A/model_ready
```

---

# Validação geral

Os três datasets estão consistentes:

```text
500 linhas
14 colunas
sem nulos inesperados
contact_id único
response_time_min válido
datas válidas
converted válido
customer_type correto
contact_source_type correto
```

E as variáveis derivadas também estão corretas:

```text
response_time_min: OK
is_payday_period: OK
is_peak_season: OK
converted: OK
0 mismatches
```

---

# Conclusões por dataset

## Dataset A — causalidade provável / associação forte

O Dataset A mostra um padrão muito claro: quanto menor o tempo de resposta, maior a conversão.

| Tempo de resposta | Conversão |
|---|---:|
| `<1h` | 73,6% |
| `1–4h` | 65,4% |
| `4–12h` | 48,1% |
| `12–24h` | 30,8% |
| `>24h` | 17,0% |

Também há diferenças por outros atributos:

| Variável | Padrão |
|---|---|
| `contact_source_type` | `product_page` converte mais: 53,8% vs 43,4% |
| `customer_type` | `two_times` e `vip` convertem mais |
| `is_peak_season` | 55,9% vs 46,0% |
| `is_payday_period` | diferença pequena: 53,2% vs 49,1% |

Diferença entre resposta até 24h e resposta acima de 24h:

```text
+40,6 pontos percentuais
IC 95%: [+31,6; +49,6]
```

**Conclusão:**  
No Dataset A, existe uma associação forte, gradual e estatisticamente clara entre menor tempo de resposta e maior conversão. A evidência é compatível com causalidade provável, embora, por ser uma base observacional, não permita afirmar causalidade absoluta sem teste experimental.

---

## Dataset B — inconclusivo

O Dataset B não mostra padrões fortes nem consistentes.

| Tempo de resposta | Conversão |
|---|---:|
| `<1h` | 31,7% |
| `1–4h` | 35,0% |
| `4–12h` | 31,9% |
| `12–24h` | 26,9% |
| `>24h` | 35,0% |

As diferenças principais são fracas:

| Variável | Diferença | IC 95% |
|---|---:|---:|
| `≤24h` vs `>24h` | -3,5 p.p. | [-12,4; +5,4] |
| `peak_season` vs não peak | +5,7 p.p. | [-2,8; +14,1] |
| `payday_period` vs não payday | +0,5 p.p. | [-9,8; +10,8] |
| `product_page` vs geral | +5,3 p.p. | [-3,0; +13,7] |

Como os intervalos de confiança incluem zero, não há evidência estatística forte.

**Conclusão:**  
No Dataset B, não existe evidência suficiente para afirmar que `response_time_min`, `contact_source_type`, `customer_type`, `is_payday_period`, `is_peak_season` ou categorias de produto expliquem claramente a conversão. As diferenças observadas são pequenas e compatíveis com ruído.

---

## Dataset C — correlação com confounding

O Dataset C mostra uma correlação aparente entre resposta rápida e conversão.

| Tempo de resposta | Conversão |
|---|---:|
| `<1h` | 62,7% |
| `1–4h` | 54,8% |
| `4–12h` | 58,9% |
| `12–24h` | 50,8% |
| `>24h` | 28,1% |

No agregado, responder até 24h parece muito melhor:

```text
≤24h: 57,0%
>24h: 28,1%
diferença: +28,9 p.p.
IC 95%: [+20,6; +37,2]
```

No entanto, as variáveis contextuais também apresentam diferenças relevantes, especialmente `is_peak_season`:

| Variável | Conversão |
|---|---:|
| `is_peak_season = 1` | 64,8% |
| `is_peak_season = 0` | 27,8% |
| `is_payday_period = 1` | 58,2% |
| `is_payday_period = 0` | 40,3% |

Além disso, a resposta rápida está muito concentrada em peak season:

```text
Fast response em peak season: 91,2%
Fast response fora de peak season: 26,4%
```

Isto indica que `is_peak_season` pode estar a influenciar simultaneamente duas coisas:
    - maior conversão
    - respostas mais rápidas


Portanto, existe risco de confounding: a relação entre resposta rápida e conversão pode estar parcialmente explicada pela sazonalidade. `is_peak_season` parece ser o principal fator contextual, enquanto `is_payday_period` também é relevante, mas não necessariamente mais forte do que o efeito agregado do tempo de resposta.

**Conclusão:**  
No Dataset C, existe uma correlação agregada entre resposta rápida e conversão, mas essa relação é fortemente influenciada por `is_peak_season` e também por `is_payday_period`. A sazonalidade explica uma parte importante da associação observada, pelo que não se deve concluir diretamente que o tempo de resposta causa a conversão. A interpretação correta é correlação com forte risco de confounding, não causalidade confirmada.

---

# Conclusão geral final

Após a validação e análise exploratória, os três datasets apresentam comportamentos distintos. O Dataset A mostra uma associação forte e gradual entre tempo de resposta e conversão, compatível com causalidade provável. O Dataset B não apresenta evidência estatística clara de que os atributos analisados expliquem a conversão, sendo por isso inconclusivo. O Dataset C apresenta uma correlação aparente entre resposta rápida e conversão, mas essa relação é fortemente confundida por fatores contextuais, sobretudo `is_peak_season` e `is_payday_period`.

No Dataset C, deve evitar-se dizer que “a época explica tudo”. O mais rigoroso é dizer que explica uma parte importante e enfraquece a interpretação causal do tempo de resposta.

#!/usr/bin/env bash

set -e

echo "=========================================="
echo " CaseOver - Model Preparation Pipeline"
echo "=========================================="
echo ""

run_pipeline () {
  DATASET_NAME=$1
  CONTACTS_FILE=$2
  PURCHASES_FILE=$3
  OUTPUT_DIR=$4

  echo "------------------------------------------"
  echo "Running pipeline for: ${DATASET_NAME}"
  echo "Contacts: ${CONTACTS_FILE}"
  echo "Purchases: ${PURCHASES_FILE}"
  echo "Output: ${OUTPUT_DIR}"
  echo "------------------------------------------"

  python3 01_validate_data.py \
    --contacts "${CONTACTS_FILE}" \
    --output-dir "${OUTPUT_DIR}"

  python3 02_confirm_derived_variables.py \
    --contacts "${CONTACTS_FILE}" \
    --purchases "${PURCHASES_FILE}" \
    --output-dir "${OUTPUT_DIR}"

  python3 03_exploratory_analysis.py \
    --contacts "${CONTACTS_FILE}" \
    --output-dir "${OUTPUT_DIR}/eda"

  python3 04_create_features.py \
    --contacts "${CONTACTS_FILE}" \
    --output-dir "${OUTPUT_DIR}"

  FEATURED_FILE="${OUTPUT_DIR}/$(basename "${CONTACTS_FILE}" .csv)_with_features.csv"

  python3 05_prepare_model_data.py \
    --input "${FEATURED_FILE}" \
    --output-dir "${OUTPUT_DIR}/model_ready"

  python3 06_train_random_forest.py \
    --input-dir "${OUTPUT_DIR}/model_ready"

  echo ""
  echo "✅ Finished ${DATASET_NAME}"
  echo ""
}

run_pipeline \
  "Dataset A - Causality Likely" \
  "dataset_A_causality_likely_contacts_train.csv" \
  "dataset_A_causality_likely_purchases.csv" \
  "outputs_A"

run_pipeline \
  "Dataset B - Inconclusive" \
  "dataset_B_inconclusive_contacts_train.csv" \
  "dataset_B_inconclusive_purchases.csv" \
  "outputs_B"

run_pipeline \
  "Dataset C - Correlation Not Causality" \
  "dataset_C_correlation_not_causality_contacts_train.csv" \
  "dataset_C_correlation_not_causality_purchases.csv" \
  "outputs_C"

echo "=========================================="
echo "✅ All pipelines completed successfully"
echo "=========================================="

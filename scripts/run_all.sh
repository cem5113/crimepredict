#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# ÜÇ MOTORLU FORECAST PIPELINE
# ============================================================

# ---------- Varsayılanlar ----------
PYTHON="python"
DATA_ROOT="data"
MODELS_ROOT="models"
FEATURES_SHORT="data/features/short"
FEATURES_MID="data/features/mid"
FEATURES_LONG="data/features/long"
OUTPUTS_ROOT="data/outputs"

SHORT_H="1,2,3,6,12,24,48,72"
MID_H="96,168,336,504,720"
LONG_H="960,1440,2160"
PRECISION_K="20,50"

NFOLDS_SHORT=4
NFOLDS_MID=3
NFOLDS_LONG=3

CONF_LAM_SHORT=0.05
CONF_LAM_MID=0.09
CONF_LAM_LONG=0.15

STAGES="features,train_short,train_mid,train_long,infer_short,infer_mid,infer_long,merge"
SKIP=""
WRITE_LOGS=1
DRYRUN=0

# ---------- Argüman ayrıştırma ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --models-root) MODELS_ROOT="$2"; shift 2 ;;
    --features-short) FEATURES_SHORT="$2"; shift 2 ;;
    --features-mid) FEATURES_MID="$2"; shift 2 ;;
    --features-long) FEATURES_LONG="$2"; shift 2 ;;
    --outputs-root) OUTPUTS_ROOT="$2"; shift 2 ;;
    --short-horizons) SHORT_H="$2"; shift 2 ;;
    --mid-horizons) MID_H="$2"; shift 2 ;;
    --long-horizons) LONG_H="$2"; shift 2 ;;
    --precision-k) PRECISION_K="$2"; shift 2 ;;
    --nfolds-short) NFOLDS_SHORT="$2"; shift 2 ;;
    --nfolds-mid) NFOLDS_MID="$2"; shift 2 ;;
    --nfolds-long) NFOLDS_LONG="$2"; shift 2 ;;
    --conf-lambda-short) CONF_LAM_SHORT="$2"; shift 2 ;;
    --conf-lambda-mid) CONF_LAM_MID="$2"; shift 2 ;;
    --conf-lambda-long) CONF_LAM_LONG="$2"; shift 2 ;;
    --stages) STAGES="$2"; shift 2 ;;
    --skip) SKIP="$2"; shift 2 ;;
    --no-logs) WRITE_LOGS=0; shift 1 ;;
    --dry-run) DRYRUN=1; shift 1 ;;
    -h|--help) sed -n '1,160p' "$0"; exit 0 ;;
    *) echo "❌ Bilinmeyen argüman: $1" && exit 1 ;;
  esac
done

# ---------- Yardımcılar ----------
ts() { date +"%Y-%m-%d %H:%M:%S"; }
run() {
  echo "[$(ts)] $*"
  [[ $DRYRUN -eq 0 ]] && eval "$@"
}
contains() {
  local csv="$1" needle="$2"
  IFS=',' read -ra arr <<< "$csv"
  for i in "${arr[@]}"; do
    [[ "$i" == "$needle" ]] && return 0
  done
  return 1
}
skip_block() {
  local name="$1"
  if contains "$SKIP" "$name"; then
    return 0
  fi
  return 1
}

# ---------- Loglama ----------
RUN_ID=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${DATA_ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/run_${RUN_ID}.log"

if [[ $WRITE_LOGS -eq 1 && $DRYRUN -eq 0 ]]; then
  exec > >(tee -a "$LOG_FILE") 2>&1
  echo "📜 Log dosyası: $LOG_FILE"
fi

echo "=== 🚀 Üç Motorlu Forecast Pipeline Başlıyor [RUN ${RUN_ID}] ==="
echo "Stages: $STAGES"
echo "Skip:   ${SKIP:-<yok>}"
echo "Roots:  DATA=$DATA_ROOT | MODELS=$MODELS_ROOT | OUTPUTS=$OUTPUTS_ROOT"
echo

# ---------- Ön kontroller ----------
command -v "$PYTHON" >/dev/null 2>&1 || { echo "❌ HATA: Python bulunamadı: $PYTHON"; exit 1; }
mkdir -p "$OUTPUTS_ROOT" "$MODELS_ROOT" "$DATA_ROOT/features" "$LOG_DIR" metrics/{short,mid,long}

# ============================================================
# 1) FEATURES
# ============================================================
if contains "$STAGES" "features" && ! skip_block "features"; then
  echo ">> [1/8] Feature üretimi"
  run "$PYTHON scripts/01_build_features.py \
    --input ${DATA_ROOT}/raw/sf_crime_full.parquet \
    --outdir ${DATA_ROOT}/features \
    --short ${SHORT_H} \
    --mid ${MID_H} \
    --long ${LONG_H} \
    --freq 1H"
else
  echo ">> [1/8] Feature üretimi atlandı."
fi

# ============================================================
# 2) TRAIN SHORT
# ============================================================
if contains "$STAGES" "train_short" && ! skip_block "short"; then
  echo ">> [2/8] SHORT eğitim"
  run "$PYTHON scripts/02_train_short.py \
    --features_dir ${FEATURES_SHORT} \
    --out_models ${MODELS_ROOT}/short \
    --out_metrics metrics/short \
    --horizons ${SHORT_H} \
    --n_folds ${NFOLDS_SHORT} \
    --precision_k ${PRECISION_K}"
else
  echo ">> [2/8] SHORT eğitim atlandı."
fi

# ============================================================
# 3) TRAIN MID
# ============================================================
if contains "$STAGES" "train_mid" && ! skip_block "mid"; then
  echo ">> [3/8] MID eğitim"
  run "$PYTHON scripts/03_train_mid.py \
    --features_dir ${FEATURES_MID} \
    --out_models ${MODELS_ROOT}/mid \
    --out_metrics metrics/mid \
    --horizons ${MID_H} \
    --n_folds ${NFOLDS_MID} \
    --precision_k ${PRECISION_K} \
    --use_rf False"
else
  echo ">> [3/8] MID eğitim atlandı."
fi

# ============================================================
# 4) TRAIN LONG
# ============================================================
if contains "$STAGES" "train_long" && ! skip_block "long"; then
  echo ">> [4/8] LONG eğitim"
  run "$PYTHON scripts/04_train_long.py \
    --features_dir ${FEATURES_LONG} \
    --out_models ${MODELS_ROOT}/long \
    --out_metrics metrics/long \
    --horizons ${LONG_H} \
    --n_folds ${NFOLDS_LONG} \
    --precision_k ${PRECISION_K}"
else
  echo ">> [4/8] LONG eğitim atlandı."
fi

# ============================================================
# 5) INFER SHORT
# ============================================================
if contains "$STAGES" "infer_short" && ! skip_block "short"; then
  echo ">> [5/8] SHORT inference"
  run "$PYTHON scripts/05_infer_short.py \
    --features_dir ${FEATURES_SHORT} \
    --models_dir ${MODELS_ROOT}/short \
    --out_path ${OUTPUTS_ROOT}/risk_short.parquet \
    --horizons ${SHORT_H} \
    --confidence_lambda ${CONF_LAM_SHORT}"
else
  echo ">> [5/8] SHORT inference atlandı."
fi

# ============================================================
# 6) INFER MID
# ============================================================
if contains "$STAGES" "infer_mid" && ! skip_block "mid"; then
  echo ">> [6/8] MID inference"
  run "$PYTHON scripts/06_infer_mid.py \
    --features_dir ${FEATURES_MID} \
    --models_dir ${MODELS_ROOT}/mid \
    --out_path ${OUTPUTS_ROOT}/risk_mid.parquet \
    --horizons ${MID_H} \
    --confidence_lambda ${CONF_LAM_MID}"
else
  echo ">> [6/8] MID inference atlandı."
fi

# ============================================================
# 7) INFER LONG
# ============================================================
if contains "$STAGES" "infer_long" && ! skip_block "long"; then
  echo ">> [7/8] LONG inference"
  run "$PYTHON scripts/07_infer_long.py \
    --features_dir ${FEATURES_LONG} \
    --models_dir ${MODELS_ROOT}/long \
    --out_path ${OUTPUTS_ROOT}/risk_long.parquet \
    --horizons ${LONG_H} \
    --confidence_lambda ${CONF_LAM_LONG}"
else
  echo ">> [7/8] LONG inference atlandı."
fi

# ============================================================
# 8) MERGE
# ============================================================
if contains "$STAGES" "merge" && ! skip_block "merge"; then
  echo ">> [8/8] Birleştirme"
  run "$PYTHON scripts/08_merge_outputs.py \
    --short ${OUTPUTS_ROOT}/risk_short.parquet \
    --mid   ${OUTPUTS_ROOT}/risk_mid.parquet \
    --long  ${OUTPUTS_ROOT}/risk_long.parquet \
    --out   ${OUTPUTS_ROOT}/risk_hourly_by_category.parquet \
    --out_daily ${OUTPUTS_ROOT}/risk_daily_by_category.parquet \
    --daily_how mean"
else
  echo ">> [8/8] Birleştirme atlandı."
fi

# ============================================================
# Tamamlandı
# ============================================================
echo
echo "✅ === PIPELINE TAMAMLANDI ==="
echo "Çıktılar:"
echo "  - ${OUTPUTS_ROOT}/risk_short.parquet"
echo "  - ${OUTPUTS_ROOT}/risk_mid.parquet"
echo "  - ${OUTPUTS_ROOT}/risk_long.parquet"
echo "  - ${OUTPUTS_ROOT}/risk_hourly_by_category.parquet"
echo "  - ${OUTPUTS_ROOT}/risk_daily_by_category.parquet"
echo
echo "RUN ID: ${RUN_ID}"
[[ $WRITE_LOGS -eq 1 && $DRYRUN -eq 0 ]] && echo "Log: ${LOG_FILE}"

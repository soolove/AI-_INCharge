#!/usr/bin/env bash
# Training of the TCN used in this folder: width 64 on tcn-dataset-v3-5g-31d (5G units, connection
# delay 0-6 min plus the building-type lead of engine spec 6), days 0-10 and 15-30 for training,
# 11-12 for validation, 13-14 for test. The defaults reproduce
# checkpoints/v3_5g_31d_w64_dual_proposal/ (best epoch 87, KPI1 9.5%):
#   stage 1: EPOCHS1=60 epochs, lr 1e-3
#   stage 2: copy stage 1 and resume at lr 1e-4 up to epoch EPOCHS2=90 (best.pt = lowest val loss)
#   batch 128, seed 0, 5 channels (traffic, UE count, ISAC, hour sin/cos), pinball 0.5 / 0.9.
# Then val/test reports and KPI1 MAPE (on the test days of the split). NPZ, OUT, CHANNELS (model
# width), SPLIT (train_dual.py --train-days/--val-days/--test-days), EPOCHS1 and EPOCHS2 can be
# overridden. The script stops if OUT already holds a run, so retrain into a new OUT.
#
# QUICK=1 is the minimal run that only checks that the steps work (about a minute on a GPU): the
# 3-day dataset tcn-dataset-quick (build_tcn_dataset.py --days 3), train day 0 / val day 1 /
# test day 2, 2 + 1 epochs, output checkpoints/quick. Its numbers are not comparable to the reported
# model.
#
#   tmux new -s tcn  ->  OUT=checkpoints/<new name> bash DGU_SC_Project/run_tcn_v3.sh
#   QUICK=1 bash DGU_SC_Project/run_tcn_v3.sh

# Run inside the conda environment of environment.yml (conda activate smart_campus_5G).
cd "$(dirname "$0")"
set -eo pipefail

if [ "${QUICK:-0}" = 1 ]; then
    NPZ=${NPZ:-../tcn-dataset-quick/tcn_dataset.npz}
    OUT=${OUT:-checkpoints/quick}
    SPLIT=${SPLIT:---train-days 0 --val-days 1 --test-days 2}
    EPOCHS1=${EPOCHS1:-2}
    EPOCHS2=${EPOCHS2:-3}
fi
NPZ=${NPZ:-../tcn-dataset-v3-5g-31d/tcn_dataset.npz}
OUT=${OUT:-checkpoints/v3_5g_31d_w64_dual_proposal}
CHANNELS=${CHANNELS:-64}  # TCN width (the original AI-team model was 32)
EPOCHS1=${EPOCHS1:-60}
EPOCHS2=${EPOCHS2:-90}
read -r -a SPLIT_ARGS <<< "${SPLIT:---train-days 0-10,15-30 --val-days 11-12 --test-days 13-14}"
FEATS=Traffic_demand_mbps,Ue_count,Isac_density,Hour_sin,Hour_cos
HL=464  # 0.8 x C_max 580 Mbps; only for the high-load report block
COMMON=(--npz "$NPZ" --batch-size 128 --seed 0 --features "$FEATS" --high-load-threshold "$HL"
        --channels "$CHANNELS" "${SPLIT_ARGS[@]}")

stamp() { echo; echo "=== [$(date '+%F %T')] $* ==="; }

if [ -e "$OUT/seed_0" ]; then
    echo "$OUT/seed_0 already exists; move it away before retraining" >&2
    exit 1
fi
mkdir -p "$OUT"
sha256sum "$NPZ" | tee "$OUT/dataset.sha256"

stamp "stage 1: $EPOCHS1 epochs, lr 1e-3"
python train_dual.py "${COMMON[@]}" --output "$OUT/seed_0" --epochs "$EPOCHS1" --lr 1e-3 \
    | tee "$OUT/seed_0.log"

stamp "stage 2: resume at lr 1e-4 up to epoch $EPOCHS2"
mkdir -p "$OUT/seed_0_finetune"
cp "$OUT"/seed_0/{last.pt,best.pt,metrics.jsonl} "$OUT/seed_0_finetune/"
python train_dual.py "${COMMON[@]}" --output "$OUT/seed_0_finetune" \
    --resume "$OUT/seed_0_finetune/last.pt" --epochs "$EPOCHS2" --lr 1e-4 \
    | tee "$OUT/seed_0_finetune.log"

stamp "evaluation (val, test)"
for split in val test; do
    python train_dual.py --npz "$NPZ" --evaluate "$OUT/seed_0_finetune/best.pt" --split "$split" \
        --high-load-threshold "$HL" | tee "$OUT/eval_$split.log"
done

stamp "KPI1 MAPE"
python eval_kpi1.py --npz "$NPZ" --ckpt "$OUT/seed_0_finetune/best.pt" --cmax 580 \
    | tee "$OUT/kpi1.log"
stamp "done"

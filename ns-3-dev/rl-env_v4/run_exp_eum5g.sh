#!/usr/bin/env bash
# e-Um 5G (이음5G) RL experiments for the one agent of this folder: PPO with reward ratio
# energy : QoS : overload = 1:10:5 and the width-64 TCN, on the e-Um 5G radio settings that are the
# ns-3 defaults (carrier 4.77 GHz, building entry loss 16.2 dB for indoor UEs; see README_env.md).
# The weights are given as decimals that sum to 1, and --reward-scale 16 (the sum of the integer
# ratio) makes the training signal equal to the one with integer weights; logged rewards are on the
# 0-1 scale.
#
# Evaluation uses --outage-samples 1 (outage judged at the minute step only): per-second judging
# re-evaluates every connected UE 60 times a minute, so it is exact but about ten times slower.
# Training and tuning always judge at the minute step.
#
#   bash rl-env_v4/run_exp_eum5g.sh impact            the agent and reactive thresholds trained in v2
#                                                     (ppo_v2_w64_r10, reactive_tuned_v2_r10), test 15 days,
#                                                     on the e-Um 5G and on the v2 radio settings
#   tmux new -s eum_tune   ->  bash rl-env_v4/run_exp_eum5g.sh tune     (reactive grid on the val days)
#   tmux new -s eum_train  ->  bash rl-env_v4/run_exp_eum5g.sh train    (train -> wait for the reactive
#                                                     file -> test evaluation) [extra train_ppo.py args]
#   bash rl-env_v4/run_exp_eum5g.sh quick             minimal run that only checks that the steps work
#                                                     (a few minutes): reactive grid of 3 pairs on 1 val
#                                                     day per scenario, PPO 2 updates of 1 day, test 1 day
#                                                     per scenario, into rl-env_v4/runs/quick/. It uses the
#                                                     TCN of QUICK=1 DGU_SC_Project/run_tcn_v3.sh and its z
#                                                     stats (TCN_CKPT, Z_STATS override them). Its numbers
#                                                     are not comparable to the reported agent.
#
# Every run has its own work folder, so they can run at once. Each evaluation process keeps two
# threads busy (ns-3 and Python wait for each other by spinning), so EVAL_WORKERS (default 3) of
# about 8 suits the server's 16 threads when nothing else runs.

# Run inside the conda environment of environment.yml (conda activate smart_campus_5G).
[ -d /snap/bin ] && export PATH="/snap/bin:$PATH"  # cmake installed as a snap, used by ./ns3
cd "$(dirname "$0")/.."
set -eo pipefail
mkdir -p rl-env_v4/runs

# reward weights 1:10:5 as decimals (sum 1) and the matching reward scale
W_E=0.0625
W_Q=0.625
W_O=0.3125
SCALE=16
AGENT=rl-env_v4/runs/ppo_v2_w64_r10/best.pt        # the agent trained in v2
AGENT_REACTIVE=rl-env_v4/runs/reactive_tuned_v2_r10.json
NEW_NAME=ppo_eum5g_w64_r10                          # a retrained agent goes here
NEW_REACTIVE=rl-env_v4/runs/reactive_tuned_eum5g_r10.json
stamp() { echo; echo "=== [$(date '+%F %T')] $* ==="; }

UPS=0.005,0.0075,0.01,0.015,0.02,0.03,0.05,0.075,0.1,0.15,0.2
DOWNS=0.001,0.0025,0.005,0.0075,0.01,0.02,0.03,0.05,0.1

case "$1" in
impact)
    # Same agent, thresholds and protocol on both radio settings, so the days pair up exactly.
    out=rl-env_v4/runs/eum5g_impact
    mkdir -p "$out"
    for radio in eum5g v2radio; do
        extra=()
        [ "$radio" = v2radio ] && extra=(--ns3-arg slmCarrierFrequency=3.5e9 --ns3-arg o2iLossDb=0)
        name="eval_test_v2agent_r10_$radio"
        stamp "impact: $AGENT + $AGENT_REACTIVE, radio $radio -> $out/$name.json"
        python rl-env_v4/evaluate.py --ckpt "$AGENT" --days 5 --reactive "$AGENT_REACTIVE" \
            --outage-samples 1 --workdir rl-env_v4/run/eval_impact --workers "${EVAL_WORKERS:-3}" \
            "${extra[@]}" --out "$out/$name.json" | tee "$out/$name.log"
    done
    stamp "impact done"
    ;;
tune)
    stamp "reactive threshold tuning (w_E=$W_E w_Q=$W_Q w_O=$W_O) -> $NEW_REACTIVE"
    python rl-env_v4/tune_reactive.py --w-energy "$W_E" --w-qos "$W_Q" --w-overload "$W_O" \
        --ups "$UPS" --downs "$DOWNS" --out "$NEW_REACTIVE" --workdir rl-env_v4/run/tune_eum5g \
        | tee "${NEW_REACTIVE%.json}.log"
    stamp "tuning done"
    ;;
train)
    shift
    stamp "training: e-Um 5G, TCN width 64, 1:10:5 (reward-scale $SCALE), extra [$*] -> rl-env_v4/runs/$NEW_NAME"
    python rl-env_v4/train_ppo.py --out "rl-env_v4/runs/$NEW_NAME" --w-energy "$W_E" --w-qos "$W_Q" \
        --w-overload "$W_O" --reward-scale "$SCALE" "$@" \
        | tee "rl-env_v4/runs/$NEW_NAME.log"
    stamp "waiting for the reactive file: $NEW_REACTIVE"
    until [ -f "$NEW_REACTIVE" ]; do sleep 60; done
    stamp "test evaluation (minute-step outage, ${EVAL_WORKERS:-3} processes)"
    python rl-env_v4/evaluate.py --ckpt "rl-env_v4/runs/$NEW_NAME/best.pt" --days 5 \
        --reactive "$NEW_REACTIVE" --outage-samples 1 --workdir rl-env_v4/run/eval_eum5g \
        --workers "${EVAL_WORKERS:-3}" | tee "rl-env_v4/runs/$NEW_NAME/eval_test.log"
    stamp "done"
    ;;
quick)
    out=rl-env_v4/runs/quick
    TCN_CKPT=${TCN_CKPT:-DGU_SC_Project/checkpoints/quick/seed_0_finetune/best.pt}
    Z_STATS=${Z_STATS:-DGU_SC_Project/checkpoints/quick/z_stats.json}
    for f in "$TCN_CKPT" "$Z_STATS"; do
        [ -f "$f" ] || { echo "$f not found: run the quick TCN and z stats steps first (README)" >&2; exit 1; }
    done
    if [ -e "$out" ]; then
        echo "$out already exists; remove it to run again (rm -r $out)" >&2
        exit 1
    fi
    mkdir -p "$out"
    stamp "quick: reactive thresholds (3 pairs, 1 val day per scenario) -> $out/reactive_tuned.json"
    python rl-env_v4/tune_reactive.py --w-energy "$W_E" --w-qos "$W_Q" --w-overload "$W_O" \
        --ups 0.02,0.03 --downs 0.01,0.02 --val-days 1 --out "$out/reactive_tuned.json" \
        --workdir rl-env_v4/run/tune_quick | tee "$out/reactive_tuned.log"
    stamp "quick: PPO 2 updates of 1 day, TCN $TCN_CKPT -> $out/ppo"
    python rl-env_v4/train_ppo.py --out "$out/ppo" --w-energy "$W_E" --w-qos "$W_Q" \
        --w-overload "$W_O" --reward-scale "$SCALE" --updates 2 --episodes-per-update 1 \
        --val-every 1 --val-days 1 --tcn-ckpt "$TCN_CKPT" --z-stats "$Z_STATS" --no-tensorboard \
        | tee "$out/ppo.log"
    stamp "quick: test evaluation (1 day per scenario)"
    python rl-env_v4/evaluate.py --ckpt "$out/ppo/best.pt" --days 1 --reactive "$out/reactive_tuned.json" \
        --outage-samples 1 --workdir rl-env_v4/run/eval_quick --workers "${EVAL_WORKERS:-3}" \
        | tee "$out/ppo/eval_test.log"
    stamp "quick done"
    ;;
*)
    echo "usage: $0 impact | tune | train [extra train_ppo.py args] | quick"
    exit 1
    ;;
esac

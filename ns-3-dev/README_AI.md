# AI 파트: 데이터셋, TCN 트래픽 예측, RL 안테나 제어 (README_AI)

- 대상: 이 저장소의 `ns-3-dev/` 폴더.
- 작성: NS-3/환경팀, 2026-10-21 (코드 정리 반영). TCN 모델 코드(`DGU_SC_Project/`)는 AI팀 코드를 바탕으로 했다.
- 함께 볼 문서: 시뮬레이션 환경(캠퍼스, 이음5G 무선, 용량, 전력, ISAC, 트래픽)은 [README_env.md](README_env.md)에 있다.
- 참고문헌: 대괄호 번호는 README_env.md 15절과 같은 번호다. 이 문서에 쓴 문헌은 13절에 모두 다시 적었다.
- 결과는 모두 이음5G 환경 기준이다(반송파 4.77 GHz, 실내 침투 손실 16.2 dB). 에이전트와 TCN은 v2 환경에서 학습한 것을 그대로 쓴다. 이음5G 환경에서 다시 학습하지 않은 이유는 README_env.md 11절에 있다(영향이 0.05%p 이하).

---

## 0. 한눈에 보기 (발표용 요약)

```text
ns-3 (1분마다 셀별 관측)
  -> TCN: 셀별 과거 60분 [트래픽, 접속 UE 수, ISAC 밀집도, 시각 sin, cos]
          -> 향후 1~15분 트래픽(중앙값), 15분 피크(상위 분위수), 맥락 임베딩 z(64차원)
  -> RL(PPO, 18셀 공유 정책): 5분마다 셀별 활성 안테나 수 {2, 4, 8, 16} 결정
  -> ns-3 (5분 동안 그 안테나로 커버리지·용량·전력·끊김 계산)
```

- 구조: 제안서대로 O-RAN Non-RT RIC의 rApp 하나가 예측과 제어를 맡는다 [32][44]. TCN은 고정(freeze)하고 정책만 PPO로 학습한다.

| KPI | 목표 | 결과 (이음5G 환경, test) | 판정 |
|---|---|---|---|
| KPI1 예측 오차 | MAPE 10% 이하 | 고부하 구간(Y >= 145 Mbps) 15분 평균 MAPE **9.5%** | 충족 |
| KPI2 전력 절감 | 상시 최대 대비 30% 이상 | 에이전트 **36.4%** | 충족 |
| KPI3 서비스 품질 | 반응형 대비 QoS 부족 50% 이상 감소 | 4.91% -> 2.20%, **55.3% 감소**, 15일 중 14일 에이전트가 낮음 | 충족 |

- 에이전트는 보상 비율(에너지 : QoS : 과부하) 1:10:5로 학습한 것 하나다(r10). TCN도 폭 64 모델 하나다.
- 같은 날끼리 비교하면 반응형 제어보다 끊김률(매분)이 높다. r10 기준 0.87% 대 0.29%이고, 15일 모두 에이전트가 높다(10절 한계).

**발표 핵심 메시지**
1. 데이터셋: ns-3 캠퍼스 시나리오에서 사람(ISAC)이 트래픽보다 먼저 보이도록 만든 93일치 분 단위 데이터(2절).
2. TCN: 60분을 보고 15분을 예측하며, 고부하 구간 MAPE 9.5%로 KPI1을 충족한다. 직전 값 유지(16.8%)보다 정확하다(3절).
3. RL: 예측(ŷ, 피크, z)을 State로 받아 18셀이 하나의 정책을 공유한다. 셀별 보상과 공간 할인으로 이웃에 떠넘긴 부하까지 책임진다(4~5절).
4. 결과: 절감 36.4%(KPI2), 반응형 대비 QoS 부족 55.3% 감소(KPI3). 대신 끊김률은 반응형보다 높다(9~10절).

**PPT 그림 자료**

| 주제 | 파일 | 내용 |
|---|---|---|
| 데이터셋 (일반일과 0일, 2-2절 명령의 `--plot`으로 생성) | `tcn-dataset-v3-5g-31d/plot_representative_cells.png` | 유형별 대표 셀의 접속 UE, ISAC, 트래픽. 사람이 트래픽보다 먼저 늘어난다 |
| | `tcn-dataset-v3-5g-31d/plot_traffic_heatmap.png` | 셀 x 시간 트래픽 히트맵 |
| | `tcn-dataset-v3-5g-31d/plot_all_features.png` | 강의동 대표 셀의 입력 특징 전체 |
| | `tcn-dataset-v3-5g-31d/plot_sample_window.png` | TCN 표본 하나: 입력 60분 + 예측 대상 15분 |
| TCN 학습 | `dashboard/training/tcn_loss_이음5G.png` | 학습·검증 손실 (epoch별) |
| | `dashboard/training/tcn_error_이음5G.png` | 검증 RMSE·MAE와 직전 값 유지 RMSE |
| RL 학습 | `dashboard/training/rl_reward_이음5G.png` | 하루 보상 (학습, 검증, 상시 최대) |
| | `dashboard/training/rl_kpi_이음5G.png` | 절감률(KPI2 목표 30%)과 QoS 부족의 학습 중 변화 |
| | `dashboard/training/rl_stability_이음5G.png` | 정책 엔트로피, 근사 KL, 클리핑 비율, 가치 손실 |
| 예측과 제어 (하루) | `dashboard/<시나리오>/cells_이음5G/all_cells_이음5G.png`, `cell_01_이음5G.png` ~ `cell_18_이음5G.png` | 셀별 요구 트래픽, TCN 예측, 에이전트가 고른 용량, 미처리 구간 |
| 세 기법 비교 (영상) | `dashboard/<시나리오>/dashboard_<시나리오>_이음5G.webm` | 지도 위 셀별 안테나·반경과 기법별 누적 전력·QoS 부족 |

- 데이터셋 그림은 저장소에 없다. 데이터셋을 만들 때 함께 생긴다.
- 대시보드는 축제 시나리오 결과만 저장소에 있다. 다른 시나리오는 `dashboard/README_이음5G.md`의 명령으로 만든다.

---

## 1. 폴더와 파일

| 위치 | 내용 |
|---|---|
| `build_tcn_dataset.py` | ns-3 데이터셋 모드를 돌려 TCN 데이터셋(npz)을 만든다 |
| `run_campaign.py` | 캠퍼스 사이트, 시나리오 인원표, UE 웨이포인트 도우미 (데이터셋·RL 공용) |
| `tcn-dataset-v3-5g-31d/` | TCN 학습 데이터셋의 생성 규칙 README. 데이터 파일(npz, CSV, 그림)은 저장소에 없고 2-2절 명령으로 만든다 |
| `DGU_SC_Project/tcn.py`, `tcn_dual.py` | TCN 기본 블록(`CausalConv1d`, `TemporalBlock`)과 모델 `DualHeadTCN` |
| `DGU_SC_Project/train_dual.py` | TCN 학습·평가 |
| `DGU_SC_Project/run_tcn_v3.sh` | TCN 학습 절차(2단계) + 평가 + KPI1. 기본값이 이 폴더 TCN의 설정이다 |
| `DGU_SC_Project/eval_kpi1.py` | KPI1 계산 |
| `DGU_SC_Project/checkpoints/v3_5g_31d_w64_dual_proposal/seed_0_finetune/best.pt` | **RL에 쓰는 TCN** (폭 64, epoch 87) |
| `rl-env_v4/campus_env.py` | Gym 환경: 날 시드, 시나리오 입력 생성, ns-3 실행 |
| `rl-env_v4/scenario_setup.py` | RL 에피소드의 ns-3 입력 파일(sites, 웨이포인트, rates) 생성 |
| `rl-env_v4/ns3_antenna_env.py` | 관측 해석, RL State, 보상 |
| `rl-env_v4/tcn_forecaster.py` | RL 안에서 TCN 예측 (`DEFAULT_CKPT` = 위 TCN) |
| `rl-env_v4/z_stats.json` | z 표준화 통계 (64차원) |
| `rl-env_v4/compute_z_stats.py` | z 표준화 통계 생성. TCN을 새로 학습했을 때 다시 만든다 |
| `rl-env_v4/policy.py` | 18셀 공유 Actor-Critic |
| `rl-env_v4/train_ppo.py` | PPO 학습 |
| `rl-env_v4/stub_policy.py` | 비교 기준 (상시 최대, 반응형) |
| `rl-env_v4/tune_reactive.py` | 반응형 임계값 조정 |
| `rl-env_v4/evaluate.py` | 평가와 KPI 출력 |
| `rl-env_v4/run_exp_eum5g.sh` | 이음5G 실험 스크립트 (`impact`, `tune`, `train`, 최소 실행 `quick`) |
| `rl-env_v4/runs/ppo_v2_w64_r10/best.pt` | **학습된 에이전트** (보상 비율 1:10:5, 업데이트 760). 학습 기록 `metrics.jsonl`, `tb/`, 로그 `runs/ppo_v2_w64_r10.log` |
| `rl-env_v4/runs/reactive_tuned_v2_r10.json` | 반응형 임계값 (같은 보상 가중치로 조정) |
| `rl-env_v4/runs/eum5g_impact/` | 이음5G 환경 평가 결과 (9절) |
| `dashboard/` | 대시보드 생성 스크립트, 축제 시나리오 그림·영상, 학습 그래프 (`dashboard/README_이음5G.md`) |

---

## 2. TCN 데이터셋

### 2-1. 만드는 방법 (`build_tcn_dataset.py`)

```text
시나리오 인원표 (건물 18개 + 캠퍼스 밖, 9개 시점, 합 1,000명)
  -> 하루 만들기: 건물·시점별 인원 x (1 + N(0, 12%)), 캠퍼스 밖 인원으로 합 1,000
  -> UE 웨이포인트: 시점마다 건물 반경 18 m 안의 고정 위치, 시점 사이 직선 이동
  -> ns-3 데이터셋 모드 (안테나 16 고정, 09:00~21:00의 721개 분)
       접속: 건물 도착 -> 건물 유형별 선행 시차 + UE별 접속 지연 U(0, 6분) 뒤 접속
       수요: 접속 UE가 매분 70% 확률로 발생. 1인당 유형별 수요 x UE 배수 U(0.7, 1.3) / 0.7
       ISAC: 가장 가까운 사이트가 100 m 안의 사람을 검출 확률 0.9로 감지 (접속 여부 무관)
       분당 셀별 CSV: 요구 트래픽, 접속 UE 수, ISAC 감지 수, 재실 인원, 평균 SINR, 핸드오버 in/out
  -> Python 파생 특징: ISAC 밀집도 = 감지 수 x (1 + N(0, 8%)), PRB 사용률 = min(1, D/580),
                       이웃 평균 부하(최근접 3셀), 시각 sin/cos
  -> 슬라이딩 창: 셀·날마다 입력 60분 -> 예측 15분 (647개)
```

- 사람이 먼저 오고 트래픽이 뒤따른다. 접속 전 사람도 ISAC에는 보이므로, ISAC 밀집도가 트래픽보다 앞서는 신호가 된다(엔진 스펙 §5, §6).
- 1인당 수요, 선행 시차, 접속 지연, 활동 확률 같은 생성 값의 근거는 README_env.md 8~9절에 있다 [26][27][40][41][42][54].

### 2-2. 데이터셋 `tcn-dataset-v3-5g-31d`

| 항목 | 값 |
|---|---|
| 생성 | `python build_tcn_dataset.py --scenarios 일반일과 시험기간 축제 --days 31 --plot --outdir tcn-dataset-v3-5g-31d` (로컬 8분 32초). 나머지는 기본값(seed 20250101, 일간 변동 0.12, 접속 지연 평균 3분, 활동 0.7, 수요 폭 0.3, P_D 0.9) |
| 단위 | 셀(18개) x 분(09:00~21:00) x 날. 안테나 16개 고정(상시 최대)에서 관측 |
| 채널 (10개) | 요구 트래픽(Mbps, 예측 대상), 접속 UE 수, ISAC 밀집도, PRB 사용률, 평균 SINR, 핸드오버 in, 핸드오버 out, 이웃 평균 부하, 시각 sin, 시각 cos |
| TCN이 쓰는 채널 | **5개**: 트래픽, 접속 UE 수, ISAC 밀집도, 시각 sin, cos (제안서의 TCN 입력) |
| 창 | 입력 60분 -> 예측 15분. 셀·시나리오·날마다 647개 |
| 크기 | X (1,083,078, 10, 60), Y (1,083,078, 15). 시나리오 3개 x 31일 |
| 나누기 | 학습 0~10일과 15~30일(27일, 943,326개), 검증 11~12일(69,876개), 테스트 13~14일(69,876개) |
| 재현성 | 날 시드 고정. 같은 명령이면 같은 데이터가 나온다. npz sha256 `11c116f4...d126` (체크포인트 config에도 기록). 지금 ns-3 기본값(이음5G 무선)으로 만들면 평균 SINR 열만 달라 sha256은 다르고, TCN이 쓰는 5채널은 같다 |
| 수요 분포 | 셀-분 중앙값 25.8 Mbps, 상위 1% 549 Mbps, 최대 1,287 Mbps. U = D/580 > 0.8인 셀-분 2.00% |

- **이음5G와의 관계**
  - 이음5G 무선 설정(반송파, 침투 손실)으로 다시 만들어도 TCN 입력 5채널은 바뀌지 않는다. 바뀌는 것은 평균 SINR 열뿐이다(README_env.md 11-1절).
  - 그래서 이 데이터셋과 TCN을 그대로 쓴다.
- **바꾸면 재생성이 필요한 것**: 데이터셋을 바꾸는 설정(1인당 수요, 선행 시차, 접속 지연, 활동 확률, 시나리오)을 고치면 데이터셋과 TCN을 다시 만들어야 한다. 이 경우 AI팀에 알린다.

---

## 3. TCN 모델과 학습

| 항목 | 값 | 근거 |
|---|---|---|
| 모델 | `DualHeadTCN`: TemporalBlock 4개(채널 64, 커널 3, 팽창 1/2/4/8, dropout 0.1). 수용 영역 61분 = 입력 60분을 모두 봄 | TCN 구조 [76] |
| 출력 | 중앙 헤드 Linear(64 -> 15): 향후 1~15분 트래픽(중앙값)<br>피크 헤드 Linear(64 -> 1): 향후 15분 최댓값의 상위 분위수<br>z: 마지막 시점의 64차원 은닉 벡터(맥락 임베딩) | 엔진 스펙 §9, 제안서 (ŷ, z) |
| 파라미터 | 88,912개 | |
| 손실 | 이중 핀볼 손실: 중앙 tau = 0.5 + 피크 tau = 0.9 (피크 가중 1.0) | 분위수 회귀 [77], 엔진 스펙 §9 "비대칭 손실, 15스텝 최댓값으로 보수적 자원 산정" |
| 정규화 | 학습 창의 채널별 z-score (입력), 중앙·피크 목표도 각각 z-score. 통계는 체크포인트에 저장 | |
| 학습 | 1단계: Adam lr 1e-3, 60 epoch<br>2단계: lr 1e-4로 이어서 90 epoch까지<br>batch 128, seed 0. 검증 손실이 가장 낮은 epoch(**87**)를 `best.pt`로 저장 | `run_tcn_v3.sh` |
| 학습 시간 | 서버 GPU에서 epoch당 약 60초(두 모델 동시 학습), 약 1시간 45분 | |
| 수렴 | 학습 손실 0.0642, 검증 손실 0.0656 (차이 2.2%) | 체크포인트 |

```bash
# 서버. 기본값이 이 모델의 설정이다(31일 데이터, 폭 64, 학습 0-10,15-30 / 검증 11-12 / 테스트 13-14).
# 출력 폴더가 이미 있으면 멈추므로 새 이름을 준다.
OUT=checkpoints/<새 이름> bash DGU_SC_Project/run_tcn_v3.sh
```

### 3-1. KPI1과 예측 성능 (test 13~14일, 69,876개 창)

- **KPI1 정의** (팀 합의, `eval_kpi1.py`)
  - 중앙 헤드의 15개 시점을 모두 모아, 실제 트래픽 Y >= 145 Mbps인 값만으로 MAPE를 계산한다.
  - 145 Mbps는 C_max의 25%, 즉 안테나 4개의 용량이다 [4].
  - 이 구간은 값 개수로 15.8%, 트래픽 양으로 61.1%를 차지한다.
  - 고부하 구간으로 한정하는 이유
    - MAPE는 실제값으로 나누므로 트래픽이 0에 가까운 구간에서 발산한다 [53].
    - 예측 오차가 안테나 결정에 영향을 주는 곳은 부하가 큰 구간이다.

| 지표 | TCN (폭 64) | 비교: 직전 값 유지(persistence, 같은 구간) |
|---|---|---|
| **KPI1 MAPE** | **9.5%** | 16.8% |
| t+1 / t+15 MAPE | 5.7% / 13.1% | 7.7% / 25.0% |
| 시나리오별 (일반일과 / 시험기간 / 축제) | 10.8 / 9.0 / 8.6% | 20.8 / 16.2 / 13.3% |
| 전체 RMSE / MAE | 18.6 / 10.1 Mbps | RMSE 34.7 Mbps |
| 피크 헤드가 실제 15분 최댓값 이상인 비율 | 0.893 (목표 분위수 0.9) | |

- 직전 값 유지 16.8%는 `eval_kpi1.py`가 KPI1과 같은 구간(Y >= 145 Mbps)에서 출력하는 값이다. 체크포인트 폴더의 `kpi1.log`는 이 출력을 바꾸기 전 기록이라 더 낮은 문턱(Y >= 72.5 Mbps)의 20.9%가 적혀 있다.

**모델 폭 비교** (같은 데이터와 절차, test 13~14일)

| 폭 | 파라미터 | KPI1 | t+15 | RMSE | 피크 덮는 비율 | 학습·검증 손실 차이 |
|---|---|---|---|---|---|---|
| 32 | 22,960 | 10.4% | 15.0% | 20.5 | 0.906 | -0.7% |
| **64 (사용)** | **88,912** | **9.5%** | **13.1%** | **18.6** | **0.893** | **+2.2%** |
| 128 | 349,840 | 9.3% | 12.6% | 18.1 | 0.892 | +6.5% (과적합 시작) |

- 데이터를 15일에서 31일로 늘렸을 때(폭 32: 10.7 -> 10.4%)보다 모델 폭을 늘렸을 때 더 좋아졌다. 그래서 모델 용량이 주 제약이었다.
- 폭 64를 쓴 이유: 목표를 달성하면서 과적합이 작다. z 차원이 64라 RL State도 89차원으로 적당하다.
- 폭 32와 128 모델은 이 폴더에 없다(v2 폴더에 폭 32 모델이 있다).

### 3-2. 알려진 약점 (2026-10-14~15 분석, test 13~14일)

- **급증 구간**(창의 6.8%)
  - 피크를 덮는 비율이 68.6%로, 전체(89.3%)보다 낮다.
  - 피크 과소예측은 평균 8.2 Mbps, 상위 10% 28.7 Mbps다(persistence 오차의 약 1/3).
- **건물별**: 이벤트 건물 MAPE 14.3%, 강의동 피크 덮는 비율 87.4%.
- **ISAC 의존도**
  - ISAC 채널을 창끼리 섞으면 KPI1이 9.5%에서 47.9%로 나빠진다. TCN이 ISAC 정보를 크게 쓴다는 뜻이다.
  - 다만 섞기는 수준 정보까지 깨므로, 선행 효과만의 크기는 ISAC 없이 다시 학습해야 분리된다.
- **단일 seed**: 신뢰구간은 산출하지 않았다.

---

## 4. RL 환경 (MDP)

| 항목 | 값 |
|---|---|
| 에피소드 | 하루(09:00~21:00) = 5분 step 144개. 시나리오와 날 시드가 정한 하루 |
| 행동 | 셀별 {0, 1, 2, 3} -> 안테나 {2, 4, 8, 16}, `MultiDiscrete([4] x 18)`. 5분 동안 유지 |
| 관측 시점 | 5분 step의 마지막 분 (ns-3는 매분 관측을 내고, Python이 5분 동안 행동을 유지한다) |
| 날 시드 | 학습: 시나리오 무작위 + 시드 0~99,999 무작위<br>검증: 시나리오당 2일(100,000~100,001), 6일<br>테스트: 시나리오당 5일(200,000~200,004), 15일<br>ns-3 RngRun = 시드 + 1 |
| ISAC 오차 | 날마다 (시나리오, 날)로 난수를 다시 정한다. 그래서 같은 날은 실행 순서와 무관하게 같다 |

### 4-1. State (셀당 89차원)

| 항목 | 차원 | 정규화 |
|---|---|---|
| TCN 예측 ŷ (향후 1~15분) | 15 | / C_max |
| TCN 피크 | 1 | / C_max |
| 현재 부하 (요구 트래픽) | 1 | / C_max |
| 현재 안테나 단계 | 1 | 0~1 (단계 / 3) |
| 시각 sin, cos | 2 | |
| 평균 SINR | 1 | / 20 dB |
| 이웃 3셀: 피크 평균, 안테나 단계 평균, 부하 평균 | 3 | 위와 같음 |
| 이웃 3셀의 최대 PRB 사용률 | 1 | 0~1 (과부하 벌점이 이웃 사용률로 정해지므로 정책이 직접 보게 함) |
| TCN 맥락 z | 64 | `z_stats.json`(TCN 학습 구간 943,326개 창)의 차원별 평균·표준편차로 표준화, +-5에서 자름(잘리는 값 0.67%) |
| **합계** | **89** | 제안서의 State = [ŷ, z, 현재 안테나 단계, SINR, PRB 사용률, 이웃 부하] |

### 4-2. 보상

셀 c의 5분 보상은 다음과 같다(N = 18).

```text
r_c = -( w_E * E_c  +  w_Q * S_c / D_bar  +  w_O * O_c ) / N
```

| 항 | 정의 |
|---|---|
| E_c | 셀 c의 5분 평균 전력 P(A) = 0.3 + 0.7 x A/16 (안테나 16 = 1) [10] |
| S_c | 셀 c의 5분 부족분 합 = Σ분 [ max(0, 처리 수요 - 용량) + 끊긴 UE 수요 ] |
| D_bar | 캠퍼스 전체 요구 트래픽 합 / N |
| O_c | max(0, U_c - 0.8) / 0.2의 5분 평균 [15][16][17][25] |

- 18셀의 r_c를 더하면 캠퍼스 전체 보상과 정확히 같다. 기록되는 하루 보상은 셀 평균의 하루 합이다.
- **가중치**: 세 값은 0~1 사이이고 합이 1이다.
- **학습 신호**: 학습할 때만 보상에 `--reward-scale`(정수 비율의 합)을 곱한다. 그래서 학습은 정수 가중치(예: 1:10:5)와 같다. 로그와 평가의 보상은 0~1 가중치 기준이라, 비율이 다르면 보상끼리 비교할 수 없다.

| 이름 | 비율 E:Q:O | w_E | w_Q | w_O | reward-scale |
|---|---|---|---|---|---|
| **r10** | **1:10:5** | **0.0625** | **0.625** | **0.3125** | **16** |

- `RewardConfig`, `train_ppo.py`, `tune_reactive.py`의 기본값이 r10 설정이다.

(v2에서 보상 비율 1:5:5와 1:20:5도 함께 학습해 비교했다. 결과는 9절 끝의 참고에 있고, 파일은 이 폴더에서 정리했다.)

- **책임 귀속**
  - 반경 밖 끊김은 가장 가까운(담당) 셀의 책임이다(ns-3에서 항상 적용). 그래서 안테나를 줄인 셀이 자기가 만든 끊김의 벌점을 받는다.
  - 과부하는 사용률을 넘긴 셀 자신의 책임이다.
- **공간 할인**
  - 학습 신호는 r_c + α Σ(이웃 3셀 r_j), α = 1이다 [51].
  - 그래서 부하를 떠넘긴 셀도 이웃의 과부하·끊김 벌점을 함께 받는다.
- **주의**
  - 제안서의 보상 문구는 "QoS가 목표에 미달할 때만 페널티"다.
  - 이 구현은 모든 부족분에 벌점을 준다. 보고서에서는 구현대로 설명한다.

---

## 5. 정책과 PPO 학습

| 항목 | 값 |
|---|---|
| 정책 신경망 | 18셀이 같은 가중치를 쓰는 Actor-Critic<br>인코더: MLP 89 -> 128 -> 128 (tanh)<br>캠퍼스 맥락: 18셀 인코딩의 평균 g<br>actor: [h_c, g] -> 128 -> 4<br>critic: [h_c, g] -> 128 -> 1 |
| 파라미터 | 94,469개 |
| 초기화 | actor 마지막 층 bias로 안테나 16 확률을 약 0.6으로 시작. 모두 켠 상태에서 출발해 줄여도 되는 셀을 찾아가게 함 |
| PPO | 셀별 GAE(γ 0.99, λ 0.95) [78], 셀별 확률 비율, clip 0.2, epoch 10, 미니배치 128 step, Adam lr 3e-4(eps 1e-5)를 선형으로 줄임, 엔트로피 0.01(셀 평균), 가치 0.5, 기울기 clip 0.5 [52] |
| 학습량 | 업데이트 800회 x 하루 4개(버퍼 576 step) = 3,200일 |
| 모델 선택 | 20회마다 검증 6일을 결정적 행동으로 평가하고, 하루 보상이 가장 높은 정책을 `best.pt`로 저장 |
| 최적 업데이트 | 760 (800회 중. 끝 무렵까지 개선 중 -> 더 길게 학습할 여지) |
| 학습 시간 | 서버에서 업데이트당 약 33~35초, 약 7.8시간 (보상 비율 3개를 동시에 학습했을 때) |
| TCN | 고정(학습하지 않음). 5분마다 셀별 60분 창으로 예측 |

**z의 효과**: 세 번 비교했다(5G 이전 w10 +0.122, w20 +0.007, 5G w10 -0.018, 보상 차이). 결과가 일관되지 않아 "z가 성능을 높인다"는 주장은 하지 않는다. 제안서 구조대로 z는 State에 넣는다. 선형 프로브로 보면 z는 "접속 대기 인원"(ISAC / 0.9 - UE 수, R² 0.06 -> 0.74)을 담지만, 미래 수요에 대해서는 ŷ보다 더 주는 정보가 거의 없다.

---

## 6. 비교 기준 (baseline)

| 기준 | 규칙 |
|---|---|
| 상시 최대 | 모든 셀 안테나 16 |
| 반응형 | 5분마다 셀의 현재 PRB 사용률이 up보다 크면 한 단계 올리고, down보다 작으면 한 단계 내린다(제안서 비교군 "현재 부하 기반 반응형 제어"). 미래를 보지 않아 부하가 오른 뒤에야 반응한다 |

- **반응형 임계값 조정**
  - 에이전트와 같은 보상 가중치로, 검증 6일에서 하루 보상이 가장 좋은 (up, down)을 고른다(`tune_reactive.py`).
  - 격자는 up {0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2} x down {0.001, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.05, 0.1} 가운데 up > down인 66개다.
  - 약 55분이 걸린다.
- **결과**: up 0.03 / down 0.02 (격자 안쪽 값). `runs/reactive_tuned_v2_r10.json`에 66개 조합의 검증 결과와 함께 있다.
- **용어**: 상시 최대와 반응형은 "비교 기준", 학습된 PPO는 "에이전트"라 부른다. 셋을 함께 말할 때는 "기법"이라 한다.

---

## 7. 평가와 지표

| 지표 | 정의 |
|---|---|
| 절감률 (KPI2) | 1 - 하루 평균 상대 전력 (상시 최대 = 0%) |
| QoS 부족 (KPI3 지표) | Σ [용량 초과 수요 + 끊긴 UE 수요] / Σ 요구 트래픽. 처리하지 못한 트래픽의 비율 |
| KPI3 | 반응형 대비 QoS 부족 감소율 = 1 - 에이전트 / 반응형 (test 평균) |
| 끊김률 | 끊긴 UE-분 / 접속 UE-분 (매분 판정). 참고 지표 |
| 과부하 | 사용률 U > 0.8인 셀-분 / 전체 셀-분 |
| 보상 | 4-2절 식의 하루 합 (비율이 같을 때만 비교) |

- **평가 방식**
  - test 15일 x 기법 3개 = 45개 하루를 `evaluate.py --workers 8`로 병렬 실행한다(서버 약 85초).
  - 신경망은 CPU 1스레드로 돌린다. 그래서 프로세스 수나 순서, 컴퓨터가 달라도 같은 결과가 나온다.
- **KPI3 변경 (2026-10-15)**
  - 제안서 원안은 "서비스 끊김률을 셀 ON/OFF 대비 90% 이상 감소"였다.
  - 이 캠퍼스는 사이트 간격(67 m)에 비해 최대 반경(85 m)이 커서 셀끼리 커버리지가 많이 겹친다. 그래서 셀 ON/OFF 비교 결과가 가정(깨어나는 조건, 재기동 시간, 수면 전력)에 크게 좌우됐다.
  - 그래서 반응형 대비 QoS 부족으로 바꿨다.
- **이음5G 평가**
  - 끊김은 매분 판정(`--outage-samples 1`)으로 본다(test 15일).
  - 1초 판정(`--outage-samples 60`)은 정확하지만 약 10배 느려서, 시나리오별 test 첫날(3일)만 따로 평가해 9절에 참고로 둔다(README_env.md 11-3절).

---

## 8. 실행 방법

```bash
conda activate smart_campus_5G
cd ns-3-dev   # 이 저장소의 ns-3-dev 폴더

# 데이터셋 생성 (로컬 약 8분 반, 2-2절)
python build_tcn_dataset.py --scenarios 일반일과 시험기간 축제 --days 31 --plot --outdir tcn-dataset-v3-5g-31d

# KPI1
cd DGU_SC_Project && python eval_kpi1.py --npz ../tcn-dataset-v3-5g-31d/tcn_dataset.npz \
    --ckpt checkpoints/v3_5g_31d_w64_dual_proposal/seed_0_finetune/best.pt --cmax 580 && cd ..

# z 표준화 통계 (TCN을 새로 학습했을 때만. 제공된 TCN이면 z_stats.json과 같은 값)
python rl-env_v4/compute_z_stats.py

# 평가 (r10 에이전트 + 비교 기준, test 15일, 매분 판정)
python rl-env_v4/evaluate.py --ckpt rl-env_v4/runs/ppo_v2_w64_r10/best.pt --days 5 \
    --reactive rl-env_v4/runs/reactive_tuned_v2_r10.json --outage-samples 1 \
    --workdir rl-env_v4/run/eval_r10 --workers 8

# 1초 판정 평가 (시나리오별 test 첫날, 프로세스 3개로 약 15~16분)
python rl-env_v4/evaluate.py --ckpt rl-env_v4/runs/ppo_v2_w64_r10/best.pt --days 1 \
    --reactive rl-env_v4/runs/reactive_tuned_v2_r10.json --outage-samples 60 \
    --workdir rl-env_v4/run/eval_sec --workers 3

# 이음5G 영향 평가 (에이전트 x 이음5G·v2 무선 설정, 서버 약 3분)
EVAL_WORKERS=8 bash rl-env_v4/run_exp_eum5g.sh impact

# 이음5G 환경에서 다시 학습 (서버, tmux 세션마다 하나씩, 약 8시간)
bash rl-env_v4/run_exp_eum5g.sh tune          # 반응형 임계값 조정
bash rl-env_v4/run_exp_eum5g.sh train         # 학습 -> 조정 파일 대기 -> test 평가
```

- TCN 학습 명령은 3절에 있다. 대시보드 원자료와 그림·영상을 만드는 명령은 `dashboard/README_이음5G.md`에 있다.
- 데이터 3일 -> TCN 2+1 epoch -> z 통계 -> RL 2회 업데이트로 전체 흐름만 몇 분 안에 확인하는 최소 실행 명령은 저장소 최상위 `README.md`의 "빠른 확인" 절에 있다.
- 새로 학습한 TCN으로 RL을 할 때는 `train_ppo.py --tcn-ckpt <best.pt> --z-stats <compute_z_stats.py 출력>`을 준다. 두 경로는 에이전트 체크포인트에 남고, `evaluate.py`가 같은 TCN으로 평가한다.
- 같은 버전의 평가나 조정을 동시에 돌릴 때는 `--workdir`를 서로 다르게 준다. ns3-ai 공유 메모리 이름은 프로세스마다 달라서 한 컴퓨터에서 여러 실험을 함께 돌려도 된다.
- 긴 작업은 서버 tmux에서 돌린다. C++를 고친 뒤 서버에서 `./ns3 build cttc-nr-demo`를 할 때는, 같은 폴더에서 학습이 돌고 있지 않은지 먼저 확인한다(실행 파일 교체가 에피소드 시작과 겹치면 학습이 죽는다).

---

## 9. 결과 (이음5G 환경, test 15일, 매분 판정)

결과 파일: `rl-env_v4/runs/eum5g_impact/eval_test_v2agent_r10_eum5g.json` (같은 날의 v2 무선 설정 결과는 `..._v2radio.json`)

| 기법 | 구간 | 절감률 | QoS 부족 | 끊김률 | 과부하 |
|---|---|---|---|---|---|
| 상시 최대 | 전체 | 0.0% | 1.40% | 0.00% | 2.11% |
| 반응형 (0.03/0.02) | 전체 | 29.2% | 4.91% | 0.29% | 2.73% |
| **에이전트** | **전체** | **36.4%** | **2.20%** | 0.87% | **1.99%** |
| 상시 최대 / 반응형 / 에이전트 | 일반일과 | 0.0 / 26.3 / 34.2% | 0.53 / 1.82 / 1.51% | 0.00 / 0.37 / 1.26% | 2.13 / 2.60 / 1.65% |
| 상시 최대 / 반응형 / 에이전트 | 시험기간 | 0.0 / 27.3 / 35.7% | 3.37 / 8.04 / 4.34% | 0.00 / 0.29 / 0.92% | 2.79 / 3.62 / 2.75% |
| 상시 최대 / 반응형 / 에이전트 | 축제 | 0.0 / 33.9 / 39.3% | 0.31 / 4.86 / 0.73% | 0.00 / 0.22 / 0.44% | 1.42 / 1.96 / 1.56% |

- 같은 날 15쌍으로 반응형과 비교하면, 에이전트는 다음과 같다.
  - 보상: 15/15일 높음
  - 절감: 15/15일 높음
  - QoS 부족: 14/15일 낮음
  - 끊김률: 15/15일 높음
- 시나리오별 KPI3 감소율은 일반일과 17.0%, 시험기간 46.1%, 축제 84.9%다.
- **v2 무선 설정과의 차이**: 같은 15일에서 절감 -0.03%p, QoS 부족 -0.02%p, 끊김 +0.004%p다(README_env.md 11-2절).

**참고: 1초 판정 (이음5G, 시나리오별 test 첫날 3일)**

| 기법 | 절감률 | QoS 부족 | 끊김 시간 비율 (1초) | 하루 끊김 횟수 | 그중 10초 이상 |
|---|---|---|---|---|---|
| 상시 최대 | 0.0% | 1.51% | 0.00% | 0 | 0 |
| 반응형 | 28.8% | 5.61% | 0.29% | 303 | 290 |
| **에이전트** | **36.6%** | **2.39%** | 0.96% | 773 | 753 |

- 대시보드와 같은 날(날 시드 200000)이다. 3일 합계 기준이고, 하루 끊김 횟수는 3일 평균이다.
- 에이전트의 하루 끊김 횟수는 일반일과 1,147회, 시험기간 862회, 축제 309회다(반응형 381, 362, 167회).
- 끊김의 97%가 10초 이상 이어진다. 1초 판정의 끊김 시간 비율은 매분 판정 끊김률과 거의 같다(에이전트 0.965% 대 0.963%).
- v2 환경 test 15일의 1초 판정은 에이전트 0.87%, 하루 726회(10초 이상 709회), 반응형 0.28%, 292회였다.

**참고: 다른 보상 비율**
- v2에서 보상 비율 1:5:5와 1:20:5로도 학습해, 이음5G 환경에서 함께 평가했다(test 15일, 각자의 반응형 임계값으로 비교).
  - 1:5:5: 절감 41.3%, QoS 부족 2.68%, 끊김 1.34%, KPI3 45.5%(미달)
  - 1:20:5: 절감 30.8%, QoS 부족 1.82%, 끊김 0.49%, KPI3 61.3%
- QoS 가중치가 클수록 절감은 줄고 QoS 부족과 끊김도 줄어든다. 가중치가 트레이드오프의 위치를 정한다.
- 이 두 에이전트의 파일은 이 저장소에 넣지 않았다(v2 작업 폴더에 있다).

---

## 10. 한계와 남은 일

1. **끊김**: 에이전트는 반응형보다 매일 끊김률이 높다. 반응형을 에이전트와 같은 절감 수준(약 36%)으로 맞추면 끊김이 비슷해지므로(v2 분석: 0.86% 대 0.84%), 차이는 주로 절감 수준 차이다.
2. **KPI3**: 평균으로는 충족하지만, 일반일과의 감소율은 17%로 작다.
3. **학습 환경**: 에이전트와 반응형 임계값은 v2 환경(3.5 GHz)에서 정했다. 이음5G에서 다시 학습하면 학습 환경까지 맞출 수 있다(약 8시간). 영향 평가로 보면 결과는 거의 같을 것이다.
4. **학습량**: 최적 업데이트가 760(800회 중)으로 끝 무렵이었다. 더 길게 학습할 여지가 있다.
5. **단일 seed**: TCN과 PPO 모두 seed 0 하나로 학습했다. 신뢰구간은 없다.
6. **z의 효과**: 일관되지 않다(5절).
7. **비교 기준 범위**: 예측 기반 규칙 비교 기준과 선행 유입(S2) 시나리오는 없다.
8. **보상 문구**: 제안서 문구("QoS 미달 시에만 페널티")와 구현(모든 부족분에 벌점)이 다르다.

---

## 11. 참고문헌 대응

| 설정 | 참고문헌 |
|---|---|
| 데이터셋: 수요 수준, 건물 유형 패턴, 선행 시차, 접속 지연, 활동 확률 | [54][55][40][41][42][30][7][8] |
| 데이터셋: ISAC 센싱 반경, 검출 확률, 측정 오차 | [26][27][28][37][38] |
| C_max 580 Mbps (KPI1 문턱 145 Mbps = 0.25 x C_max, State 정규화) | [4] |
| TCN 구조 (팽창 인과 합성곱, 잔차 블록) | [76] |
| 분위수(핀볼) 손실 | [77] |
| 입력 창·예측 지평·관측 해상도 | [43][29] |
| KPI1을 고부하 구간 MAPE로 한정 | [53] |
| 5분 제어 주기, O-RAN rApp 시간 척도 | [44][45][46][47][32] |
| 예측 + RL 기지국 에너지 절감 선행 연구 | [46][47][48][49] |
| 시스템 수준 추상화, 반경 밖 = 끊김 관행 | [48][49][50] |
| 공간 할인 | [51] |
| PPO, GAE | [52][78] |
| 과부하 기준 0.8 (보상 O_c) | [15][16][17][25] |
| 전력 모델 (보상 E_c) | [10] |

---

## 12. 용어

| 용어 | 뜻 |
|---|---|
| 비교 기준 | 상시 최대, 반응형 (학습하지 않는 규칙) |
| 에이전트 | 학습된 PPO 정책 |
| 기법 | 비교 기준과 에이전트를 함께 부를 때 |
| r10 | 보상 비율 1:10:5 (에너지 : QoS : 과부하). 파일 이름의 `ppo_v2_w64_r10` = v2 환경에서 학습, TCN 폭 64, 비율 1:10:5 |
| QoS 부족 | 처리하지 못한 트래픽의 비율 (용량 초과 + 끊긴 UE 수요) |
| ISAC 밀집도 | 기지국 센싱으로 감지한 사람 수(측정 오차 포함). 접속 전 사람도 포함한다 |

---

## 13. 참고문헌

번호는 README_env.md 15절과 같다. 이 문서에 쓴 문헌만 적었다. "(2차)"는 2차 출처이거나 원문을 열어 보지 못한 문헌이다.

4. ITU-R Report M.2410-0 (2017), *Minimum requirements related to technical performance for IMT-2020 radio interface(s)*. https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-M.2410-2017-PDF-E.pdf  
   근거: C_max의 5G 요구 하향 효율 7.8 b/s/Hz
7. 3GPP TR 36.814 V9.0.0, *Further advancements for E-UTRA physical layer aspects* (Table A.2.1.1.2-3 pico 전력, A.2.1.3.1 FTP 트래픽 모델). https://www.3gpp.org/ftp/Specs/archive/36_series/36.814/36814-900.zip  
   근거: 피코셀 총 송신 전력 30 dBm, FTP 트래픽 모델
8. 3GPP TR 38.913 V19.0.0, *Study on scenarios and requirements for next generation access technologies* (§6.1 시나리오, Table 6.1.5-1). https://www.etsi.org/deliver/etsi_tr/138900_138999/138913/19.00.00_60/tr_138913v190000p.pdf  
   근거: 실내 핫스팟 간격, 활동 비율 가정 비교
10. 3GPP TR 38.864 V18.1.0 (2023), *Study on network energy savings for NR* (§5.1 전력 모델, §6.3.1.2 공간 영역 적응 평가). https://www.3gpp.org/ftp/Specs/archive/38_series/38.864/38864-i10.zip  
   근거: 안테나당 출력 일정·커버리지 4.5~6.5 dB 감소, 전력 모델 s = 0.3
15. A. Albanna, H. Yousefi'zadeh, "Congestion Minimization of LTE Networks: A Deep Learning Approach", *IEEE/ACM Trans. Networking* 28(1), 2020. https://newport.eecs.uci.edu/~hyousefi/publ/cmlteTON.pdf  
   근거: 과부하 기준 U_th 80%
16. E. Pang, "AI and network capacity planning decisions", Ericsson Blog, 2020-12-18 (아카이브). https://web.archive.org/web/2024id_/https://www.ericsson.com/en/blog/2020/12/ai-and-network-capacity-planning-decisions  
   근거: 과부하 기준 U_th 80%
17. Nokia, *Sustainable traffic growth in LTE network*, white paper, 2017. https://onestore.nokia.com/asset/f/200996/  
   근거: 과부하 기준 U_th 80%, 대안 90%
25. F. González-Horta et al., "Mathematical model for the knee of the utilization-response time curve" (2011). https://arxiv.org/abs/1106.2380  
   근거: 사용률-지연 곡선의 무릎 70~90%
26. 3GPP TS 22.137 V19.1.0, *Service requirements for Integrated Sensing and Communication* (Table 6.2-1). https://www.etsi.org/deliver/etsi_ts/122100_122199/122137/19.01.00_60/ts_122137v190100p.pdf  
   근거: ISAC 검출 확률 P_D, 오경보
27. 3GPP TR 22.837 V19.4.0, *Feasibility Study on Integrated Sensing and Communication* (§5.14, PR 5.19.6-1). https://www.3gpp.org/ftp/Specs/archive/22_series/22.837/22837-j40.zip  
   근거: ISAC 센싱 반경 100 m
28. ETSI GR ISC 001 V1.1.1 (2025), *Integrated Sensing And Communication; Use Cases and Deployment Scenarios* (§5.14). https://www.etsi.org/deliver/etsi_gr/isc/001_099/001/01.01.01_60/gr_isc001v010101p.pdf  
   근거: ISAC 센싱 반경 100 m
29. 3GPP TS 32.401 V19.0.0, *Performance Management (PM); Concept and requirements* (§5.4.1.4 수집 입도). https://www.etsi.org/deliver/etsi_ts/132400_132499/132401/19.00.00_60/ts_132401v190000p.pdf  
   근거: 관측 해상도, 예측 지평 15분
30. 3GPP TS 38.331 V19.4.0, *NR; RRC protocol specification* (T-Reselection 0..7 s). https://www.etsi.org/deliver/etsi_ts/138300_138399/138331/19.04.00_60/ts_138331v190400p.pdf  
   근거: 접속 지연의 해석
32. O-RAN Alliance SuFG, *Potential Energy Savings Features in O-RAN*, white paper, Jan 2025. https://mediastorage.o-ran.org/white-papers/O-RAN.SuFG.Potential%20Energy%20Savings%20Features%20in%20O-RAN%20white%20paper%202025-01.pdf  
   근거: O-RAN rApp 에너지 절감 구조
37. Custance, Khan, Parkinson, Sheng, "Parameter-Efficient Domain Adaption for CSI Crowd-Counting" (2026). https://arxiv.org/pdf/2601.02203  
   근거: ISAC 사람 수 세기 정확도
38. "Modeling Classroom Occupancy using Data of WiFi Infrastructure in a University Campus" (2021). https://arxiv.org/pdf/2104.10667  
   근거: ISAC 오차 비교: 교실 재실 추정
40. D. Kotz, K. Essien, "Characterizing Usage of a Campus-wide Wireless Network", Dartmouth TR2002-432. https://cs.dartmouth.edu/dfk/research/kotz-campus-tr/kotz-campus-tr.pdf  
   근거: 건물 유형별 트래픽 패턴, 기숙사 저녁 피크
41. H. Wang et al., "Understanding Mobile Traffic Patterns of Large Scale Cellular Towers in Urban Environment", *ACM IMC 2015*. https://arxiv.org/pdf/1510.04026  
   근거: 주거 지역 야간 피크, 사용자별 수요 긴 꼬리
42. T. Favale et al., "Campus Traffic and e-Learning during COVID-19 Pandemic", *Computer Networks* 2020. https://arxiv.org/pdf/2004.13569  
   근거: 캠퍼스 트래픽이 강의 시간표를 따름
43. C. Zhang, P. Patras, "Long-Term Mobile Traffic Forecasting Using Deep Spatio-Temporal Neural Networks", *ACM MobiHoc 2018*. https://arxiv.org/pdf/1712.08083  
   근거: 관측 해상도·입력 창·예측 지평
44. M. Polese et al., "Understanding O-RAN: Architecture, Interfaces, Algorithms, Security, and Research Challenges", *IEEE COMST* 2023. https://arxiv.org/html/2202.01032  
   근거: O-RAN Non-RT RIC 시간 척도
45. L. Kundu, X. Lin, R. Gadiyar (NVIDIA), "Towards Energy Efficient RAN: From Industry Standards to Trending Practice" (2024). https://arxiv.org/pdf/2402.11993  
   근거: 셀·반송파 조절의 시간 척도
46. Q. Wu et al., "Deep Reinforcement Learning with Spatio-temporal Traffic Forecasting for Data-Driven Base Station Sleep Control", *IEEE/ACM ToN* 2021. https://arxiv.org/pdf/2101.08391  
   근거: 트래픽 예측 + RL 기지국 제어, 제어 주기
47. M. Ozturk et al., "Energy Optimization in Ultra-Dense Radio Access Networks via Traffic-Aware Cell Switching", *IEEE TGCN* 2021. https://arxiv.org/pdf/2007.04133  
   근거: 트래픽 기반 셀 전환, 제어 주기
48. T. Cai, Q. Wang, S. Zhang, Ö. T. Demir, C. Cavdar, "Multi-agent Reinforcement Learning for Energy Saving in Multi-Cell Massive MIMO Systems", *IEEE ICMLCN 2024*. https://arxiv.org/abs/2402.03204  
   근거: 시스템 수준 추상화, 끊김 집계 관행
49. M. Bordin et al., "Design and Evaluation of Deep Reinforcement Learning for Energy Saving in Open RAN", *IEEE CCNC 2025*. https://arxiv.org/pdf/2410.14021  
   근거: 시스템 수준 추상화, 끊김 집계 관행
50. J. Ye, Y.-J. A. Zhang, "DRAG: Deep Reinforcement Learning Based Base Station Activation in Heterogeneous Networks" (2018). https://arxiv.org/pdf/1809.02159  
   근거: 소형셀 반경 고정 관행
51. T. Chu, J. Wang, L. Codecà, Z. Li, "Multi-Agent Deep Reinforcement Learning for Large-Scale Traffic Signal Control", *IEEE T-ITS* 2020 (공간 할인). https://arxiv.org/abs/1903.04527 (2차: 이번 조사에서 원문 미열람)  
   근거: 셀별 보상의 공간 할인
52. J. Schulman et al., "Proximal Policy Optimization Algorithms" (2017). https://arxiv.org/abs/1707.06347 (2차)  
   근거: PPO 학습 알고리즘
53. R. J. Hyndman, A. B. Koehler, "Another look at measures of forecast accuracy", *Int. J. Forecasting* 22(4), 2006 (2차)  
   근거: KPI1 MAPE를 고부하 구간으로 한정한 근거
54. 과학기술정보통신부, 무선데이터 트래픽 통계 (2024-05, 5G 1인당 월 29,114 MB), 전자신문 통계 경유 (2차). https://m.etnews.com/stats/detail_stats.html?id=51209  
   근거: 1인당 수요 수준 비교
55. Netflix, 인터넷 연결 속도 권장 사항. https://help.netflix.com/en/node/306  
   근거: 영상 비트레이트 비교
76. S. Bai, J. Z. Kolter, V. Koltun, "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling," arXiv:1803.01271, 2018. https://arxiv.org/abs/1803.01271 (2차)  
   근거: TCN 구조: 팽창 인과 합성곱과 잔차 블록
77. R. Koenker, G. Bassett, "Regression Quantiles," *Econometrica* 46(1), 1978. (2차)  
   근거: 분위수(핀볼) 손실, 중앙값 tau 0.5와 피크 tau 0.9
78. J. Schulman, P. Moritz, S. Levine, M. Jordan, P. Abbeel, "High-Dimensional Continuous Control Using Generalized Advantage Estimation," *ICLR* 2016. https://arxiv.org/abs/1506.02438 (2차)  
   근거: GAE lambda 0.95

**팀 내부 문서**
- 최종 공통 엔진 스펙 (대회 제공 자료, 이 저장소에 없음): §5 TCN 특징, §6 건물 유형 계수, §8 스티어링·U_th, §9 예측
- INCharge 제안서 (이 저장소에 없음): TCN 입력·출력, RL State/Action/Reward, rApp 구조, KPI 1~3, 비교군

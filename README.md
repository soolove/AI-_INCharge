# 이음5G 스마트 캠퍼스: ISAC 밀집도 예측 기반 안테나 제어

캠퍼스 이음5G 망(4.77 GHz, 100 MHz)에서 18개 셀이 5분마다 활성 안테나 수 {2, 4, 8, 16}을 정해 기지국 전력을 줄이는 시뮬레이션이다. ns-3로 캠퍼스를 1분 단위로 계산하고, TCN이 셀별 트래픽을 15분 앞까지 예측하며, PPO 에이전트가 그 예측을 보고 안테나 수를 고른다.

```text
Scenario/<시나리오>.csv (건물 18개의 시각별 인원표)
  -> ns-3 scratch/cttc-nr-demo.cc (1분 단위 시스템 수준 시뮬레이션, 이음5G 무선)
       데이터셋 모드: 안테나 16 고정, 셀별 분 단위 관측            -> 3단계 데이터 생성
       RL 모드 (ns3-ai): 매분 관측 전달, 5분마다 안테나 결정 받음  -> 6단계 RL
  -> TCN: 과거 60분 [트래픽, 접속 UE 수, ISAC 밀집도, 시각 sin/cos]
          -> 향후 1~15분 트래픽, 15분 피크, 맥락 임베딩 z(64차원)     -> 4단계 TCN 학습
  -> PPO (18셀 공유 정책): [예측, z, 부하, SINR, 이웃 상태] -> 셀별 안테나 수
```

**결과** (이음5G 환경, test 15일, 제공된 TCN과 에이전트)

| 지표 | 목표 | 결과 |
|---|---|---|
| KPI1 예측 오차 | 고부하 구간(Y >= 145 Mbps) MAPE 10% 이하 | 9.5% |
| KPI2 전력 절감 | 상시 최대 대비 30% 이상 | 36.4% |
| KPI3 서비스 품질 | 반응형 제어 대비 QoS 부족 50% 이상 감소 | 4.91% -> 2.20% (55.3% 감소) |

- 끊김률은 반응형 제어보다 높다(0.87% 대 0.29%). 한계와 해석은 [README_AI.md](ns-3-dev/README_AI.md) 9\~10절에 있다.
- 환경의 설정값과 근거는 [README_env.md](ns-3-dev/README_env.md), 데이터셋·TCN·RL의 설계와 결과는 [README_AI.md](ns-3-dev/README_AI.md)에 자세히 있다.

---

## 저장소 구성

```text
.
├── ns-3-dev/                      ns-3 (아래 "외부 코드 버전") + 이 프로젝트 코드
│   ├── scratch/cttc-nr-demo.cc    캠퍼스 시나리오 (데이터셋 모드, RL 모드)
│   ├── run_campaign.py            사이트 표, 시나리오 인원표 읽기, UE 이동 경로, 지도 자르기
│   ├── build_tcn_dataset.py       데이터셋 생성 (3단계)
│   ├── tcn-dataset-v3-5g-31d/     데이터셋 설명 README (데이터 파일은 3단계에서 만든다)
│   ├── DGU_SC_Project/            TCN 모델, 학습, KPI1 (4단계)
│   │   └── checkpoints/v3_5g_31d_w64_dual_proposal/   제공된 TCN (폭 64)과 학습 기록
│   ├── rl-env_v4/                 RL 환경, PPO 학습, 비교 기준, 평가 (5~6단계)
│   │   ├── z_stats.json           제공된 TCN의 z 표준화 통계
│   │   └── runs/                  제공된 에이전트(ppo_v2_w64_r10), 반응형 임계값, 평가 결과
│   ├── dashboard/                 대시보드 스크립트, 축제 시나리오 예시 영상, 학습 그래프 (7단계)
│   ├── environment.yml            conda 환경
│   ├── README_env.md              환경 상세 (캠퍼스, 이음5G 무선, 용량, 전력, ISAC, 트래픽)
│   └── README_AI.md               데이터셋, TCN, RL 상세와 결과
├── Scenario/                      시나리오 인원표 (일반일과, 시험기간, 축제; CSV와 설명 PDF)
└── Map_info/                      캠퍼스 지도 이미지
```

- 코드는 `ns-3-dev`의 상위 폴더에서 `Scenario/`와 `Map_info/`를 읽는다. 폴더 구조를 그대로 둔다.
- 아래 명령은 따로 적지 않으면 `ns-3-dev/` 폴더에서, conda 환경 `smart_campus_5G`를 켠 상태로 실행한다.

---

## 0. 준비물

| 항목 | 확인한 환경 |
|---|---|
| OS | Ubuntu 22.04 |
| 컴파일러 | g++ 11.4 |
| CMake | 3.25 이상 (ns-3 요구). Ubuntu 22.04 apt의 3.22는 낮으니 snap 등으로 설치한다 |
| Ninja | 1.10 |
| conda | Anaconda 또는 Miniconda |
| GPU | 선택 사항. 아래 시간은 RTX 3070 8 GB + Ryzen 16스레드 기준이다 |

## 1. Python 환경

```bash
conda env create -f ns-3-dev/environment.yml    # 저장소 최상위에서. 환경 이름 smart_campus_5G
conda activate smart_campus_5G
```

- PyTorch 2.13.0 (CUDA 12.6), Gymnasium 1.3.0, numpy 2.2.6이 설치된다.
- protobuf는 3.20.3으로 고정했다. ns3-ai의 `messages_pb2.py`가 4.x 이상에서 오류를 낸다.
- 같은 이름의 환경이 이미 있으면 `-n <새 이름>`을 붙여 새로 만든다.
  - 2단계의 ns3-ai 설치는 그 환경이 이 폴더를 가리키게 바꾼다.
  - 다른 ns-3 폴더가 쓰는 환경에 설치하면 그 폴더의 RL 실행이 깨진다.

## 2. ns-3 빌드와 ns3-ai 연결

```bash
cd ns-3-dev
./ns3 configure                 # 기본값: examples·tests 꺼짐, default 프로필, Ninja
./ns3 build cttc-nr-demo        # 16스레드에서 약 3분
```

ns3-ai의 Python 쪽을 이 폴더에 연결한다.

```bash
pip install -e contrib/ai/python_utils                 # ns3ai_utils
pip install -e contrib/ai/model/gym-interface/py       # ns3ai_gym_env

# messages_pb2.py와 메시지 모듈이 있는 폴더를 Python 경로에 등록한다
echo "$PWD/contrib/ai/model/gym-interface/py" \
    > "$(python -c 'import site; print(site.getsitepackages()[0])')/ns3ai_gym_interface_path.pth"

# 메시지 모듈을 conda Python 3.10 헤더로 컴파일한다 (경고 "extra ';'"는 무시해도 된다)
g++ -O2 -std=c++20 -fPIC -shared -fvisibility=hidden \
    -DHAVE_STDINT_H -D__LINUX__ -DNS3_ASSERT_ENABLE -DNS3_LOG_ENABLE \
    -I build/include -isystem "$CONDA_PREFIX/include/python3.10" \
    contrib/ai/model/gym-interface/py/msg_py_binding.cc \
    -o contrib/ai/model/gym-interface/py/ns3ai_gym_msg_py.cpython-310-x86_64-linux-gnu.so \
    -L build/lib -lns3-dev-core-default -lns3-dev-ai-default \
    -Wl,-rpath,"$PWD/build/lib" -lrt -lpthread

python -c "import ns3ai_gym_env, ns3ai_gym_msg_py; print('ns3-ai OK')"
```

- **메시지 모듈**: ns-3 빌드가 만드는 모듈은 conda가 아닌 시스템 Python을 잡을 수 있어서 직접 컴파일한다.
  - 전체 빌드(`./ns3 build`)가 이 파일을 덮어쓴 뒤 `undefined symbol` 오류가 나면 위 g++ 명령을 다시 실행한다.
- **절대 경로**: `.pth` 파일과 `.so`에는 이 폴더의 절대 경로가 들어간다. 폴더를 옮기거나 다른 컴퓨터로 복사했다면 2단계를 다시 한다.
- **ns3-ai 수정**: ns3-ai는 numpy 2.x에 맞춰 `ns3_environment.py`의 `np.int`/`np.float` 3곳을 `np.int64`/`np.float64`로 고친 상태로 들어 있다.

## 빠른 확인 (최소 실행, 약 5분)

전체 절차(3\~6단계)는 10시간 넘게 걸린다. 2단계까지 마쳤다면, 아래 명령으로 데이터 생성부터 RL 평가까지 끝까지 이어지는지 먼저 확인할 수 있다.

```bash
# 3. 데이터: 시나리오 3개 x 3일 -> tcn-dataset-quick/
python build_tcn_dataset.py --scenarios 일반일과 시험기간 축제 --days 3 --outdir tcn-dataset-quick

# 4. TCN: 학습 0일 / 검증 1일 / 테스트 2일, 2+1 epoch -> DGU_SC_Project/checkpoints/quick/
QUICK=1 bash DGU_SC_Project/run_tcn_v3.sh

# 5. z 통계: 위 TCN의 것
python rl-env_v4/compute_z_stats.py --ckpt DGU_SC_Project/checkpoints/quick/seed_0_finetune/best.pt \
    --npz tcn-dataset-quick/tcn_dataset.npz --out DGU_SC_Project/checkpoints/quick/z_stats.json

# 6. RL: 반응형 임계값 3조합 조정, 위 TCN으로 PPO 2회 업데이트, 시나리오당 1일 평가 -> rl-env_v4/runs/quick/
bash rl-env_v4/run_exp_eum5g.sh quick
```

- **결과 해석**: 최소 설정이라 숫자는 보고한 결과와 비교할 수 없다. 단계가 오류 없이 이어지는지만 본다.
  - 에이전트는 2회 업데이트로는 거의 배우지 못해 상시 최대처럼 행동한다. 평가에 절감 0%, "KPI2 미달"이 나오는 것이 정상이다.
  - 걸린 시간(RTX 3070, 16스레드): 데이터 48초, TCN 15초, z 통계 2초, RL 2분 30초.
- **다시 돌리기**: 결과 폴더가 있으면 멈춘다. `rm -r tcn-dataset-quick DGU_SC_Project/checkpoints/quick rl-env_v4/runs/quick`로 지운 뒤 다시 한다.
- **업로드**: 이 결과물들은 `.gitignore`로 저장소에서 빠진다.

## 3. 데이터 생성

```bash
python build_tcn_dataset.py --scenarios 일반일과 시험기간 축제 --days 31 --plot --outdir tcn-dataset-v3-5g-31d
```

- **시간**: 약 8분 30초가 걸린다. ns-3 실행 파일이 없으면 먼저 `./ns3 build cttc-nr-demo`를 부른다.
- **만드는 과정**
  1. 시나리오 인원표에 날마다 인원 변동(표준편차 12%)을 준다.
  2. 사용자 1,000명의 분 단위 이동 경로를 만든다.
  3. ns-3 데이터셋 모드(안테나 16 고정)에서 셀별로 매분 다음을 기록한다: 요구 트래픽, 접속 UE 수, ISAC 감지 수, 평균 SINR, 핸드오버.
  4. Python에서 ISAC 밀집도, PRB 사용률, 이웃 부하, 시각 특징을 더한다.
  5. 입력 60분, 예측 15분 슬라이딩 창으로 자른다.
- **결과** (`tcn-dataset-v3-5g-31d/`)

| 파일 | 내용 |
|---|---|
| `tcn_dataset.npz` | TCN 학습 입력. X (1,083,078, 10, 60), Y (1,083,078, 15). 시나리오 3개 x 31일 x 18셀 (약 69 MB) |
| `dataset_perminute.csv` | 셀-분 단위 특징 10개 (약 106 MB) |
| `sites.csv`, `rates.csv`, `antenna.csv` | ns-3에 넘긴 사이트, 건물 유형별 수요, 안테나 설정 |
| `plot_*.png`, `campus_bg.png` | 데이터 확인용 그림 (`--plot`), 캠퍼스 지도 배경 |

- **재현성**: 날 시드가 (시나리오, 날)로 고정이라 같은 명령이면 같은 데이터가 나온다. 앞쪽 날은 `--days`를 줄여도 같다(예: `--days 15`의 0\~14일 = `--days 31`의 0\~14일). 생성 규칙과 크기는 [tcn-dataset-v3-5g-31d/README.md](ns-3-dev/tcn-dataset-v3-5g-31d/README.md)에 있다.
- **`--days 1`**: 날 변동 없이 시나리오 인원표 그대로 하루를 만든다. 빠른 동작 확인용이고, 위 데이터셋의 0일과는 다르다.
- **제공된 TCN과의 관계**
  - 제공된 TCN의 학습 데이터는 이전 무선 설정(3.5 GHz, 침투 손실 없음)으로 만들었다.
  - 지금 기본값(이음5G)으로 만들면 `Mean_sinr_db` 열만 다르다(이 열은 TCN이 쓰지 않는다).
  - 위 명령을 `--days 15`로 다시 만들어 제공된 데이터셋의 0\~14일과 비교했다. `Mean_sinr_db`를 뺀 X 9채널과 Y가 모두 같았고, 제공된 TCN의 KPI1도 9.5%로 같았다.
  - npz의 sha256은 이 열 때문에 체크포인트에 기록된 값(`11c116f4...`)과 다르다.

## 4. TCN 학습

```bash
OUT=checkpoints/my_tcn bash DGU_SC_Project/run_tcn_v3.sh      # 출력: DGU_SC_Project/checkpoints/my_tcn/
```

- **절차**
  1. Adam lr 1e-3으로 60 epoch 학습한다.
  2. lr 1e-4로 이어서 90 epoch까지 학습한다. 검증 손실이 가장 낮은 epoch를 `seed_0_finetune/best.pt`로 둔다.
  3. val/test 평가와 KPI1을 계산한다.
- **설정**: 31일 데이터, 폭 64, 학습 0\~10·15\~30일 / 검증 11\~12일 / 테스트 13\~14일이 기본값이다.
- **시간**: GPU에서 약 1.5시간 걸린다.
- **출력 폴더**: 같은 이름의 출력 폴더가 이미 있으면 덮어쓰지 않고 멈춘다.

제공된 TCN의 KPI1 확인 (9.5%가 나와야 한다). KPI1은 test 13\~14일만 쓰므로, 확인만 할 때는 3단계를 `--days 15`로 만들어도 된다(약 4분):

```bash
cd DGU_SC_Project
python eval_kpi1.py --npz ../tcn-dataset-v3-5g-31d/tcn_dataset.npz \
    --ckpt checkpoints/v3_5g_31d_w64_dual_proposal/seed_0_finetune/best.pt --cmax 580
cd ..
```

- **RL과의 연결**: RL은 기본으로 제공된 TCN(`DGU_SC_Project/checkpoints/v3_5g_31d_w64_dual_proposal/seed_0_finetune/best.pt`)을 쓴다.
- 새로 학습한 TCN을 쓰려면 5단계에서 그 TCN의 z 통계를 만들고, 6단계 학습에 `--tcn-ckpt`와 `--z-stats`를 넘긴다.

## 5. z 표준화 통계

RL State에는 TCN의 맥락 임베딩 z가 들어간다. 환경은 z를 TCN 학습 구간에서 구한 차원별 평균·표준편차로 표준화하므로, TCN을 새로 학습했다면 그 통계를 다시 만든다.

```bash
python rl-env_v4/compute_z_stats.py --ckpt DGU_SC_Project/checkpoints/my_tcn/seed_0_finetune/best.pt \
    --out DGU_SC_Project/checkpoints/my_tcn/z_stats.json          # 데이터는 기본값 tcn-dataset-v3-5g-31d
```

- 제공된 TCN을 그대로 쓰면 이 단계는 필요 없다. 인자 없이 실행하면 제공된 TCN의 통계를 `rl-env_v4/z_stats.json`에 다시 쓴다.
- 제공된 TCN으로 실행하면 제공된 `z_stats.json`과 같은 값이 나온다. GPU에서는 비트 단위로 같고, CPU에서는 상대 차이가 1e-6 이하다.

## 6. RL 학습과 평가

반응형 비교 기준의 임계값 조정과 PPO 학습을 터미널(또는 tmux) 두 개에서 함께 돌린다.

```bash
bash rl-env_v4/run_exp_eum5g.sh tune      # 반응형 임계값을 val 날에서 격자 조정 -> runs/reactive_tuned_eum5g_r10.json
bash rl-env_v4/run_exp_eum5g.sh train     # PPO 800회 업데이트 -> 조정 파일을 기다림 -> test 15일 평가

# 4단계에서 새로 학습한 TCN을 쓸 때 (train 뒤의 인자는 train_ppo.py로 넘어간다)
bash rl-env_v4/run_exp_eum5g.sh train --tcn-ckpt DGU_SC_Project/checkpoints/my_tcn/seed_0_finetune/best.pt \
    --z-stats DGU_SC_Project/checkpoints/my_tcn/z_stats.json
```

- **TCN 기록**: TCN과 z 통계 경로는 에이전트 체크포인트에 남는다. 평가(`evaluate.py`)는 그 경로를 읽어 학습 때와 같은 TCN을 쓴다. 반응형 임계값 조정은 TCN 예측을 쓰지 않아 TCN과 무관하다.

- **학습 설정**
  - 하루(09:00\~21:00) = 에피소드 1개(5분 step 144개). 업데이트마다 4일을 모은다.
  - 20회마다 val 6일로 `best.pt`를 고른다.
  - 보상 비율은 에너지 : QoS 부족 : 과부하 = 1:10:5다. State, 보상, 책임 귀속은 README_AI.md 4\~5절에 있다.
- **출력**: `rl-env_v4/runs/ppo_eum5g_w64_r10/`에 `best.pt`, `metrics.jsonl`, TensorBoard `tb/`, `eval_test.log`가 생긴다.
- **시간**: 업데이트당 약 25\~35초라 학습에 6\~8시간 걸린다. test 평가는 몇 분이다.
- **에이전트 차이**: 제공된 에이전트(`runs/ppo_v2_w64_r10`)는 이전 무선 설정(3.5 GHz)에서 학습해 이음5G 환경에서 평가한 것이다. 이음5G 설정으로 다시 학습한 결과는 아직 없다. 두 무선 설정의 평가 차이는 0.05%p 이하였다(README_env.md 11-2절).

**제공된 가중치로 평가만 하기** (TCN·z 통계·에이전트·반응형 임계값 모두 제공된 것)

```bash
python rl-env_v4/evaluate.py --ckpt rl-env_v4/runs/ppo_v2_w64_r10/best.pt --days 5 \
    --reactive rl-env_v4/runs/reactive_tuned_v2_r10.json --outage-samples 1 \
    --workdir rl-env_v4/run/eval_r10 --workers 8
```

| 기법 | 절감률 | QoS 부족 | 끊김률 | 과부하 |
|---|---|---|---|---|
| 상시 최대 | 0.0% | 1.40% | 0.00% | 2.11% |
| 반응형 (up 0.03 / down 0.02) | 29.2% | 4.91% | 0.29% | 2.73% |
| 에이전트 | 36.4% | 2.20% | 0.87% | 1.99% |

- 이음5G와 이전 무선 설정을 같은 날끼리 비교하려면 `EVAL_WORKERS=8 bash rl-env_v4/run_exp_eum5g.sh impact`를 쓴다(결과는 `rl-env_v4/runs/eum5g_impact/`).
- `--workers`는 평가 프로세스 수다. 프로세스마다 스레드 2개를 쓰므로 16스레드 CPU에는 8이 알맞다.
- 평가나 조정을 동시에 여러 개 돌릴 때는 `--workdir`를 서로 다르게 준다.

## 7. 대시보드 (선택)

하루 동안 세 기법(상시 최대, 반응형, 에이전트)의 안테나, 커버리지, 전력, QoS 부족을 지도 위에 보여 주는 영상과 셀별 예측 그래프를 만든다. 수집과 렌더링 명령은 [dashboard/README_이음5G.md](ns-3-dev/dashboard/README_이음5G.md)에 있다.

- **예시**: 축제 시나리오 결과만 저장소에 넣었다.
  - [dashboard/축제/dashboard_축제_이음5G.webm](ns-3-dev/dashboard/축제/dashboard_축제_이음5G.webm): 시나리오 대시보드 영상
  - `dashboard/축제/cells_이음5G/`: 셀별 그래프·영상
- **학습 그래프**: `dashboard/training/*.png`에 TCN·RL 학습 곡선이 있다.

---

## 외부 코드 버전

| 코드 | 저장소 | 커밋 | 이 저장소에서 바꾼 것 |
|---|---|---|---|
| ns-3 | https://gitlab.com/nsnam/ns-3-dev | `7db999bf9e9a5ed3d947778b593f80d6936c60ec` (2026-07-20) | 없음. `scratch/cttc-nr-demo.cc`와 프로젝트 폴더를 더했다 |
| 5G-LENA (`contrib/nr`) | https://gitlab.com/cttc-lena/nr | `7706af4421e0405a11320aaa70b783547542cef4` (2026-07-22) | 없음. 빌드에는 들어가지만 이 시나리오는 쓰지 않는다 |
| ns3-ai (`contrib/ai`) | https://github.com/hust-diangroup/ns3-ai | `b8c9858294b1d6a7f122b5154a3ce25057a54740` (2025-01-23) | `ns3_environment.py`의 numpy 2.x 대응 3줄 |

## 라이선스

GPL-2.0-only ([LICENSE](LICENSE)). ns-3, 5G-LENA, ns3-ai가 모두 GPL-2.0-only이고, 시나리오 코드는 5G-LENA 예제 `cttc-nr-demo`에서 출발했다. ns-3 안의 일부 파일은 다른 호환 라이선스를 따른다(`ns-3-dev/LICENSES/`).

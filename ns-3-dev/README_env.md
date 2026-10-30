# 이음5G 캠퍼스 시뮬레이션 환경 (README_env)

- 대상: 이 저장소의 `ns-3-dev/` 폴더.
- 작성: NS-3/환경팀, 2026-10-21 (코드 정리 반영. 이전 이름 `README_이음5G.md`)
- 바탕: v2 환경을 복사한 뒤 이음5G 4.7 GHz 대역 조건을 넣었다. 2026-10-21에 이음5G 결과를 만드는 코드만 남기고 정리했다(1-1절, 11-1절).
- 함께 볼 문서: 데이터셋, TCN 예측, RL 안테나 제어는 [README_AI.md](README_AI.md)에 있다.
- 판정 기호
  - **근거 있음**: 표준 문서나 논문, 업계 자료로 뒷받침된다.
  - **스펙**: 팀 공통 `최종공통엔진스펙.pdf`의 규칙이다(바꾸지 않음).
  - **가정**: 외부 근거가 약한 우리 선택이다. 보고서에 가정으로 밝힌다.
- 참고문헌: 대괄호 번호는 15절에 있다. README_AI.md와 같은 번호를 쓴다. "(2차)"는 보도·인용 문서로만 확인했거나 원문을 열어 보지 못한 문헌이다.

---

## 0. 한눈에 보기 (발표용 요약)

**한 문장 요약**: 이음5G 4.7 GHz 대역(3GPP n79 안의 4.72\~4.82 GHz, 100 MHz)을 쓰는 캠퍼스 단일 SA 망에서, 18개 셀의 활성 안테나 수를 5분마다 조절하는 상황을 1분 단위로 계산하는 ns-3 시스템 수준 시뮬레이션이다.

- 채널: 3GPP TR 38.901 UMi. 실내 사용자에게는 ITU-R P.2109 건물 침투 손실을 준다.
- 제어: 캠퍼스 18개 셀의 활성 안테나 수 {2, 4, 8, 16}을 5분마다 바꾼다(ns-3 + ns3-ai).
- 시나리오: 일반일과, 시험기간, 축제. 사용자 1,000명, 09:00\~21:00.

| 핵심 설정 | 값 | 근거 | 판정 |
|---|---|---|---|
| 반송파 / 대역폭 | 4.77 GHz (이음5G, n79) / 100 MHz | 이음5G 공급 대역 [62], n79 [67], 4.77 GHz 사례 [70] | 근거 있음 |
| 건물 침투 손실 | 실내 단말 16.2 dB | ITU-R P.2109-2 일반 건물 중앙값 [68], 4.7 GHz 사설망 연구 [69] | 근거 있음 |
| 셀 수 / 사용자 | 18셀(건물 단위) / 1,000명 | 엔진 스펙 §0, §11 | 스펙 |
| 커버리지 반경 | 안테나 2/4/8/16 -> 40/55/70/85 m | 엔진 스펙 §2 | 스펙 |
| 셀 용량 C_max | 580 Mbps (= 7.8 b/s/Hz x 100 MHz x 하향 0.743) | ITU-R M.2410 [4] | 근거 있음 |
| 과부하 기준 | 사용률 80% | 업계 혼잡 임계값 [15][16][17], M/M/1 [25] | 근거 있음 |
| 전력 모델 | P(A) = 0.3 + 0.7 x A/16 | TR 38.864 [10] | 근거 있음 |
| 끊김 기준 | SINR -5 dB 또는 커버리지 반경 밖 | ns-3 / 5G-LENA Qout [56] | 근거 있음 |
| 시간 | 09:00\~21:00, 1분 관측, 5분 제어 | 엔진 스펙 §3, §9 | 스펙 |

**v2에서 바뀐 것**: 반송파(3.5 -> 4.77 GHz)와 실내 침투 손실(없음 -> 16.2 dB) 두 가지다. 나머지 무선 값(대역폭, 출력, 반경, 높이, TDD)은 이음5G 근거 범위 안이라는 것만 확인하고 유지했다(4절).

**발표 핵심 메시지**
1. 무선 조건은 이음5G 규정 [62][65], 국내 4.7 GHz 사설망 연구 [69][70], 해외 캠퍼스 사설망 사례 [73]로 정했다(4절).
2. 이음5G로 바꿔도 결과 변화는 0.05%p 이하다. 모든 셀이 항상 송신하는 간섭 지배 환경이라, 반송파와 침투 손실은 사실상 잡음만 키운다(11-2절).
3. 이음5G 결과(데이터셋, TCN, RL 평가, 대시보드)를 만드는 코드만 남기고 정리했고, 정리 전후 결과가 같음을 확인했다(11-1절).

**처리 흐름**

```text
../Scenario/<시나리오>.csv  (건물 18개 + 캠퍼스 밖, 9개 시점 인원표)
  -> run_campaign.py      하루 인원 변동, UE별 웨이포인트 (실좌표, 분 단위)
  -> ns-3 scratch/cttc-nr-demo.cc   (ns-3 1초 = 1분)
       CoverageModel      TR 38.901 UMi + 침투 손실. 안테나 수 -> 송신 전력, 빔 이득, 반경
       --dataset          안테나 16 고정, 분당 셀별 관측 CSV -> build_tcn_dataset.py -> TCN 데이터셋
       --rlEnv            ns3-ai Gym: 매분 관측 전달, 안테나 적용 <-> rl-env_v4 (TCN 예측 + PPO 정책)
  -> rl-env_v4/evaluate.py (KPI), dashboard/scripts (대시보드)
```

**PPT 그림 자료**

| 그림 | 파일 | 내용 |
|---|---|---|
| 캠퍼스 지도 | `dashboard/data_이음5G/campus_bg.png` | 캠퍼스 경계와 건물. 좌표 x 0\~600 m, y 0\~550 m |
| 환경 동작 장면 (영상) | `dashboard/<시나리오>/dashboard_<시나리오>_이음5G.webm` | 셀별 안테나 수, 커버리지 반경, UE 위치, 기법별 전력과 QoS 부족 (5분마다 한 프레임) |
| 환경 동작 장면 (정지) | `python dashboard/scripts/render_frame.py dashboard/data_이음5G/시험기간 . frame.png 15:00` | 위 영상의 한 시각을 PNG로 만든다 (하루 기록을 먼저 수집해야 한다) |
| 셀별 하루 | `dashboard/<시나리오>/cells_이음5G/all_cells_이음5G.png` | 18셀의 요구 트래픽, TCN 예측, 에이전트 용량 |

- 대시보드 영상과 그림은 축제 시나리오 결과만 저장소에 있다. 하루 기록 수집과 다른 시나리오 렌더링 명령은 `dashboard/README_이음5G.md`에 있다.

---

## 1. 폴더와 실행 환경

| 항목 | 값 |
|---|---|
| 폴더 | 코드는 `ns-3-dev/`에 있다. 그 상위 폴더에 `Scenario/`(시나리오 인원표)와 `Map_info/`(캠퍼스 지도)가 있어야 한다 |
| 실험 환경 | Ubuntu 22.04, Ryzen 16스레드, RTX 3070 8 GB |
| Python | conda `smart_campus_5G` (`environment.yml`: Python 3.10, PyTorch 2.13.0 + CUDA 12.6, Gymnasium 1.3.0) |
| ns3-ai 연결 | conda 환경의 ns3-ai Python 모듈이 이 폴더의 `contrib/ai`와 그 `build/lib`를 가리키도록 설치한다(아래 ns3-ai 설치) |
| ns-3 빌드 | `./ns3 configure` (기본: examples·tests 꺼짐, default 프로필, Ninja) -> `./ns3 build cttc-nr-demo`. cmake를 snap으로 설치했다면 먼저 `export PATH=/snap/bin:$PATH` |
| 주의 | 복사한 폴더에서는 `./ns3 clean`을 쓰지 않는다. `./ns3`는 `.lock-ns3_linux_build`에 적힌 출력 폴더를 쓰므로, 복사 직후에는 원본 폴더의 build를 지울 수 있다 |

**ns3-ai 설치** (처음 환경을 만들 때 한 번)

1. **패키지**: `environment.yml`로 conda 환경을 만든다. protobuf는 3.20.3으로 고정한다. ns3-ai의 구형 `messages_pb2.py`가 4.x 이상에서 오류를 낸다.
2. **Python 모듈**: `pip install -e contrib/ai/model/gym-interface/py`로 `ns3ai_gym_env`를 설치하고, `ns3ai_python_utils`도 같은 방식으로 설치한다.
3. **경로 등록**: site-packages에 `.pth` 파일을 두어 `contrib/ai/model/gym-interface/py` 경로를 등록한다. `messages_pb2`와 `ns3ai_gym_msg_py`를 import하는 데 필요하다.
4. **numpy 2.x 대응**: `ns3_environment.py`의 `np.int` / `np.float`를 `np.int64` / `np.float64`로 바꾼다. numpy 2.x에서 없어진 별칭이다.
5. **메시지 모듈 컴파일**: `ns3ai_gym_msg_py.cpython-310-*.so`는 conda Python 3.10 헤더로 g++ 직접 컴파일한다(ns-3 core·ai 라이브러리 링크, RUNPATH = 그 폴더의 `build/lib`).
   - `./ns3 build`가 만드는 것은 Python 3.9를 잡아 비어 있다.
   - 다른 컴퓨터로 복사한 `.so`는 쓰지 않는다.

### 1-1. 코드 구성

| 파일 | 역할 |
|---|---|
| `scratch/cttc-nr-demo.cc` | ns-3 시나리오 (1,613줄). `CoverageModel`(전파, 안테나, SINR, 반경), `DatasetCollector`(데이터셋 모드), `RlAntennaEnv`(RL 모드, ns3-ai Gym). 5G-LENA 예제 `cttc-nr-demo`에서 시작했고, Python 쪽이 이 이름으로 실행하므로 이름은 그대로 둔다 |
| `run_campaign.py` | 캠퍼스 도우미: 사이트 표 `SITES`, `sites.csv` 쓰기, 시나리오 인원표 읽기, UE 웨이포인트 생성, 지도 자르기, ns-3 빌드. 데이터셋 생성과 RL 환경이 import 한다 |
| `build_tcn_dataset.py` | 데이터셋 모드 실행과 TCN 데이터셋(npz) 생성. C_max, 1인당 수요, 선행 시차 등 상수 (README_AI.md 2절) |
| `rl-env_v4/` | RL 환경, PPO 학습, 평가 (README_AI.md 4\~8절) |
| `DGU_SC_Project/` | TCN 모델, 학습, KPI1 (README_AI.md 3절) |
| `dashboard/` | 대시보드 생성 스크립트, 축제 시나리오 그림·영상, 학습 그래프 (`dashboard/README_이음5G.md`) |
| `environment.yml` | conda 환경 정의 |
| 상위 `Scenario/`, `Map_info/` | 시나리오 인원표 CSV 3개, 캠퍼스 지도 이미지 |

- ns-3 본체(`src/`)와 5G-LENA(`contrib/nr`)는 고치지 않았다. ns3-ai(`contrib/ai`)는 numpy 2.x 대응 파일 하나만 고쳤다(위 설치 4번).
- **2026-10-21 코드 정리**: 이음5G 결과에 쓰지 않는 코드를 지웠다.
  - ns-3 시나리오(3,975 -> 1,613줄): 원래 5G-LENA 데모(NR 스택, UDP, FlowMonitor), 스냅샷 수요 집계, REM 지도, NetAnim 애니메이션 모드, 데이터셋 V1 생성, 1초 판정 생략 증명, RL 옛 스위치(반경 규칙 끄기, 담당 셀 귀속 끄기)
  - Python: 옛 캠페인 실행부(`run_campaign.py` 845 -> 254줄), 데이터셋 V1과 `--version`·`--isac` 옵션, 5G 이전 단위 보정(`traffic_scale`), 쓰지 않는 TCN 모델·정책·옵션
  - 정리 전후 결과가 같음을 확인했다(11-1절). 지운 코드는 이 저장소에 넣지 않았다.

---

## 2. 시뮬레이션 방식

| 항목 | 값 | 비고 |
|---|---|---|
| 네트워크 시뮬레이터 | ns-3 (3-dev) | 시나리오 코드 `scratch/cttc-nr-demo.cc` |
| 5G NR 모듈 | 5G-LENA NR (`contrib/nr`) | ns-3 트리에 있지만 이 시나리오는 쓰지 않는다 |
| RL 연동 | ns3-ai v1.2.0 (Gym 인터페이스, 공유 메모리) | Python 정책 <-> C++ 환경 |
| 모드 | 시스템 수준. 데이터셋 모드(`--dataset`), RL 모드(`--rlEnv`) 두 가지 | 위치, 전파 손실, 링크 버짓으로 셀 지표를 계산한다. 스케줄러, MCS, HARQ는 없다 |
| 시간 해상도 | 1분 (ns-3 1초 = 1분) | 하루 = 09:00\~21:00 = 720분 |

쉽게 말하면, 패킷을 하나하나 보내는 대신 "누가 어디 있고, 신호가 얼마나 세며, 셀이 얼마나 찼는가"를 1분마다 계산한다. RL 학습에 하루 3,200일 분량이 필요해서 전체 프로토콜 시뮬레이션은 쓸 수 없다. RAN 에너지 절감 연구 대부분도 같은 수준의 추상화를 쓴다 [48][49][50].

---

## 3. 캠퍼스 배치와 시나리오

### 3-1. 사이트(셀)

- 셀 번호는 C++와 Python 모두 (y, x) 좌표 순으로 정한다.
- 셀마다 gNB를 사이트 좌표에 하나씩, 높이 10 m로 둔다.
- "gNB 수"는 엔진 스펙이 정한 실제 기지국 하드웨어 수다. 고정값이고 계산에는 쓰지 않는다.
- 실내 = 1이면 건물이다. 그 사이트 근처 단말에 침투 손실이 적용된다(4-3절).

| 셀 | 건물 | 유형 | gNB 수 | 실내 | 최근접 3셀 |
|---|---|---|---|---|---|
| 1 | 대운동장 | 이벤트 | 3 | 0 (운동장) | 3, 5, 4 |
| 2 | 경영관 | 강의동 | 2 | 1 | 4, 7, 8 |
| 3 | 과학관 | 강의동 | 4 | 1 | 6, 1, 11 |
| 4 | 혜화관 | 강의동 | 4 | 1 | 5, 2, 7 |
| 5 | 만해관 | 강의동 | 2 | 1 | 11, 4, 3 |
| 6 | 중앙도서관 | 도서관 | 7 | 1 | 10, 13, 3 |
| 7 | 문화관 | 이벤트 | 4 | 1 | 8, 9, 2 |
| 8 | 학술문화관 | 도서관 | 1 | 1 | 7, 9, 2 |
| 9 | 학술관 | 도서관 | 3 | 1 | 8, 7, 2 |
| 10 | 신공학관 | 강의동 | 12 | 1 | 12, 6, 14 |
| 11 | 만해광장 | 이벤트 | 2 | 0 (광장) | 13, 5, 3 |
| 12 | 남산학사 | 기숙사 | 4 | 1 | 10, 14, 6 |
| 13 | 본관 | 강의동 | 2 | 1 | 11, 14, 6 |
| 14 | 원흥관 | 강의동 | 2 | 1 | 15, 13, 10 |
| 15 | 정보문화관P | 강의동 | 6 | 1 | 17, 18, 14 |
| 16 | 체육관 | 이벤트 | 2 | 1 | 18, 15, 13 |
| 17 | 정보문화관Q | 강의동 | 1 | 1 | 15, 18, 14 |
| 18 | 학생회관 | 이벤트 | 2 | 1 | 15, 17, 16 |

| 항목 | 값 | 근거 / 출처 | 판정 |
|---|---|---|---|
| 사이트 | 18개 (강의동 9, 도서관 3, 기숙사 1, 이벤트 5), 좌표는 `run_campaign.py` `SITES` | 엔진 스펙 §0, §1 (실측 배치) | 스펙 |
| 기지국 간격 | 최근접 중앙값 67 m (18 m \~ 110 m 이상) | ITU-R M.2412 Dense Urban 소형셀 층의 등가 간격 약 67 m [3]. TR 36.872 소형셀 클러스터 [6] | 근거 있음 |
| 기지국 / 단말 높이 | 10 m / 1.5 m | TR 38.901 UMi [1] | 근거 있음 |
| 이웃 | 최근접 3셀 (위 표) | 엔진 스펙 §5 | 스펙 |

### 3-2. 사람과 이동

| 항목 | 값 | 근거 / 출처 | 판정 |
|---|---|---|---|
| 사용자 수 | 1,000명 | 엔진 스펙 §11 | 스펙 |
| 시나리오 | 일반일과, 시험기간, 축제 (`../Scenario/*.csv`) | 엔진 스펙 §10. 선행 유입(S2) 시나리오는 없음 | 스펙 |
| 인원표 | 건물 18개와 캠퍼스 밖의 인원, 9개 시점(09:00\~21:00, 1.5시간 간격), 시점별 합 1,000 | 엔진 스펙 §11 | 스펙 |
| 단말 배치 | 시점마다 건물 중심 반경 18 m 안의 고정 위치(단말마다 정해짐). 캠퍼스 밖 인원은 지도상 캠퍼스 경계 밖의 고정 지점 | `run_campaign.py` `BUILDING_SCATTER_R` | 가정 |
| 이동 | 두 시점 사이를 직선 등속으로 이동(ns-3 WaypointMobility) | 우리 구현 | 가정 |
| 날 생성 | 건물별·시점별 인원에 (1 + N(0, 12%))를 곱해 하루를 만들고, 캠퍼스 밖 인원으로 합을 1,000에 맞춤. 날은 정수 시드로 고정되어 같은 시드면 같은 날 | `day_var = 0.12`. 셀 단위 요일 간 변동을 측정한 공개 자료는 찾지 못함 | 가정 |
| 알려진 문제 | 캠퍼스를 떠나는 사람이 다음 시점까지 접속 UE로 남는다 | TCN 학습 데이터와 같은 현상이라 유지. RL에서는 서비스 구역 밖으로 처리해 끊김에서만 뺀다 | |

---

## 4. 무선 (이음5G)

### 4-1. 이음5G란

| 항목 | 내용 | 근거 |
|---|---|---|
| 정의 | 이동통신사가 아닌 기업·기관이 건물, 캠퍼스, 공장 같은 특정 구역에 직접(또는 위탁해) 구축하는 5G 특화망의 국내 명칭. 과학기술정보통신부가 주파수를 공급한다 | [62][63] |
| 공급 주파수 | 4.72\~4.82 GHz 100 MHz (10 MHz 블록 10개, 신청 단위)<br>28.9\~29.5 GHz 600 MHz (50 MHz 블록 12개) | [62] |
| 개방과 구조 | 2021-10-28 개방. SA(단독 모드) 기반 구축을 지원한다 | [63][64] |
| 3GPP 대역 | 4.72\~4.82 GHz는 n79(4,400\~5,000 MHz, TDD) 안에 있다 | [67] |
| 송신 출력 한도 | 4.7 GHz는 100 MHz 기준 80 W(약 49 dBm), 28 GHz는 400 MHz 기준 2 W | 행정예고 보도 [65], 이 조항을 인용한 2023 고시 [66], 49 dBm/100 MHz로 시뮬레이션한 논문 [70]. 고시 원문(HWP)은 열어 보지 못해 간접 확인이다 |
| TDD 프레임 | 규제로 정한 비율은 찾지 못했다. 같은 채널을 쓰는 4.7 GHz 사설망끼리 동기화가 안 되면 약 500 m, 동기화되면 약 100 m 떨어져야 공존할 수 있다 | [69]. 장비는 상향 비중을 늘린 설정(2:8\~5:5)을 지원한다 [71] |
| 4.7 GHz 사설망 평가 파라미터 | 40 MHz 채널, 셀 반경 50 m, 기지국 27 dBm/40 MHz, 8x8 배열, 안테나 높이 실외 6 m·실내 3 m, 단말 잡음지수 9 dB, 건물 침투 손실 16.3 dB(일반)/31.5 dB(단열) | [69] |
| 4.77 GHz 시뮬레이션 사례 | 4.77 GHz/100 MHz, 기지국 49 dBm/100 MHz, 서비스 영역 RSRP -105 dBm 이상, SEAMCAT 몬테카를로. 이음5G에 머신러닝(SVM)을 쓴 국내 사례지만 주제는 망 사이 주파수 공동 사용이다 | [70] |
| 대학 사례 | 경희대(LG CNS, 2022-12): 드론·로봇 원격 제어 실증. 주파수, 기지국 수, 출력은 공개되지 않았다 | [72] |
| 해외 캠퍼스 사설망 | Edinburgh Campus5G: 영국 공유 면허 대역 n77(3.8\~4.2 GHz), 건물 6곳에 실외 RU 20개, RU 100 MHz, O-RAN RIC | [73] |
| 정부 실증 | 2022년 이음5G 융합서비스 실증 11개 과제(480억 원) | [74] |

- **우리 과제의 위치**
  - 이번 조사 범위의 국내 이음5G 연구는 망 사이 주파수 공존과 전파 측정이 중심이다 [69][70][75]. 예측 기반 에너지 절감 연구는 찾지 못했다.
  - 그래서 이 과제는 "이음5G 캠퍼스 조건에서의 예측 기반 MIMO 에너지 절감"으로 설명할 수 있다.
  - KCI·DBpia에서 직접 검색하지는 않았으므로 "없다"가 아니라 "이번 조사에서 찾지 못했다"로 쓴다.
- **보고서 표현**
  - 대학 이음5G 구축 사례 가운데 기술 수치가 공개된 것은 없다.
  - 그래서 이 환경은 "이음5G 4.7 GHz 캠퍼스 조건 모사"라고 쓰고, 근거는 다음 세 가지로 든다.
    - 이음5G 규정 [62][65]
    - 국내 4.7 GHz 사설망 연구 [69][70]
    - 해외 캠퍼스 사설망 사례 [73]

### 4-2. 무선 파라미터

| 항목 | 값 | 근거 / 출처 | 판정 |
|---|---|---|---|
| 반송파 | 4.77 GHz (v2까지는 3.5 GHz, n78) | 이음5G 공급 대역의 중심 [62], n79 [67], 4.77 GHz 사례 [70] | 근거 있음 |
| 대역폭 | 100 MHz | 이음5G 4.7 GHz 최대 폭 [62], 캠퍼스 사설망 RU 100 MHz [73], TR 38.864 Set 1 [10] | 근거 있음 |
| 열잡음 | -174 dBm/Hz + 10log10(100 MHz) + 잡음지수 9 dB = -85.0 dBm | 단말 잡음지수 9 dB는 TR 38.901 보정 가정 [1] | 근거 있음 |
| 경로손실 | TR 38.901 UMi Street Canyon (ns-3 `ThreeGppUmiStreetCanyonPropagationLossModel`) | [1] | 근거 있음 |
| LOS / 섀도잉 | 항상 LOS, 섀도잉 없음 | 38.901은 LOS 확률(67 m에서 0.38)과 섀도잉 4\~7.8 dB를 둔다 [1] | **제외 (단순화)** |
| 건물 침투 손실 | 실내 단말의 모든 링크에 16.2 dB (4-3절) | P.2109-2 [68], [69] | 근거 있음 (중앙값 고정은 단순화) |
| 총 송신 전력 | 안테나 16개일 때 30 dBm (안테나당 약 18 dBm) | TR 36.814 피코셀 "총 24 또는 30 dBm" [7], TS 38.104 Local Area [2]. 이음5G 한도 80 W(약 49 dBm)/100 MHz 안 [65][66]. 4.7 GHz 소형셀 27 dBm/40 MHz(100 MHz 환산 약 31 dBm) [69] | 근거 있음 |
| SINR | 서빙 신호(빔포밍 이득 포함) / (다른 모든 셀 전력의 합 + 잡음), 광대역 | ITU-R M.2101 [59] | 근거 있음 |
| 간섭 가정 | 모든 셀이 항상 송신(full buffer) | ITU-R M.2412 [3]. 간섭을 실제보다 크게 보는 보수적 가정 | 근거 있음 |
| TDD | 하향 비율 0.743 (DDDSU, S 슬롯 10:2:2), 동기 단일 망. C_max에만 반영 | 이음5G 고유 규정은 찾지 못함. 동기 운용이 공존에 유리 [69], 장비는 2:8\~5:5 지원 [71] | 가정 |
| 끊김 기준 | SINR < -5 dB | ns-3 LTE·5G-LENA Qout = -5 dB [56], NR 데이터 채널 최저 동작점 약 -5.7 dB [36], TR 36.942 최소 SINR -10 dB [31] | 근거 있음 |

**반송파 4.77 GHz의 효과**
- TR 38.901 UMi LOS 식 PL1 = 32.4 + 21 log10(d3D) + 20 log10(fc) [1]로 보면, 경로손실이 3.5 GHz보다 20 log10(4.77/3.5) = **2.69 dB** 크다.
- 이 식은 꺾임 거리 d'BP = 4 h'BS h'UT fc / c 안에서 쓴다. d'BP는 3.5 GHz에서 210 m, 4.77 GHz에서 286 m다. 서빙 거리(85 m 이하)는 모두 이 안에 든다.
- 18개 셀이 모두 같은 주파수라 서빙 신호와 간섭이 함께 약해진다. 그래서 SINR은 잡음 항이 상대적으로 커지는 만큼만 낮아진다.

### 4-3. 건물 침투 손실과 실내 판정

- **P.2109-2 식** [68]
  - 전체 손실: L_BEL(P) = 10 log10(10^(0.1A) + 10^(0.1B) + 10^(0.1C))
  - A = F^-1(P) σ1 + μ1, B = F^-1(P) σ2 + μ2, C = -3.0
  - μ1 = L_h + L_e, L_h = r + s log10(f) + t (log10 f)^2, L_e = 0.212 |θ|
  - μ2 = w + x log10(f), σ1 = u + v log10(f), σ2 = y + z log10(f) (f는 GHz)
- **Table 1 계수** (일반 건물 / 단열 건물)

| r | s | t | u | v | w | x | y | z |
|---|---|---|---|---|---|---|---|---|
| 12.64 / 28.19 | 3.72 / -3.00 | 0.96 / 8.48 | 9.6 / 13.5 | 2.0 / 3.8 | 9.1 / 27.8 | -3.0 / -2.9 | 4.5 / 9.4 | -2.0 / -2.1 |

- **4.77 GHz 값**: 중앙값(P = 50%, F^-1 = 0)과 수평 입사(θ = 0)로 계산하면 **일반 건물 16.23 dB, 단열 건물 31.45 dB**다. 3.5 GHz였다면 15.72 / 30.89 dB다.
- **다른 근거와의 비교**: 4.7 GHz 사설망 연구의 16.3 / 31.5 dB [69]와 일치한다. TR 38.901 저손실 O2I 평균(약 17 dB) [1]과도 비슷하다.
- **적용 방식**: 실내 단말에는 모든 기지국 링크(서빙과 간섭 모두)에 같은 손실을 준다. 그래서 접속 셀과 핸드오버는 바뀌지 않는다. SINR은 S / (I + N x L)로, 잡음이 L배 커진 것과 같다.
- **실내 판정**
  - 건물 사이트 중심에서 18 m(단말 산포 반경) 안에 있는 단말을 실내로 본다.
  - 만해광장과 대운동장은 야외라 제외한다(`run_campaign.py` `OUTDOOR_SITES` -> `sites.csv` `indoor` 열).
  - 건물 사이를 이동 중인 단말과 캠퍼스 밖 단말은 실외다.
- **실내 비율**
  - test 첫날 기준으로, 캠퍼스 안 단말-분 가운데 일반일과 44%, 시험기간 56%, 축제 63%가 실내다.
  - TR 38.901 UMi 가정 80% [1]보다 낮다. 단말이 두 시점 사이 1.5시간 동안 직선으로 이동해서, 실외로 잡히는 시간이 실제보다 길기 때문이다.
  - 같은 날을 다른 규칙으로 계산하면, 정지한 단말만 실내로 볼 때 21/40/49%, 캠퍼스 안 전원을 실내로 볼 때 100%다.
- **가정**
  - P.2109와 TR 38.901은 손실을 분포로 다루지만, 여기서는 중앙값 하나로 고정했다.
  - 단열 건물 31.5 dB는 민감도 확인용이다(`--o2iLossDb 31.5`).

---

## 5. 커버리지와 안테나

| 항목 | 값 | 근거 / 출처 | 판정 |
|---|---|---|---|
| 제어 변수 | 셀별 활성 안테나 수 A in {2, 4, 8, 16}, 시작값 16 | 엔진 스펙 §2, 제안서 | 스펙 |
| 커버리지 반경 | 2 -> 40 m, 4 -> 55 m, 8 -> 70 m, 16 -> 85 m (수평 거리) | 엔진 스펙 §2 ("최소 단계에서 실질적 공백이 생기도록") | 스펙 |
| 반경 크기의 타당성 | 40\~85 m | TR 36.872 소형셀 반경 50 m·단말 70 m [6], TR 38.913 실내 핫스팟 간격 20 m [8], TR 38.807 도심 micro(대학 캠퍼스) 간격 100 m [9], 4.7 GHz 사설망 셀 반경 50 m [69] | 근거 있음 |
| 빔포밍 이득 | 서빙하는 단말 쪽에만 10log10(A) (3/6/9/12 dB). 다른 셀 단말 쪽(간섭)은 이득 없음 | 배열 이득 [13], TR 37.840 복합 패턴 [14], 비서빙 방향 [58][59] | 근거 있음 |
| 총 송신 전력 | 30 + 10log10(A/16) dBm (21/24/27/30 dBm) | 안테나당 출력 일정: TR 38.864 공간 영역 적응 [10] | 근거 있음 |
| 안테나 절반당 커버리지 감소 | 서빙 쪽 EIRP 6 dB 감소 (24/30/36/42 dBm) | TR 38.864 "64 -> 32 TxRU로 줄이면 하향 커버리지 4.5\~6.5 dB 감소" [10], NGMN "최대 6 dB" [11] | 근거 있음 |
| 접속 규칙 | 자기를 반경 안에 두는 셀 중 수신 전력(빔포밍 이득 포함)이 가장 센 셀. 안테나를 줄이면 경계 단말이 이웃 셀로 넘어간다(스티어링) | 엔진 스펙 §8 | 스펙 |
| 끊김 판정 | 서비스 구역 안에서, 어느 셀의 반경에도 없거나 SINR < -5 dB | 우리 구현 | 스펙 + 우리 구현 |
| 끊김 귀속 | 반경 밖 끊김은 가장 가까운(담당) 셀의 책임. 항상 적용한다(`RlAntennaEnv::ClassifyConnected`) | 안테나를 줄인 셀이 자기가 만든 끊김의 벌점을 받게 함 | 우리 구현 |
| 서비스 구역 | 모든 셀이 85 m여도 닿지 않는 위치(캠퍼스를 떠나는 사람)는 끊김에서 뺀다. 수요와 UE 수에는 넣는다 | 어떤 제어로도 서비스할 수 없는 위치 | 우리 구현 |
| 원형 반경 추상화 | 반경 안 = 서비스, 밖 = 끊김 | RL 에너지 절감 논문의 관행 [48][49][50] | 근거 있음 (관행) |

**링크 버짓으로 본 반경 (참고)**: 조건은 SINR -5 dB, 잡음만 있는 경우, TR 38.901 UMi NLOS 식(35.3 log10 d + 22.4 + 21.3 log10 fc), 4.77 GHz다.

| 사용자 위치 | A=2 | A=4 | A=8 | A=16 |
|---|---|---|---|---|
| 실외 | 153 m | 227 m | 336 m | 497 m |
| 실내 일반 건물 (16.2 dB) | 53 m | 79 m | 117 m | 173 m |
| 실내 단열 건물 (31.5 dB) | 20 m | 29 m | 43 m | 64 m |
| **엔진 스펙** | **40 m** | **55 m** | **70 m** | **85 m** |

- 스펙 반경은 실내 사용자 두 경우 사이에 있다. 보고서에서는 "스펙 반경은 실내 사용자(침투 손실 16\~31 dB)의 링크 버짓 범위 안에 있는 서비스 반경 추상화"로 쓴다.
- **알아둘 불일치**
  - 스펙 사다리(85 -> 70 -> 55 -> 40 m)는 NLOS 기울기로 환산하면 절반당 3.0\~4.9 dB에 해당한다.
  - 이 모델(절반당 6 dB)로 계산하면 85 -> 58 -> 39 -> 26 m가 된다. 그래서 스펙은 안테나가 적을 때의 커버리지를 물리보다 후하게 본다.
  - 비교 기법 모두에 같은 규칙이 적용되므로 비교의 공정성은 유지된다. 반경 +-20% 민감도 평가로 보강할 수 있다.

---

## 6. 용량, 부하, 과부하

| 항목 | 값 | 근거 / 출처 | 판정 |
|---|---|---|---|
| 셀 최대 용량 C_max | 580 Mbps = 100 MHz x 하향 0.743 x 7.8 b/s/Hz | ITU-R M.2410 Dense Urban 하향 평균 요구 효율 [4]. 비교: 4G 2.6 [5], NR 자체평가 9.2\~14.8 [34] | 근거 있음 (하향 비율은 가정) |
| 안테나별 용량 | C(A) = C_max x A/16 = 72.5 / 145 / 290 / 580 Mbps | 엔진 스펙 §2. 실측은 정비례보다 덜 줄어든다 [33] -> 안테나 감축 비용을 크게 보는 보수적 규칙 | 스펙 |
| 사용률 U | min(1, 처리 수요 / C(A)), 처리 수요 = 셀 요구 트래픽 - 끊긴 UE 수요 | 엔진 스펙 §2 (PRB 사용률 = D/C) | 스펙 |
| 과부하 기준 U_th | 0.8. 벌점 O = max(0, U - 0.8) / 0.2 | 업계 공통 혼잡 임계값 80% [15], Ericsson [16], Nokia [17], M/M/1 무릎 70\~90% [25]. 표준은 사업자 설정값으로 둔다 [20][21][22][23]. 대안 70% [18][19], 90% [17] | 근거 있음 |
| 받는 셀 상한 | 받는 셀이 상한을 넘으면 그 셀에 벌점. 떠넘긴 이웃도 공간 할인으로 함께 부담한다(README_AI.md 4-2절) | 엔진 스펙 §8, 부하 분산 구조 [24], TS 38.300 [21] | 스펙 + 근거 있음 |

시험기간처럼 수요가 셀 최대 용량을 넘는 시간에는, 안테나를 모두 켜도 처리하지 못하는 트래픽(QoS 부족 약 3.4%)이 남는다.

---

## 7. 전력 모델

| 항목 | 값 | 근거 / 출처 | 판정 |
|---|---|---|---|
| 셀 상대 전력 | P(A) = s + (1 - s) x A/16, s = 0.3 (안테나 16 = 1.0) -> 0.3875 / 0.475 / 0.65 / 1.0 | TR 38.864 §5.1 공간 영역 적응: 정적 부분 + 활성 TxRU 비율에 비례하는 부분. 정적 비율 0.196(기본) / 0.295(옵션) [10] | 근거 있음 |
| 부하 의존성 | 없음 (활성 상태, full buffer) | 슬립 상태를 넣으면 정비례 용량 규칙에서 안테나 조절 절감이 0이 되어 제외 -> 절감률은 활성 상태 절감의 상한 | 가정 (근거 명시) |
| 절감률 | 1 - 하루 평균 P (상시 최대 = 0%) | | |
| 참고 값 | NGMN 실측 AAU 정적 비율 56% [12], EARTH micro 정적 비중 0.77 [35] | s가 커지면 절감률이 작아진다 | |

---

## 8. ISAC 센싱 (사람 수 감지)

| 항목 | 값 | 근거 / 출처 | 판정 |
|---|---|---|---|
| 방식 | 접속 여부와 무관하게, 가장 가까운 사이트가 반경 안의 사람을 감지(presence) | 제안서, 엔진 스펙 §5 | 스펙 |
| 센싱 반경 | 100 m | TR 22.837 [27], ETSI GR ISC 001 [28]. 실증 디바이스프리 카운팅은 실내 10 m, 실외 수십 m [37][39] | 근거 있음 (표준 목표치 수준) |
| 검출 확률 P_D | 0.9 (사람당, 분당) | TS 22.137 실외 사람 검출 요구치 0.95 이상 [26] -> 보수적 | 근거 있음 |
| 측정 오차 | 감지 수 x (1 + N(0, 8%)), Python에서 더함 | 엔진 스펙 §5. Wi-Fi 카운팅 정확도 89\~93% [37], 교실 재실 오차 13% [38] | 스펙 + 근거 있음 |
| 오경보 | 없음 | 3GPP 가정 2\~5% [26] | 가정 (미반영) |

---

## 9. 트래픽 발생 모델

| 항목 | 값 | 근거 / 출처 | 판정 |
|---|---|---|---|
| 1인당 기본 수요 | 강의동 2.32, 도서관 0.87, 기숙사 1.74 (18시 이후 x2 = 3.48), 이벤트 1.45 Mbps | 엔진 스펙 §6의 정성 등급을 0.8/0.3/0.6/0.5로 수치화하고 C_max와 같은 배수(x2.9)로 확대 (`build_tcn_dataset.py` `TYPE_RATE_MBPS`) | 가정 |
| 수요 수준 | 셀-분 중앙값 25.8 Mbps, 상위 1% 549 Mbps, 최대 1,287 Mbps. U = D/580 > 0.8인 셀-분 2.00% (데이터셋 v3-5g-31d) | 국내 5G 1인당 월 29 GB [54]를 피크 시간으로 환산한 값보다 크다. HD 영상 3\~5 Mbps [55] -> "캠퍼스 피크 강부하 시나리오" | 가정 |
| 선행 시차 (도착 -> 트래픽) | 강의동 5분, 도서관 2분, 기숙사 0분, 이벤트 12분 | 엔진 스펙 §6. 캠퍼스 트래픽이 강의 시간표를 따른다는 측정 [42] | 스펙 (크기는 가정) |
| 접속 지연 | 건물에 도착한 뒤 선행 시차 + 0\~6분 균등(평균 3분, 단말마다 고정) 뒤 접속. 09:00 재실자는 막 도착한 것으로 본다 | 무선 접속 자체는 초 단위 [30] -> "자리를 잡고 첫 데이터 세션을 시작하기까지"로 해석 | 가정 |
| 활동 확률 | 접속 UE가 매분 70% 확률로 트래픽 발생, 수요 = 기본 x 배수 / 0.7 (평균 보존) | 3GPP 평가는 세션 도착 기반 [7], 활동 비율 10\~30% [8] -> 우리가 높음 | 가정 |
| 사용자별 편차 | 배수 균등 [0.7, 1.3] | 실측은 긴 꼬리 분포 [41] -> 우리가 좁음 | 가정 |
| 건물 유형 패턴 | 기숙사는 저녁, 강의동은 주간 | 캠퍼스 무선망 [40], 주거 지역 야간 피크 [41] | 방향은 근거 있음, 배수 2는 가정 |

이 절의 값은 TCN 학습 데이터에도 들어간다. 바꾸면 데이터셋을 다시 만들고 TCN도 다시 학습해야 한다(AI팀 영향).

---

## 10. 환경이 내보내는 관측

ns-3는 1분마다 셀별로 다음 값을 낸다. 데이터셋 모드는 이를 CSV로 쓰고, RL 모드는 ns3-ai로 넘긴다. TCN 입력 창과 RL State는 [README_AI.md](README_AI.md)에 있다.

| 값 | 정의 | 데이터셋 CSV 열 |
|---|---|---|
| 요구 트래픽 | 셀에 접속한 UE 요구량의 합 (Mbps) | `traffic_demand_mbps` |
| 접속 UE 수 | 셀에 접속한 UE 수 | `ue_count` |
| ISAC 감지 수 / 재실 인원 | 감지 반경 안에서 감지된 사람 수 / 반경 안의 사람 수 (8절) | `isac_count` / `presence_count` |
| 평균 SINR | 접속 UE SINR의 선형 평균 (dB) | `mean_sinr_db` |
| 핸드오버 in / out | 직전 분과 서빙 셀이 바뀐 UE 수 | `ho_in` / `ho_out` |
| (RL 모드) 활성 안테나 수, 끊긴 UE 수와 그 수요 | 5절, 6절 | 데이터셋에는 없음 |

PRB 사용률 U(6절), 이웃 평균 부하(최근접 3셀), ISAC 측정 오차(8절)는 Python이 이 값으로 계산한다.

---

## 11. 검증과 이음5G 영향

### 11-1. 구현 검증

| 확인 | 방법 | 결과 |
|---|---|---|
| v2 재현 (데이터셋 모드) | `--slmCarrierFrequency=3.5e9 --o2iLossDb=0`으로 v2 실행 파일과 같은 입력(시나리오 3개, test 첫날)을 실행 | 출력 CSV 3개가 **바이트 단위로 같음** |
| v2 재현 (RL 모드) | 같은 인자로 r10 에이전트와 비교 기준 두 개를 시나리오별 test 첫날(9일)에 평가(1초 판정) | v2 결과와 **9/9일 모든 지표가 같음** |
| 실내 판정 배선 | 모든 사이트를 `indoor=0`으로 바꿔 실행 | 손실 0과 바이트 단위로 같음 (손실은 실내 사이트를 통해서만 들어감) |
| 데이터셋 (TCN 입력) | 이음5G 기본값으로 시나리오 3개 하루치를 실행 | 11개 열 중 10개(UE 수, ISAC, 요구 트래픽, 핸드오버, presence)가 v2와 같고 평균 SINR만 바뀐다(평균 -0.15 dB, 최대 -1.5 dB). 그래서 **데이터셋과 TCN을 다시 만들 필요가 없다** |
| 평가 인자 경로 | 영향 평가의 v2 무선 설정 결과를 기존 v2 평가와 비교 | 매분 지표 기준 45/45일 같음 |
| **코드 정리 (2026-10-21)** | 정리 전 코드로 만든 결과와 정리 후 결과를 같은 입력으로 비교 (아래 목록) | **모두 같음** |

코드 정리 전후 비교 항목은 다음과 같다.
- 데이터셋 원시 CSV (시나리오 2개 x 하루): 로컬과 서버 모두 바이트 단위로 같음
- 데이터셋 생성 (시나리오 3개 x 2일, 그림 포함): npz, CSV, 그림 10개 파일이 같음
- RL 평가 (test 15일 x 기법 3개 = 45일, 매분 판정): 이음5G와 v2 무선 설정 모두 45/45일 같음. 서버에서도 이음5G 45/45일 같음
- RL 평가 (시나리오별 test 첫날 x 기법 3개, 1초 판정): 9/9 같음
- 대시보드 원자료 9개(npz, 경로 CSV, meta.json)와 대시보드 정지 화면, KPI1 출력, TCN 짧은 학습(20 배치)의 지표가 같음
- 데이터셋 모드는 예전 V1 생성의 ISAC 기기 추첨 난수를 값만 버리고 그대로 뽑는다. 같은 RngRun에서 공개한 데이터셋과 같은 값을 내기 위해서다.

### 11-2. 영향 (v2 에이전트 그대로, test 15일, 매분 판정)

- v2에서 학습한 에이전트와 반응형 임계값을 바꾸지 않고, 같은 15일을 두 무선 설정에서 평가해 날짜별로 비교했다.
- 결과 파일: `rl-env_v4/runs/eum5g_impact/eval_test_v2agent_r10_{eum5g,v2radio}.json`

| 에이전트 | v2 무선 설정 | 이음5G |
|---|---|---|
| 절감률 (KPI2) | 36.47% | 36.43% |
| QoS 부족 | 2.220% | 2.195% |
| KPI3 (반응형 대비 감소) | 54.8% | 55.3% |
| 끊김률 (매분) | 0.870% | 0.875% |

- **상시 최대**: 15일 모두 같다.
- **다른 보상 비율**: 함께 확인한 1:5:5와 1:20:5 에이전트도 차이가 절감 0.05%p, QoS 부족 0.03%p, 끊김 0.005%p 이하였다(README_AI.md 9절 참고).
- **영향이 작은 이유**
  - 모든 셀이 항상 송신해서, 셀 경계 단말의 SINR은 이웃 셀 간섭으로 정해진다.
  - 반송파와 침투 손실은 같은 단말의 모든 링크에 같게 들어가므로 사실상 잡음만 키운다.
  - 실내에서도 실효 잡음(약 -69 dBm)이 간섭보다 충분히 작다.
- **결론**: 이음5G 조건에서도 제안 기법의 성능과 KPI 판정이 그대로 유지된다. 전체 결과는 README_AI.md 9절에 있다.

### 11-3. 1초 끊김 판정

- 분 사이 끊김을 1초마다 판정할 수 있다(ns-3 `--rlOutageSamplesPerMin 60`, `evaluate.py --outage-samples 60`). 관측과 보상은 그대로이고, 평가 지표(끊김 시간 비율, 하루 끊김 횟수, 10초 이상 이어진 끊김 횟수)만 더해진다.
- 매초 접속 UE를 모두 다시 계산하므로 정확하지만 느리다. 하루 9개(기법 3개 x 시나리오 3개)를 프로세스 3개로 돌리면 약 15\~16분(907\~950초) 걸린다. 매분 판정의 약 10배다.
- v2 시절의 판정 생략 기법(경로 위 끊김 상태를 미리 증명)은 침투 손실을 반영하지 못해 이음5G에서는 쓰이지 않았고, 코드 정리 때 지웠다.
- 그래서 이음5G 평가의 기본은 매분 판정(`--outage-samples 1`)이다. 1초 판정 결과(시나리오별 test 첫날)는 README_AI.md 9절에 참고로 둔다.

---

## 12. 한계와 가정

**모델 단순화**
1. 항상 LOS, 섀도잉 없음, 모든 셀 항상 송신 [1][3]. 그래서 이음5G 반송파와 침투 손실의 효과가 작게 나온다. LOS 확률과 섀도잉을 넣으면 효과가 커질 수 있다.
2. 건물 침투 손실은 중앙값 하나(16.2 dB)로 고정했다. 실내 비율은 44\~63%로 TR 38.901 가정 80%보다 낮다(이동 모델 때문).
3. 핸드오버는 매분 가장 센 셀로 즉시 전환한다. 3GPP A3 이벤트의 오프셋과 유지 시간은 없다.
4. 스케줄링, MCS, HARQ가 없어 용량은 SINR과 무관하다.
5. 전력은 부하와 무관한 활성 상태 값이다. 그래서 절감률은 상한이다.
6. ISAC 오경보는 반영하지 않았다.

**이음5G 관련**

7. 출력 한도 80 W는 고시 원문(HWP)을 열어 보지 못해 간접 확인이다 [65][66]. 우리 출력(30 dBm)은 한도보다 19 dB 낮아 결론에는 영향이 없다.
8. TDD 하향 비율은 이음5G 고유 근거가 없어 DDDSU를 가정했다. 5:5로 바꾸면 C_max는 약 390 Mbps가 된다(계산값). C_max는 Python 상수(`build_tcn_dataset.py` `CELL_CMAX_MBPS`)라, 이 경우는 별도 민감도 평가가 필요하다(아직 하지 않음).
9. 대학 이음5G 구축의 기술 수치는 공개 사례가 없다(4-1절 "보고서 표현").
10. 28 GHz, 다른 망과의 공존, 상향 링크, 슬롯 단위 동작은 범위 밖이다.

**엔진 스펙 규칙 (팀 공통, 유지)**

11. 커버리지 반경 사다리는 전파 모델보다 완만하다(5절).
12. 용량은 안테나 수에 정비례한다(실측보다 보수적) [33].
13. 1분 해상도, 09\~21시, UE 1,000명, 이웃 3셀, ISAC 오차 8%.

**우리 가정**

14. 1인당 수요(캠퍼스 피크 강부하), 접속 지연 0\~6분, 활동 확률 0.7, 수요 편차 +-30%, 기숙사 저녁 x2, 선행 시차 크기, 일간 변동 12%, 산포 반경 18 m, 실내 판정 규칙.

**권장 보강 (재학습 불필요, 평가만)**: 단열 건물 31.5 dB, 과부하 기준 0.7/0.9, 커버리지 반경 +-20%.

---

## 13. 코드 위치와 실행 인자

| 설정 | 파일 | 이름 |
|---|---|---|
| 반송파, 잡음, 송신 전력 | `scratch/cttc-nr-demo.cc` `SimConfig`, `CoverageModel` | `slmCarrierFrequency`(4.77e9), `bwHz`, `nfDb`, `m_txPowerDbm`, `kRefAntennas` |
| 건물 침투 손실, 실내 판정 | 같은 파일 / `run_campaign.py` | `o2iLossDb`(16.2), `o2iIndoorRadius`(18), `SetBuildingEntryLoss()`, `EntryLossDb()`, `LoadSitePositions()`의 `indoor` 열 / `OUTDOOR_SITES` |
| 빔포밍, 전력 스케일, 간섭 | `CoverageModel` | `SetLink()`, `RxDbm()`, `ServingCellSinrInCoverage()` |
| 커버리지 반경 | `CoverageModel` | `CoverageRadiusM()` |
| 끊김 기준 | `SimConfig` | `rlSinrThresholdDb`(-5) |
| 반경 규칙, 담당 셀 귀속, 서비스 구역 | `RlAntennaEnv` | `ClassifyConnected()` (항상 적용, 인자 없음) |
| 분 사이 끊김 샘플 (평가 지표) | `SimConfig`, `RlAntennaEnv` | `rlOutageSamplesPerMin`, `rlOutageEventSec`, `SubSample()`, `AddOutageSample()` |
| ISAC 반경·검출 확률, 접속 지연, 활동 확률, 수요 편차 | `SimConfig` | `isacSenseRadius`, `v2Pd`, `v2ConnDelayMean`, `v2ActivityProb`, `v2RateCv` |
| C_max, 1인당 수요, 선행 시차, 저녁 배수, ISAC 오차, 이웃 수 | `build_tcn_dataset.py` | `CELL_CMAX_MBPS`, `TYPE_RATE_MBPS`, `TYPE_LAG_MIN`, `DORM_EVENING_MULT`, `ISAC_SIGMA`, `N_NEIGHBORS` |
| 사이트 좌표, 시점, 산포 반경 | `run_campaign.py` | `SITES`, `SLOT_HOURS`, `BUILDING_SCATTER_R` |
| 일간 변동, 날 시드 | `rl-env_v4/campus_env.py` | `day_var`, `TRAIN_SEED_RANGE`, `SPLIT_SEED_BASE` |
| 전력 고정분, 과부하 기준 | `rl-env_v4/ns3_antenna_env.py` | `RewardConfig`, `cell_power()` |

**ns-3 실행 인자** (`./ns3 run "cttc-nr-demo --PrintHelp"`로 전체 목록을 볼 수 있다)

| 구분 | 인자 (기본값) | 의미 |
|---|---|---|
| 모드 | `--dataset`, `--rlEnv` | 데이터셋 모드 / RL 모드. 둘 다 없으면 멈춘다 |
| 입력 파일 | `--sitesFile`, `--waypoints`, `--ratesFile`, `--antennaFile` | 사이트 (x, y, indoor), UE 웨이포인트 (ueId, t, x, y, on), 셀별 수요 (cell, baseRate, eveningMult, lagMin), 셀별 안테나 수(데이터셋 모드) |
| | `--gnbAntennas` (16) | `antennaFile`에 없는 셀의 안테나 수. 데이터셋 생성은 모든 셀을 파일로 16으로 준다 |
| 시간 | `--datasetMinutes` (720) | 하루 분 수. RL 에피소드 길이도 같다 |
| 사람·트래픽 | `--isacSenseRadius` (100), `--v2Pd` (0.9), `--v2ConnDelayMean` (3.0), `--v2ActivityProb` (0.7), `--v2RateCv` (0.3) | 8\~9절. 이름의 `v2`는 데이터셋 생성 방식의 이름에서 왔다 |
| RL | `--rlAntennaInit` (16), `--rlSinrThresholdDb` (-5), `--rlOutageSamplesPerMin` (1), `--rlOutageEventSec` (10), `--rlShmTag` | 시작 안테나 수, 끊김 SINR, 분당 끊김 판정 횟수, 긴 끊김 기준(초), ns3-ai 공유 메모리 이름 접미사 |
| 이음5G 무선 | `--slmCarrierFrequency` (4.77e9), `--o2iLossDb` (16.2), `--o2iIndoorRadius` (18) | 반송파(Hz), 실내 침투 손실(dB, 0이면 끔), 실내 판정 거리(m) |
| 출력, 난수 | `--simTag`, `--outputDir`, `--RngRun` | 데이터셋 출력 `<simTag>-obs.csv`, ns-3 난수 실행 번호 |

평가할 때는 `rl-env_v4/evaluate.py --ns3-arg 이름=값`으로 ns-3 인자를 바꿀 수 있다.
- v2 무선 설정: `--ns3-arg slmCarrierFrequency=3.5e9 --ns3-arg o2iLossDb=0`
- 단열 건물: `--ns3-arg o2iLossDb=31.5`

---

## 14. 환경 정의별 참고문헌 대응표

| 영역 | 설정 | 참고문헌 |
|---|---|---|
| 시뮬레이션 방식 | 시스템 수준 모델(1분), 원형 반경 추상화 | [48][49][50] |
| 배치 | 기지국 간격 67 m(소형셀 밀도) | [3][6] |
| | 기지국 10 m / 단말 1.5 m | [1] |
| | 반경 40\~85 m의 규모 | [6][7][8][9][69] |
| 이음5G 대역 | 반송파 4.77 GHz, n79 | [62][67][70] |
| | 대역폭 100 MHz | [62][73] |
| | 출력 한도 80 W/100 MHz | [65][66][70] |
| | SA 단일 망, 동기 TDD | [63][69][71] |
| 전파 | 경로손실 UMi, 잡음지수 9 dB | [1] |
| | 건물 침투 손실 16.2 / 31.5 dB | [68][69], 비교 [1] |
| | 실내 비율(비교 기준 80%) | [1] |
| | 총 송신 전력 30 dBm | [7][2], 비교 [69] |
| | SINR 계산, full buffer 간섭 | [59][3] |
| | 끊김 SINR -5 dB | [56][36][31] |
| 안테나·커버리지 | 배열 이득, 비서빙 방향 이득 없음 | [13][14][58][59] |
| | 안테나당 출력 일정, 절반당 6 dB 감소 | [10][11] |
| 용량·부하 | C_max | [4], 비교 [5][34] |
| | 용량 정비례(보수적) | [33] |
| | 과부하 기준 0.8 | [15][16][17][25], 대안 [18][19], 표준 [20][21][22][23] |
| | 받는 셀 상한 구조 | [24][21] |
| 전력 | P(A) = 0.3 + 0.7 x A/16 | [10], 참고 [12][35] |
| ISAC | 반경, P_D, 오차, 오경보 | [26][27][28][37][38][39] |
| 사람·트래픽 | 수요 수준, 건물 유형 패턴, 선행 시차, 접속 지연, 활동 확률 | [54][55][40][41][42][30][7][8] |
| 시간·예측·제어 | 1분 관측, 15분 예측, 5분 제어, O-RAN rApp | [29][43][44][45][46][47][32] |
| TCN | 구조, 분위수(핀볼) 손실, MAPE 한정 | [76][77][53] |
| RL | 공간 할인, PPO, GAE, 관련 RL 에너지 절감 연구 | [51][52][78][46][47][48][49][50] |
| 배경 | Massive MIMO·RAN 에너지 효율 개관 | [57][60][61] |
| 이음5G 배경·사례 | 정의, 개방, 대학·해외 캠퍼스 사례, 실증, 4.7 GHz 측정 | [63][64][72][73][74][75] |

---

## 15. 참고문헌

표기: "근거"는 그 문헌이 뒷받침하는 설정이다. "(2차)"는 2차 출처이거나 원문을 열어 보지 못한 문헌이다.

**3GPP, ITU, ETSI, NGMN**
1. 3GPP TR 38.901 V17.0.0, *Study on channel model for frequencies from 0.5 to 100 GHz* (Tables 7.2-1, 7.4.1-1, 7.4.3-2, 7.8-1). https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/17.00.00_60/tr_138901v170000p.pdf  
   근거: 기지국·단말 높이, 잡음지수 9 dB·UMi 경로손실, 실내 비율 80%·O2I 손실 비교, LOS·섀도잉 제외 한계
2. 3GPP TS 38.104 V17.9.0, *NR; Base Station (BS) radio transmission and reception* (§4.4 BS classes). https://www.etsi.org/deliver/etsi_ts/138100_138199/138104/17.09.00_60/ts_138104v170900p.pdf  
   근거: 총 송신 전력의 기지국 등급
3. ITU-R Report M.2412-0 (2017), *Guidelines for evaluation of radio interface technologies for IMT-2020*. https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-M.2412-2017-PDF-E.pdf  
   근거: 기지국 간격 67 m, 모든 셀 항상 송신 간섭 가정
4. ITU-R Report M.2410-0 (2017), *Minimum requirements related to technical performance for IMT-2020 radio interface(s)*. https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-M.2410-2017-PDF-E.pdf  
   근거: C_max의 5G 요구 하향 효율 7.8 b/s/Hz
5. ITU-R Report M.2134 (2008), *Requirements related to technical performance for IMT-Advanced radio interface(s)*. https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-M.2134-2008-PDF-E.pdf  
   근거: 4G 요구 효율 비교
6. 3GPP TR 36.872 V12.1.0, *Small cell enhancements for E-UTRA and E-UTRAN; Physical layer aspects* (Table A.1-1). https://www.3gpp.org/ftp/Specs/archive/36_series/36.872/36872-c10.zip  
   근거: 소형셀 반경·셀 간 거리
7. 3GPP TR 36.814 V9.0.0, *Further advancements for E-UTRA physical layer aspects* (Table A.2.1.1.2-3 pico 전력, A.2.1.3.1 FTP 트래픽 모델). https://www.3gpp.org/ftp/Specs/archive/36_series/36.814/36814-900.zip  
   근거: 피코셀 총 송신 전력 30 dBm, FTP 트래픽 모델
8. 3GPP TR 38.913 V19.0.0, *Study on scenarios and requirements for next generation access technologies* (§6.1 시나리오, Table 6.1.5-1). https://www.etsi.org/deliver/etsi_tr/138900_138999/138913/19.00.00_60/tr_138913v190000p.pdf  
   근거: 실내 핫스팟 간격, 활동 비율 가정 비교
9. 3GPP TR 38.807 §5.2.3, Urban micro 시나리오 (미러). https://itecspec.com/3gpp/38.807/s/5.2.3  
   근거: 도심 micro(대학 캠퍼스) 간격 100 m
10. 3GPP TR 38.864 V18.1.0 (2023), *Study on network energy savings for NR* (§5.1 전력 모델, §6.3.1.2 공간 영역 적응 평가). https://www.3gpp.org/ftp/Specs/archive/38_series/38.864/38864-i10.zip  
   근거: 안테나당 출력 일정·커버리지 4.5\~6.5 dB 감소, 전력 모델 s = 0.3
11. NGMN Alliance, *Green Future Networks: Network Energy Efficiency* v1.1 (Dec 2021), §5.3.3. https://www.ngmn.org/wp-content/uploads/211009-GFN-Network-Energy-Efficiency-1.0.pdf  
   근거: 채널 절반 끄면 커버리지 최대 6 dB 감소
12. NGMN Alliance, *Network Energy Efficiency Phase 2* v1.0 (Oct 2023). https://www.ngmn.org/wp-content/uploads/NGMN_Network_Engergy_Efficiency_Phase2.pdf  
   근거: AAU 정적 전력 비율 56%
13. E. Björnson, J. Hoydis, L. Sanguinetti, *Massive MIMO Networks: Spectral, Energy, and Hardware Efficiency*, Foundations and Trends in Signal Processing 11(3-4), 2017, pp. 185-208. https://github.com/emilbjornson/massivemimobook/raw/master/massive_MIMO_networks.pdf  
   근거: 배열 이득 10log10(A), 비서빙 방향 이득 없음
14. 3GPP TR 37.840 §5.4.4.2, AAS 복합 안테나 패턴 (미러). https://www.itecspec.com/3gpp/37.840/s/5.4.4.2  
   근거: AAS 복합 안테나 패턴
15. A. Albanna, H. Yousefi'zadeh, "Congestion Minimization of LTE Networks: A Deep Learning Approach", *IEEE/ACM Trans. Networking* 28(1), 2020. https://newport.eecs.uci.edu/~hyousefi/publ/cmlteTON.pdf  
   근거: 과부하 기준 U_th 80%
16. E. Pang, "AI and network capacity planning decisions", Ericsson Blog, 2020-12-18 (아카이브). https://web.archive.org/web/2024id_/https://www.ericsson.com/en/blog/2020/12/ai-and-network-capacity-planning-decisions  
   근거: 과부하 기준 U_th 80%
17. Nokia, *Sustainable traffic growth in LTE network*, white paper, 2017. https://onestore.nokia.com/asset/f/200996/  
   근거: 과부하 기준 U_th 80%, 대안 90%
18. A. El Mahjoubi, T. Mazri, N. Hmina, "LTE capacity expansion criteria ...", *Int. J. Online Eng.* 14(11), 2018. https://online-journals.org/index.php/i-joe/article/download/9256/5311/30210  
   근거: 과부하 대안 기준 70%
19. 3GPP RAN2 draft CR R2-2010663 (CMCC, Nokia), "PRB usage for Massive MIMO", Nov 2020 (CMCC: 70%를 과부하 기준으로 언급). https://3gpp.org/FTP/tsg_ran/WG2_RL2/TSGR2_112-e/Inbox/Drafts/  
   근거: 과부하 대안 기준 70%
20. 3GPP TS 28.554 V20.3.0, *Management and orchestration; 5G end to end KPI* (§6.4.10 PrbHighLoadRatio). https://www.3gpp.org/ftp/Specs/archive/28_series/28.554/28554-k30.zip  
   근거: PRB 고부하 임계값은 사업자 설정값
21. 3GPP TS 38.300 V19.4.0, *NR; Overall description; Stage-2* (§15.5 부하 분산). https://www.3gpp.org/ftp/Specs/archive/38_series/38.300/38300-j40.zip  
   근거: 부하 분산, 받는 셀 상한
22. 3GPP TS 38.423 V19.4.0, *NG-RAN; Xn Application Protocol* (§9.2.2.50 Radio Resource Status, §9.2.2.52 Composite Available Capacity). https://www.3gpp.org/ftp/Specs/archive/38_series/38.423/38423-j40.zip  
   근거: 부하·가용 용량 보고
23. 3GPP TS 32.551 V19.0.0, *Energy Saving Management (ESM); Concepts and requirements*. https://www.3gpp.org/ftp/Specs/archive/32_series/32.551/32551-j00.zip  
   근거: 에너지 절감 부하 임계값
24. A. Lobinger, S. Stefanski, T. Jansen, I. Balan, "Load Balancing in Downlink LTE Self-Optimizing Networks", *IEEE VTC 2010-Spring*. https://biblio.ugent.be/publication/976783/file/1026693.pdf  
   근거: 받는 셀 상한 구조
25. F. González-Horta et al., "Mathematical model for the knee of the utilization-response time curve" (2011). https://arxiv.org/abs/1106.2380  
   근거: 사용률-지연 곡선의 무릎 70\~90%
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
31. 3GPP TR 36.942 V16.0.0, *E-UTRA; RF system scenarios* (Table A.1). https://www.etsi.org/deliver/etsi_tr/136900_136999/136942/16.00.00_60/tr_136942v160000p.pdf  
   근거: 끊김 기준 비교 최소 SINR -10 dB
32. O-RAN Alliance SuFG, *Potential Energy Savings Features in O-RAN*, white paper, Jan 2025. https://mediastorage.o-ran.org/white-papers/O-RAN.SuFG.Potential%20Energy%20Savings%20Features%20in%20O-RAN%20white%20paper%202025-01.pdf  
   근거: O-RAN rApp 에너지 절감 구조

**업계 자료, 측정 보고서**
33. Samsung, *Massive MIMO for New Radio*, technical white paper v1.2 (2020). https://images.samsung.com/is/content/samsung/assets/global/business/networks/insights/white-papers/1208_massive-mimo-for-new-radio/MassiveMIMOforNRTechnicalWhitePaper-v1.2.0.pdf  
   근거: 안테나 수 대비 용량(정비례는 보수적)
34. TTA SPG33, *IMT-2020 Evaluation Report* (Dec 2019). https://www.itu.int/dms_pub/itu-r/oth/0a/06/R0A060000960001PDFE.pdf  
   근거: NR 자체평가 스펙트럼 효율 비교
35. G. Auer et al., "How much energy is needed to run a wireless network?", *IEEE Wireless Communications* 18(5), 2011 (EARTH 전력 모델).  
   근거: EARTH 전력 모델 정적 비중
36. S. Moloudi et al. (Ericsson), "Coverage Evaluation for 5G Reduced Capability Devices" (2021), Table 17. https://arxiv.org/pdf/2101.07310  
   근거: NR 데이터 채널 최저 동작점 약 -5.7 dB

**연구 논문**
37. Custance, Khan, Parkinson, Sheng, "Parameter-Efficient Domain Adaption for CSI Crowd-Counting" (2026). https://arxiv.org/pdf/2601.02203  
   근거: ISAC 사람 수 세기 정확도
38. "Modeling Classroom Occupancy using Data of WiFi Infrastructure in a University Campus" (2021). https://arxiv.org/pdf/2104.10667  
   근거: ISAC 오차 비교: 교실 재실 추정
39. Higeta, Ogawa, Murakami, Ohara, Otsuki (NTT), "Outdoor Crowd Flow Estimation Using RSRP from Commercial LTE Base Station" (2025). https://arxiv.org/pdf/2512.10447  
   근거: 실외 군중 추정 범위
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
56. ns-3 소스: `src/lte/model/lte-ue-phy.cc`, `contrib/nr/model/nr-ue-phy.cc` (Qout = -5 dB, Qin = -3.9 dB)  
   근거: 끊김 기준 SINR -5 dB
57. L. Sanguinetti, E. Björnson, J. Hoydis, "Towards Massive MIMO 2.0" (2019). https://arxiv.org/pdf/1904.03406  
   근거: 배경: Massive MIMO 개관
58. T. Bai, R. W. Heath, "Coverage and Rate Analysis for Millimeter Wave Cellular Networks" (2014). https://arxiv.org/abs/1402.6430  
   근거: 비서빙 방향 배열 이득
59. ITU-R Recommendation M.2101-0 (2017), *Modelling and simulation of IMT networks and systems for use in sharing and compatibility studies*. https://www.itu.int/dms_pubrec/itu-r/rec/m/R-REC-M.2101-0-201702-I!!PDF-E.pdf  
   근거: SINR 평가 방식, 간섭 쪽 이득
60. S. Wesemann, J. Du, H. Viswanathan (Nokia Bell Labs), "Energy Efficient Extreme MIMO: Design Goals and Directions", *IEEE Commun. Mag.* 2023. https://arxiv.org/pdf/2301.01119  
   근거: 배경: 대규모 안테나의 에너지 효율
61. D. López-Pérez et al., "A Survey on 5G Radio Access Network Energy Efficiency", *IEEE COMST* 2022. https://arxiv.org/pdf/2101.11246  
   근거: 배경: 5G RAN 에너지 효율 개관

**이음5G, 4.7 GHz 사설망**
62. 과학기술정보통신부, 「5G 특화망 주파수 공급방안」 (2021-06): 4.72\~4.82 GHz 100 MHz(10 MHz x 10), 28.9\~29.5 GHz 600 MHz(50 MHz x 12). 보도 경유 (2차): 비즈워치 2021-06-29 https://news.bizwatch.co.kr/article/industry/2021/06/29/0009 , Netmanias https://www.netmanias.com/ko/private-5g/korea-private-5g/1676/  
   근거: 이음5G 정의와 공급 대역, 반송파 4.77 GHz·대역폭 100 MHz
63. 이음5G 주파수 이용관리 시스템, "이음5G 소개" (SA 기반 구축 지원, 기술기준 개정 연혁). https://spectrummap.kr/eum5gportal/jointuseAnly/about2  
   근거: 이음5G 정의, SA 단일 망
64. Netmanias, "세계 각국의 5G 특화망 주파수 상용화 현황" (2021-10-28 개방). https://www.netmanias.com/ko/post/oneshot/15510/5g-5g-privatre-5g/current-status-of-commercialization-of-private-5g-frequencies-in-countries-around-the-world  
   근거: 이음5G 개방 시점
65. 한국전파진흥협회, 공고 제2021-58호 기술기준 일부개정(안) 행정예고 (안 제4조제10항: 4.7 GHz 80 W/100 MHz, 28 GHz 2 W/400 MHz). https://www.spectrum.or.kr/bbs/board.php?bo_table=notice&wr_id=17 . 보도: Netmanias "5G 특화망 28GHz/4.7GHz 전파출력 지정" https://www.netmanias.com/ko/post/operator_news/15176 (2차: 첨부 HWP 원문 미열람)  
   근거: 송신 출력 한도
66. 국립전파연구원 고시 제2023-22호 (4.7 GHz 출력 조항을 기존 조항으로 인용), HCT 웹진 2023-12 경유. http://hctinsight.com/webzine/webzine/202312/file/kc/kc1.pdf (2차)  
   근거: 송신 출력 한도
67. 3GPP TS 38.101-1, *NR; User Equipment (UE) radio transmission and reception; Part 1: Range 1 Standalone*, Table 5.2-1 (n79: 4,400\~5,000 MHz, TDD). 확인: TS 38.101-1 대역표가 실린 RAN4 draft CR R4-2315778 (2차) https://3gpp.org/ftp/TSG_RAN/WG4_Radio/TSGR4_108bis/Inbox/Drafts/%5B108bis%5D%5B147%5D%20NR_LTE_UAV/Revised%20R4-2315778%20-%20draft%20CR%20to%20TR%2038.101-1%20-%20Introduction%20of%20bands%20n31%20and%20n72.docx  
   근거: 3GPP 대역 n79
68. ITU-R Recommendation P.2109-2 (08/2023), *Prediction of building entry loss* (식 (1)\~(6), Table 1 계수). https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.2109-2-202308-I!!PDF-E.pdf  
   근거: 건물 침투 손실 16.2 dB / 31.5 dB 계산
69. H.-J. Kim, I.-K. Lee, "Separation Distance for Frequency Coexistence between Unsynchronized TDD 5G Private Networks at 4.7 GHz," *Journal of Communications* 18(4), 2023. https://www.jocm.us/2023/JCM-V18N4-250.pdf  
   근거: 4.7 GHz 사설망 평가 파라미터: 셀 반경 50 m, 27 dBm/40 MHz, 안테나 높이, 침투 손실 16.3/31.5 dB, 동기 TDD 공존
70. 유원석, 조승우, 김민준, 이원철, "5G 특화망 조정대상 영역 내 주파수 공동사용을 위한 SVM 기반 자율 능동형 후행 기법 연구," 한국전자파학회논문지 36(4), 2025. https://www.jkiees.org/archive/view_article?pid=jkiees-36-4-380  
   근거: 4.77 GHz/100 MHz, 기지국 49 dBm/100 MHz 시뮬레이션 사례
71. 스마트투데이, 삼성전자 5G 특화망 장비 보도 (상하향 비율 2:8\~5:5 지원). https://www.smarttoday.co.kr/ko-kr/articles/23168 (2차)  
   근거: TDD 비율 가정의 비교
72. LG 보도자료, "LG CNS, 경희대 이음5G 구축" (2022-12). https://www.lg.co.kr/media/release/25926  
   근거: 대학 이음5G 사례
73. Ferguson et al. (Univ. of Edinburgh), "Campus5G: A Campus Scale Private 5G Open RAN Testbed," arXiv:2506.23740, 2025. https://arxiv.org/abs/2506.23740  
   근거: 해외 캠퍼스 사설 5G O-RAN 사례, RU 100 MHz
74. 대한민국 정책브리핑, 이음5G 융합서비스 실증 (480억 원, 11개 과제, 2022). https://www.korea.kr/news/policyNewsView.do?newsId=148902999  
   근거: 이음5G 정부 실증
75. 이상화, 송용수, 조오현, 「철도환경 이음 5G 네트워크 성능 분석」, 아시아태평양융합연구교류논문지 11(2), 2025. https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003176464  
   근거: 4.7 GHz 이음5G 전파 측정 사례 (설정 근거로는 쓰지 않음)

**AI 모델**
76. S. Bai, J. Z. Kolter, V. Koltun, "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling," arXiv:1803.01271, 2018. https://arxiv.org/abs/1803.01271 (2차)  
   근거: TCN 구조: 팽창 인과 합성곱과 잔차 블록 (README_AI)
77. R. Koenker, G. Bassett, "Regression Quantiles," *Econometrica* 46(1), 1978. (2차)  
   근거: 분위수(핀볼) 손실, 중앙값 tau 0.5와 피크 tau 0.9 (README_AI)
78. J. Schulman, P. Moritz, S. Levine, M. Jordan, P. Abbeel, "High-Dimensional Continuous Control Using Generalized Advantage Estimation," *ICLR* 2016. https://arxiv.org/abs/1506.02438 (2차)  
   근거: GAE lambda 0.95 (README_AI)

**팀 내부 문서**
- 최종 공통 엔진 스펙 (대회 제공 자료, 이 저장소에 없음): §0 제어 단위, §2 반경·용량, §3 시간축, §5 TCN 특징, §6 건물 유형 계수, §8 스티어링·U_th, §9 예측, §10\~11 시나리오·인원표
- INCharge 제안서 (이 저장소에 없음): TCN 입력·출력, RL State/Action/Reward, rApp 구조, KPI 1\~3, 비교군

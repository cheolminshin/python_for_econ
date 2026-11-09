# 상경계를 위한 Python 입문 — 실행 결과가 포함된 실습 노트북

첨부된 최신 강의록 `python_economics_handout_revised(1).zip`을 기준으로 만든 섹션별 실습 자료입니다. 각 노트북에는 한국어 설명, 원문의 프로그램, 실제 실행 출력과 그래프가 함께 저장되어 있습니다. 처음 열 때 다시 실행하지 않아도 결과를 읽을 수 있습니다.

## 시작하기

1. ZIP 파일 **전체를 압축 해제**하고 `python_economics_notebooks` 폴더를 VS Code에서 엽니다. 노트북만 다른 폴더로 옮기면 입력자료와 공통 함수를 찾지 못할 수 있습니다.
2. 아래 순서대로 `.ipynb` 파일을 엽니다. 각 노트북은 다른 노트북을 먼저 실행하지 않아도 사용할 수 있습니다.
3. 다시 실행하려면 해당 노트북의 커널을 선택한 뒤 `Restart Kernel`과 `Run All`을 차례로 사용합니다. 같은 노트북 안에서는 위에서 아래로 실행합니다.
4. `data/`는 입력자료, `output/`와 `figures/`는 실행 중 저장되는 결과입니다. 결과 파일은 다시 실행할 때 갱신될 수 있습니다.

## 섹션별 파일

| 순서 | 노트북 | 내용 |
| --- | --- | --- |
| 1 | [section01_python_setup.ipynb](section01_python_setup.ipynb) | Python·VS Code·Jupyter 준비, 첫 프로그램 |
| 2 | [section02_data_types.ipynb](section02_data_types.ipynb) | 기본 자료형, 자료형 변환, 실수 계산 |
| 3 | [section03_data_io.ipynb](section03_data_io.ipynb) | CSV·Excel·JSON·Stata 자료 입출력 |
| 4 | [section04_plotting.ipynb](section04_plotting.ipynb) | 2차원·3차원 그래프와 그림 저장 |
| 5 | [section05_numpy_scipy_functions.ipynb](section05_numpy_scipy_functions.ipynb) | NumPy·SciPy, 함수, 제어문, 옵션가격 격자 |
| 6 | [section06_binomial_option_pricing.ipynb](section06_binomial_option_pricing.ipynb) | CRR, 유럽형·미국형 옵션, CEV·Heston 트리 |
| 7 | [section07_monte_carlo_option_pricing.ipynb](section07_monte_carlo_option_pricing.ipynb) | Euler–Maruyama, 유럽형·Asian 옵션, Heston MC |
| 8 | [section08_pde_option_pricing.ipynb](section08_pde_option_pricing.ipynb) | Explicit 예제, Implicit FDM, local volatility·Heston 연산자 |
| 9 | [section09_option_greeks.ipynb](section09_option_greeks.ipynb) | 해석적·수치적 Greeks, Delta hedge, CEV·Heston |
| 10 | [section10_els_pricing_hedging.ipynb](section10_els_pricing_hedging.ipynb) | 2자산 step-down ELS, 상환확률, Greeks, 별도의 1자산 FDM·헤지 |
| 11 | [section11_cuda_monte_carlo.ipynb](section11_cuda_monte_carlo.ipynb) | NumPy·CuPy OU 채권가격, 해석식 비교, 실행시간 측정 |

## 프로그램을 읽는 방법

- **실행 셀 아래의 출력**은 실제 실행 결과입니다. 함수 정의처럼 원래 출력이 없는 셀에는 별도 출력을 억지로 붙이지 않았습니다.
- **강의록 참고 출력**은 원고의 설명용 출력이며 Markdown으로 구별했습니다. 실행 환경에 따라 소수점 마지막 자리, 자료형 표시, 경로, 실행시간이 다를 수 있습니다.
- 오류를 배우는 예제는 해당 오류만 `try/except`로 처리하여 실제 예외 메시지를 보여줍니다. 따라서 전체 실행을 막지 않습니다.
- 무한반복 예제, 터미널 설치 명령, 일부 함수 내부 발췌와 대화식 `input()` 원문은 설명용 코드로 남겼습니다. 필요한 곳에는 별도로 안전하게 실행되는 예제를 붙였습니다.
- `보충 실습`·`참고 풀이`는 원문 프로그램을 이해하기 위한 추가 자료입니다. 모형의 원래 경로 수·시간 단계 수·격자 설정은 유지했습니다.
- 강의록에 입력파일이 제공되지 않은 Stata 예제는 작은 **합성 실습자료**를 생성하여 실행했습니다. 이 자료는 원자료나 실제 조사자료가 아닙니다.

## CUDA 결과의 범위

제작 환경에는 NVIDIA CUDA GPU가 없어 실제 GPU 계산은 수행하지 못했습니다. Section 11에는 CPU Monte Carlo 가격·표준오차·실행시간, 해석적 가격, 표본 수 비교의 실제 결과를 저장했습니다. GPU 전용 셀은 미실행 이유를 출력하며, GPU 가격·실행시간·속도 향상률을 만들어 넣지 않았습니다.

NVIDIA GPU와 CuPy가 준비된 환경에서 Section 11을 다시 실행하면 GPU 가용성을 확인한 뒤 GPU 코드를 실행합니다. CuPy 설치 명령은 강의록과 동일하게 해당 노트북에 안내되어 있습니다. CPU 학습용 필수 패키지에는 CuPy를 포함하지 않았습니다.

## 다시 실행할 환경 준비

Windows에서 이 폴더를 연 PowerShell 터미널을 기준으로 합니다. 아래 명령은 Python 코드 셀에 입력하지 않습니다.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

VS Code에 Python·Jupyter 확장을 설치하고 노트북의 `Select Kernel`에서 `.venv`의 Python을 선택합니다. 사용한 패키지의 정확한 버전은 `requirements-tested.txt`, 검증 환경과 범위는 `validation/execution_report.json`에 기록했습니다. 실제 검증은 Python 3.12.14에서 수행했습니다.

`pricing_models.py`는 원래 강의록에 포함된 공통 함수입니다. Section 9에서 앞선 장의 함수를 불러올 때 사용하므로 노트북과 같은 폴더에 두세요.

` scripts/regenerate_section10_figures.py `와 `scripts/section10_figure_model.py`도 원본 그대로 함께 제공합니다. Section 10의 노트북에는 동일한 1자산 계산과 그림을 셀로 풀어 넣었습니다. 보조 스크립트의 한글 그림은 맑은 고딕 등 한글 글꼴이 있는 환경에서 사용하세요. 노트북의 그래프는 글꼴 문제를 줄이기 위해 영문 축 이름을 사용했습니다.

## 포함·실행 확인

원문의 코드·터미널 명령·참고 출력 **365개 블록을 모두 반영**했습니다. 각 원문 블록과 노트북 셀의 연결은 `validation/listing_coverage.csv`에 기록되어 있습니다. 모든 코드 셀은 초기 상태의 독립적인 Python 실행 환경에서 순서대로 실행했으며, CUDA 전용 계산의 예외는 위에 명시한 것과 같습니다.

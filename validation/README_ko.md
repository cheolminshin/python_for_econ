# 검증 파일 안내

- `run_validation.py`: 원고에서 함수 정의를 추출하고 주요 점검 40개 실행. GPU와 무한루프는 실행하지 않음.
- `extract_models.py`: Python listing 구문검사와 함수 모듈 생성. 원고 예제의 최상위 실행문은 실행하지 않음.
- `test_results.json`: 환경 버전, 점검별 결과·소요시간.
- `syntax_results.json`: Python-style listing 254개의 구문 분류.
- `latex_build_summary.json`: 실제 그림을 포함한 최종 XeLaTeX·참조·링크 점검.
- `data_integrity.json`: 원본 데이터와 바이트 해시 동일 확인.
- `excel_csv_check.json`: artifact_tool로 XLSX와 CSV를 대조한 결과. pandas 입출력 테스트와는 별도 확인.
- `crr_csv_check.json`: 42개 계약의 CRR 일괄평가.
- `els_figure_metadata.json`: 별도 1자산 그림의 가정·격자·손익·교차검증.
- `els_hedge_trace.csv`, `els_state_curves.csv`: 그림의 원시 수치.
- `check_figure_mc.py`: 같은 일별 감시의 1자산 MC와 그림 FDM을 비교하는 추가 확인.

1자산 MC 교차검증과 XLSX 대조는 자동 점검 40개와 별도로 수행했습니다. 모든 수치는 교육용 모형과 지정한 입력 아래의 계산 결과입니다. `extracted_models/`는 재실행 시 생성됩니다.

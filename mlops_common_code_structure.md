# MLOps 공통 Python 코드 구조 정리

## 목적

모델 종류와 상관없이 공통으로 사용할 MLOps 코드를 분리한다.

GRU, LSTM, XGBoost, Transformer 등 어떤 모델이 들어오더라도 다음 흐름은 동일해야 한다.

```text
학습 실행
성능 평가
MLflow 기록
모델 등록
champion 모델 승격
FastAPI에서 champion 모델 로드
예측 시 모델 버전 기록
```

따라서 모델별 코드는 학습/예측 구현에만 집중하고, MLflow 연동과 모델 운영 코드는 공통 모듈로 관리한다.

## 권장 디렉터리 구조

FastAPI 앱에서는 `mlops`를 단순 유틸이 아니라 하나의 운영 도메인으로 둔다.

```text
app/
  main.py

  core/
    config.py
    logging.py
    exceptions.py

  api/
    v1/
      router.py
      endpoints/
        health.py
        prediction.py
        mlops.py

  domains/
    prediction/
      schemas.py
      services/
        prediction_service.py
        gru_service.py
        xgboost_service.py

    mlops/
      __init__.py
      config.py
      schemas.py
      mlflow_client.py
      model_registry.py
      model_loader.py
      evaluation.py
      training_pipeline.py

  jobs/
    train_model_job.py
    reload_model_job.py

  infra/
    database.py
    storage.py
```

초기 문서에서 제안한 MLOps 도메인 내부 구조는 다음과 같다.

```text
app/domains/mlops/
  __init__.py
  config.py
  mlflow_client.py
  model_registry.py
  model_loader.py
  training_pipeline.py
  evaluation.py
  schemas.py
```

모델별 구현은 별도 위치에 둔다.

예:

```text
app/domains/prediction/services/gru_service.py
app/domains/prediction/services/xgboost_service.py
app/domains/prediction/services/lstm_service.py
```

공통 MLOps 코드는 모델별 서비스에 직접 종속되지 않게 만든다.

## 1. config.py

MLflow와 모델 운영에 필요한 환경 설정을 모은다.

담당 역할:

```text
MLflow Tracking URI
기본 Experiment 이름
기본 champion alias
모델 이름 규칙
자동 승격 기준
```

예시 설정:

```text
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_EXPERIMENT=default-model-serving
MODEL_CHAMPION_ALIAS=champion
MODEL_CANDIDATE_ALIAS=candidate
```

환경별로 값이 달라질 수 있으므로 `.env` 또는 Docker 환경변수에서 읽는다.

## 2. mlflow_client.py

MLflow 기본 연결과 run 기록 기능을 담당한다.

담당 역할:

```text
MLflow tracking uri 설정
experiment 설정
run 시작/종료
params 기록
metrics 기록
tags 기록
artifacts 기록
```

모델별 학습 코드는 직접 `mlflow.set_tracking_uri()`를 호출하지 않고, 이 모듈을 통해 기록한다.

예시 흐름:

```text
configure_mlflow()
start_run()
log_params()
log_metrics()
log_artifacts()
end_run()
```

## 3. model_registry.py

MLflow Model Registry 연동을 담당한다.

담당 역할:

```text
모델 이름 생성
모델 등록
모델 버전 조회
champion/candidate alias 설정
기존 champion metric 조회
신규 모델 승격 여부 판단
```

모델 이름은 일관된 규칙을 사용한다.

규칙:

```text
{model_type}_global
{model_type}_{target_type}_{target_id}
{model_type}_{target_type}_{target_id}_{qualifier_key}_{qualifier_value}
```

실제 예:

```text
classifier_customer_42_region_apac
forecast_store_12_horizon_24
ranker_global
```

alias는 stage 대신 명확한 이름을 사용한다.

```text
champion = 현재 운영 모델
candidate = 운영 후보 모델
archived = 과거 모델
```

## 4. model_loader.py

FastAPI에서 운영 모델을 로드하고 캐싱하는 역할을 담당한다.

담당 역할:

```text
champion 모델 로드
메모리 캐시
모델 재로딩
로드 실패 시 기존 모델 유지
모델 버전 정보 반환
```

기본 로드 방식:

```text
models:/모델명@champion
```

예:

```text
models:/classifier_customer_42_region_apac@champion
```

주의사항:

```text
요청마다 모델을 새로 로드하지 않는다.
앱 시작 시 또는 첫 요청 시 로드하고 메모리에 캐싱한다.
새 champion 배포 후에는 reload API 또는 주기적 캐시 갱신으로 반영한다.
로드 실패 시 운영 중인 기존 모델을 버리지 않는다.
```

## 5. training_pipeline.py

모델별 학습 함수를 공통 MLOps 흐름에 연결한다.

담당 역할:

```text
학습 대상 결정
학습 데이터 범위 결정
모델별 train 함수 호출
평가 함수 호출
MLflow 기록
모델 등록
승격 기준 확인
champion 변경
```

이 모듈은 모델 알고리즘을 직접 구현하지 않는다.

대신 모델별 구현체가 다음 인터페이스를 제공한다고 가정한다.

```text
train_model(context) -> trained_model
evaluate_model(model, validation_data) -> metrics
log_model(model, artifact_path) -> model_uri
predict(model, input_data) -> prediction
```

공통 파이프라인은 이 인터페이스만 호출한다.

## 6. evaluation.py

모델 성능 평가와 승격 기준을 담당한다.

담당 역할:

```text
MAE 계산
RMSE 계산
MAPE 계산
피크 예측 오차 계산
기존 champion과 신규 candidate 비교
최소 성능 기준 확인
```

예시 승격 기준:

```text
신규 모델 MAPE <= 15%
신규 모델 RMSE <= 기존 champion RMSE
검증 데이터 개수 >= 최소 기준
피크 예측 오차 <= 허용 기준
```

승격 기준은 코드에 하드코딩하지 않고 설정값으로 관리한다.

## 7. schemas.py

MLOps 공통 데이터 구조를 정의한다.

예시:

```text
TrainingContext
TrainingResult
EvaluationMetrics
ModelRegistryInfo
ModelLoadResult
PredictionLogPayload
```

포함할 정보:

```text
model_type
target_type
target_id
qualifiers
train_start_at
train_end_at
validation_start_at
validation_end_at
run_id
model_version
model_uri
predicted_at
target_timestamp
predicted_value
actual_value
error_value
```

## FastAPI API 경계

MLOps 운영 API는 모델 학습 구현 자체를 노출하지 않고, 운영 상태와 런타임 캐시 제어를 담당한다.

```text
GET  /api/v1/mlops/models/{model_name}
POST /api/v1/mlops/models/{model_name}/reload
GET  /api/v1/mlops/models/{model_name}/loaded
```

예측 API는 항상 champion 모델을 사용한다.

```text
POST /api/v1/predictions
```

요청마다 모델을 새로 로드하지 않는다. `model_loader.py`가 프로세스 메모리에 모델을 캐싱하고, champion 변경 후에는 reload API 또는 job을 통해 갱신한다.

Compose로 실행하면 FastAPI 서비스도 함께 올라온다.

```text
http://localhost:8000/api/v1/health
http://localhost:8000/docs
```

## 모델별 코드가 담당할 것

모델별 서비스는 알고리즘과 데이터 처리에 집중한다.

담당 역할:

```text
학습 데이터 전처리
모델 구조 정의
모델 학습
모델 파일 저장 형식 정의
모델 로드 후 예측 실행
모델별 하이퍼파라미터 관리
```

모델별 코드가 직접 담당하지 않는 것:

```text
MLflow tracking uri 설정
experiment 생성
champion alias 변경
공통 평가 기준 판단
FastAPI 모델 캐시 정책
운영 모델 버전 기록 형식
```

## FastAPI 연동 흐름

예측 API는 모델명과 champion alias를 기준으로 모델을 로드한다.

예시:

```text
요청: target_type=customer, target_id=42, qualifiers={"region": "apac"}
모델명: classifier_customer_42_region_apac
로드 URI: models:/classifier_customer_42_region_apac@champion
```

예측 응답에는 모델 버전 정보를 포함하는 것을 권장한다.

예:

```json
{
  "prediction": [],
  "model": {
    "name": "classifier_customer_42_region_apac",
    "version": "12",
    "run_id": "abc123",
    "alias": "champion"
  }
}
```

운영 오차를 추적하려면 예측 로그에도 모델 정보를 저장한다.

```text
prediction_id
model_name
model_version
run_id
target_type
target_id
qualifiers
predicted_at
target_timestamp
predicted_value
actual_value
error_value
```

## 자동 학습 흐름

초기에는 기존 스케줄러에서 공통 파이프라인을 호출한다.

```text
APScheduler
→ training_pipeline.run_training_job()
→ 모델별 train/evaluate 호출
→ MLflow 기록
→ candidate 등록
→ 기준 통과 시 champion 승격
```

나중에 필요하면 이 실행부만 Prefect, Airflow, Kubeflow로 교체한다.

교체되어도 다음 요소는 유지된다.

```text
MLflow
Model Registry
model_loader
evaluation 기준
model naming rule
prediction log
```

## 구현 순서

권장 구현 순서는 다음과 같다.

```text
1. config.py
2. mlflow_client.py
3. model_registry.py
4. model_loader.py
5. schemas.py
6. evaluation.py
7. training_pipeline.py
8. 기존 GRU 학습 코드와 연결
9. FastAPI 예측 응답에 model metadata 추가
10. 자동 재학습 스케줄러 연결
```

## 핵심 원칙

```text
모델별 학습 코드는 모델만 책임진다.
MLflow 기록/등록/승격은 공통 MLOps 코드가 책임진다.
FastAPI는 항상 champion 모델을 기준으로 서빙한다.
모든 예측은 어떤 모델 버전이 사용됐는지 추적 가능해야 한다.
오케스트레이션 도구는 나중에 바뀔 수 있으므로 학습 실행부와 MLflow 공통 코드를 분리한다.
```

## 로그 운영

FastAPI 앱 로그는 기본적으로 stdout에 JSON 형식으로 출력한다. Docker, Kubernetes, CloudWatch, Loki 같은 로그 수집기는 이 stdout 로그를 수집한다.

파일 로그가 필요한 환경에서는 다음 값을 켠다.

```text
LOG_TO_FILE=true
LOG_FILE_PATH=logs/app.log
LOG_RETENTION_DAYS=30
```

파일 로그는 매일 서버 로컬 자정 기준으로 회전하고, `LOG_RETENTION_DAYS`보다 오래된 백업 로그는 자동 삭제한다. 컨테이너 운영에서는 파일 로그보다 stdout 수집을 기본값으로 둔다.

운영 기준 시간대는 `APP_TIMEZONE=Asia/Seoul`로 둔다. 앱 로그 timestamp와 앱이 직접 생성하는 이벤트 시간은 한국 시간대가 포함된 timezone-aware ISO 형식으로 남긴다. 파일 로그 회전도 서버 로컬 시간대 기준으로 동작하므로 컨테이너의 `TZ=Asia/Seoul` 설정과 함께 사용한다.

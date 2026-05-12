# MLOps Docker

FastAPI, MLflow, PostgreSQL, MinIO를 기반으로 한 공통 MLOps 운영 골격입니다.

이 저장소는 특정 모델 알고리즘 자체를 구현하기보다, 모델별 trainer/data processor/validator를 꽂아 쓸 수 있는 공통 운영 흐름을 제공합니다.

## 제공 기능

- MLflow experiment tracking
- MLflow model registry 연동
- champion/candidate alias 관리
- champion 모델 serving cache, reload, rollback
- 동기/비동기 training job API
- training job status/retry
- training job Postgres 저장
- prediction log 저장 및 actual update
- scheduled retraining
- drift check API
- notification dispatcher
- webhook notification sink
- MLOps event/audit log
- 운영 상태 조회 API
- scheduler 수동 tick/dry-run API
- notification test API
- app DB schema migration runner

## 실행

예시 환경 파일을 복사합니다.

```bash
cp .env.mlops.example .env.mlops
```

비밀번호와 포트를 필요에 맞게 수정한 뒤 실행합니다.

```bash
docker compose --env-file .env.mlops up -d
```

접속 주소:

- FastAPI: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- MLflow UI: `http://localhost:5000`
- MinIO Console: `http://localhost:9001`

중지:

```bash
docker compose --env-file .env.mlops down
```

## 저장소 설정

기본값은 in-memory입니다. API 프로세스 재시작 후에도 기록을 유지하려면 Postgres 저장소를 사용합니다.

```env
PREDICTION_LOG_STORE=postgres
TRAINING_JOB_STORE=postgres
MLOPS_EVENT_STORE=postgres
```

Postgres 저장소를 쓰면 `APP_DATABASE_URL`이 필요합니다. Compose 환경에서는 `APP_DB_*` 값으로 자동 구성됩니다.

```env
APP_DB_NAME=app
APP_DB_USER=app
APP_DB_PASSWORD=change_me_app_db_password
```

App DB schema는 `app/infra/migrations.py`의 migration runner가 준비합니다.

## 주요 API

모든 API prefix는 `/api/v1`입니다.

### 운영 상태

```bash
curl http://localhost:8000/api/v1/mlops/status
```

확인 가능한 항목:

- 등록된 trainer/data processor/input validator
- training job store 타입
- prediction log store 타입
- event store 타입
- scheduler 활성 상태
- notification sink 설정
- drift 설정

### Training Job

동기 실행:

```bash
curl -X POST http://localhost:8000/api/v1/mlops/training-jobs \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "xgboost",
    "target_type": "global",
    "train_start_at": "2026-01-01T00:00:00",
    "train_end_at": "2026-01-07T00:00:00",
    "validation_start_at": "2026-01-07T00:00:00",
    "validation_end_at": "2026-01-08T00:00:00"
  }'
```

비동기 submit:

```bash
curl -X POST "http://localhost:8000/api/v1/mlops/training-jobs/async?max_attempts=2" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "xgboost",
    "target_type": "global",
    "train_start_at": "2026-01-01T00:00:00",
    "train_end_at": "2026-01-07T00:00:00",
    "validation_start_at": "2026-01-07T00:00:00",
    "validation_end_at": "2026-01-08T00:00:00"
  }'
```

상태 조회:

```bash
curl http://localhost:8000/api/v1/mlops/training-jobs/{job_id}
```

실패 job retry:

```bash
curl -X POST http://localhost:8000/api/v1/mlops/training-jobs/{job_id}/retry
```

Retry는 `failed` 상태이고 `attempts < max_attempts`인 job만 가능합니다.

## 모델 운영

Champion 상태 조회:

```bash
curl http://localhost:8000/api/v1/mlops/models/{model_name}
```

현재 API 프로세스에 로드된 모델 조회:

```bash
curl http://localhost:8000/api/v1/mlops/models/{model_name}/loaded
```

Champion reload:

```bash
curl -X POST http://localhost:8000/api/v1/mlops/models/{model_name}/reload
```

Rollback:

```bash
curl -X POST http://localhost:8000/api/v1/mlops/models/{model_name}/rollback \
  -H "Content-Type: application/json" \
  -d '{"version": "2"}'
```

Rollback API는 MLflow champion alias를 변경한 뒤 현재 API 프로세스의 모델 cache도 강제로 reload합니다.

## Prediction Log

예측:

```bash
curl -X POST http://localhost:8000/api/v1/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "xgboost_global",
    "request_id": "req-1",
    "inputs": [{"x": 1.0}]
  }'
```

Actual update:

```bash
curl -X PATCH http://localhost:8000/api/v1/predictions/req-1/actual \
  -H "Content-Type: application/json" \
  -d '{
    "actual_value": 10.0,
    "error_value": 1.5,
    "error_metrics": {"mape": 12.0}
  }'
```

Prediction log 조회:

```bash
curl "http://localhost:8000/api/v1/predictions/logs?model_name=xgboost_global"
```

## Scheduled Retraining

기본값은 비활성화입니다.

```env
MLOPS_ENABLE_SCHEDULED_RETRAINING=true
MLOPS_SCHEDULED_RETRAINING_JOBS=[{"model_type":"xgboost","interval_seconds":86400,"train_window_hours":168,"validation_window_hours":24,"max_attempts":2}]
```

수동 dry-run:

```bash
curl -X POST http://localhost:8000/api/v1/mlops/scheduler/tick \
  -H "Content-Type: application/json" \
  -d '{"dry_run": true}'
```

수동 tick:

```bash
curl -X POST http://localhost:8000/api/v1/mlops/scheduler/tick \
  -H "Content-Type: application/json" \
  -d '{"dry_run": false}'
```

## Drift Check

최근 prediction log 중 actual이 있는 로그를 기준으로 평균 error 또는 metric threshold를 확인합니다.

```bash
curl "http://localhost:8000/api/v1/mlops/models/xgboost_global/drift?min_samples=10&metric_name=mape&max_mean_metric_value=15.0"
```

기본 설정:

```env
MLOPS_DRIFT_MIN_SAMPLES=30
MLOPS_DRIFT_METRIC_NAME=mape
# MLOPS_DRIFT_MAX_MEAN_ERROR_VALUE=10.0
# MLOPS_DRIFT_MAX_MEAN_METRIC_VALUE=15.0
```

## Notification

기본값은 비활성화입니다.

```env
MLOPS_NOTIFICATION_SINK=disabled
```

로그로 남기기:

```env
MLOPS_NOTIFICATION_SINK=logging
```

Webhook 사용:

```env
MLOPS_NOTIFICATION_SINK=webhook
MLOPS_NOTIFICATION_WEBHOOK_URL=https://example.com/mlops-webhook
```

테스트 이벤트 발송:

```bash
curl -X POST http://localhost:8000/api/v1/mlops/notifications/test \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "notification_test",
    "severity": "info",
    "message": "MLOps notification test",
    "payload": {"source": "manual"}
  }'
```

알림과 audit log에 연결된 주요 이벤트:

- `training_job_succeeded`
- `training_job_failed`
- `scheduled_retraining_submitted`
- `scheduled_retraining_submit_failed`
- `drift_detected`
- `champion_promoted`
- `rollback_completed`
- `model_reload_failed`

## Audit Log

최근 MLOps 운영 이벤트 조회:

```bash
curl http://localhost:8000/api/v1/mlops/events
```

이벤트 타입 필터:

```bash
curl "http://localhost:8000/api/v1/mlops/events?event_type=drift_detected&limit=20"
```

## 실제 모델 연결

현재 `XGBoostTrainer`, `GruTrainer`, `PassthroughTrainingDataProcessor`는 공통 구조를 위한 placeholder입니다. 실제 동작을 위해서는 모델/서비스 도메인에 맞게 아래 구현이 필요합니다.

- `train_model()`
- `evaluate_model()`
- `log_model()`
- `load_training_data()`
- `preprocess()`
- `build_features()`
- `split_validation()`
- prediction input validator

등록 위치는 `app/domains/mlops/bootstrap.py`입니다.

예:

```python
ModelComponentRegistration(
    model_type="xgboost",
    trainer_factory=XGBoostTrainer,
    data_processor_factory=XGBoostTrainingDataProcessor,
    prediction_input_validator_factory=lambda: RequiredFieldsValidator({"x"}),
)
```

## 로컬 검증

현재 환경에 의존성이 설치되어 있다면:

```bash
python3 -m pytest
```

문법/컴파일 확인:

```bash
python3 -m compileall app tests
```

Compose 설정 확인:

```bash
docker compose config --quiet
```

## 주의사항

- 이 저장소는 공통 MLOps 운영 골격입니다.
- 실제 모델 학습/데이터 처리/입력 schema는 서비스 도메인에 맞게 구현해야 합니다.
- 외부 오케스트레이터, feature store, 승인 workflow, 권한/감사 강화는 현재 범위에 포함하지 않습니다.

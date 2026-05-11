# MLOps 초기 인프라 구성

## 목적

FastAPI 앱과 모델 종류에 상관없이 공통으로 사용할 MLOps 기반 인프라와 API 서버를 구성한다.

이 구성은 다음 역할을 담당한다.

```text
FastAPI = 공통 MLOps API와 모델 서빙 엔드포인트
MLflow = 실험 추적, 모델 레지스트리
PostgreSQL = MLflow 메타데이터 저장
MinIO = 모델 파일과 아티팩트 저장
```

서비스 운영 DB와 MLflow DB는 분리한다.

```text
service DB = 서비스 운영 데이터
mlflow DB = MLflow 실험/모델 메타데이터
```

## 추가 파일

```text
docker-compose.yml
Dockerfile
.dockerignore
.env.mlops.example
.env.app.example
mlops_initial_setup.md
mlops_common_code_structure.md
```

Compose 실행 시 FastAPI API와 MLOps 인프라가 함께 올라온다.

## 실행 방법

먼저 예시 환경 파일을 복사해서 비밀번호를 변경한다.

```bash
cp .env.mlops.example .env.mlops
```

그 다음 실행한다.

```bash
docker compose --env-file .env.mlops up -d
```

MLflow UI:

```text
http://localhost:5000
```

FastAPI:

```text
http://localhost:8000
http://localhost:8000/docs
```

MinIO Console:

```text
http://localhost:9001
```

## 앱에서 사용할 값

운영 기준 시간대는 한국 시간으로 맞춘다.

```text
TZ=Asia/Seoul
APP_TIMEZONE=Asia/Seoul
```

FastAPI 앱에서 직접 기록하는 로그와 이벤트 시간은 `APP_TIMEZONE`을 따른다. Docker 기반 MLOps 인프라 컨테이너는 `TZ`를 통해 한국 시간 기준으로 동작하도록 설정한다. 단, MLflow나 일부 외부 라이브러리 내부 메타데이터는 도구 특성상 UTC가 섞일 수 있으므로 조회/표시 단계에서 한국 시간으로 변환하는 것을 원칙으로 한다.

FastAPI 컨테이너는 같은 Compose 네트워크 안에서 MLflow 서비스명으로 접근한다.

```text
MLFLOW_TRACKING_URI=http://mlflow:5000
```

외부 앱이나 다른 Compose에서 접근하면 호스트 포트 또는 공통 Docker 네트워크를 사용한다.

호스트 포트로 접근할 때:

```text
MLFLOW_TRACKING_URI=http://localhost:5000
```

컨테이너에서 호스트를 바라볼 때는 환경에 따라 다음 주소를 사용할 수 있다.

```text
MLFLOW_TRACKING_URI=http://host.docker.internal:5000
```

## 저장 구조

MLflow 메타데이터는 `mlflow-db`의 PostgreSQL에 저장된다.

```text
experiment
run
param
metric
model registry
model version
```

모델 파일과 아티팩트는 MinIO bucket에 저장된다.

```text
s3://mlflow-artifacts/...
```

예시 파일:

```text
model.pt
scaler.pkl
config.json
loss_curve.png
MLmodel
```

## 향후 앱 연동 방향

모델 종류가 GRU든 다른 모델이든 공통 흐름은 동일하다.

```text
1. 학습 Job 실행
2. MLflow run 생성
3. 파라미터, 메트릭, 모델 파일 저장
4. 기준 통과 시 모델 레지스트리에 등록
5. champion 모델을 FastAPI에서 로드
```

초기에는 현재 스케줄러에서 학습 Job을 호출하고, 나중에 필요하면 Airflow, Prefect, Kubeflow 같은 별도 오케스트레이션으로 분리한다.

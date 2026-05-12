# MLOps Next Steps

현재 상태:

- Level 2 골격 완료: MLflow tracking, model registry, champion serving, prediction log.
- Level 3 초입 완료: async training job submit/status/retry, input validator registry, champion rollback API.
- Training job store는 in-memory/postgres 선택 구조로 확장 완료.
- Postgres 사용 시 `TRAINING_JOB_STORE=postgres`와 `APP_DATABASE_URL` 설정이 필요하다.
- Async training job retry 조건 정리 완료: `failed` 상태이고 `attempts < max_attempts`인 job만 수동 retry 가능하다.
- 자동 retry는 아직 보류한다.
- MLOps component 등록 위치 정리 완료: `bootstrap_mlops_components()`에서 trainer, data processor, prediction input validator를 한 곳에서 등록한다.
- Rollback 후 reload 흐름 정리 완료: rollback API에서 champion alias 변경 후 API 프로세스 캐시를 강제 reload한다.
- 스케줄 기반 재학습 골격 완료: `MLOPS_ENABLE_SCHEDULED_RETRAINING`과 `MLOPS_SCHEDULED_RETRAINING_JOBS`로 rolling-window 학습 job을 주기적으로 submit한다.
- Drift 모니터링 골격 완료: prediction log의 actual/error metric 기반으로 threshold 초과 여부를 확인하는 drift API를 제공한다.
- 알림 골격 완료: disabled/logging/webhook sink 기반 공통 notification dispatcher를 만들고 training 실패, drift 감지, rollback 완료 이벤트에 연결한다.

다음에 이어서 할 우선순위:

1. 알림 확장
   - champion 승격, reload 실패, scheduled retraining submit 실패 이벤트까지 연결
   - Slack/Email 등 전용 sink가 필요하면 추가

주의:

- 아직 feature store, 승인 workflow, 권한/감사 로그는 넣지 않는다. 지금 단계에서는 과하다.
- 다음 작업은 알림 이벤트 연결 범위를 넓히거나 실제 모델/데이터 구현을 붙이는 것이 자연스럽다.

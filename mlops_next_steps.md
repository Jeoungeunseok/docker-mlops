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

다음에 이어서 할 우선순위:

1. Drift 모니터링
   - prediction log와 actual update를 기반으로 성능/출력 drift 감지

2. 알림
   - training 실패, drift 감지, champion 변경, rollback 이벤트를 Slack/Email/Webhook 등으로 전송

주의:

- 아직 feature store, 승인 workflow, 권한/감사 로그는 넣지 않는다. 지금 단계에서는 과하다.
- 다음 작업은 drift 모니터링부터 진행하는 것이 자연스럽다.

# Lambda container image: scikit-learn is too large for a zip package.
FROM public.ecr.aws/lambda/python:3.12
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY triage/ ${LAMBDA_TASK_ROOT}/triage/
COPY config/ ${LAMBDA_TASK_ROOT}/config/
COPY model/triage_model.joblib ${LAMBDA_TASK_ROOT}/model/
CMD ["triage.handler.lambda_handler"]

FROM apache/airflow:3.1.8-python3.10

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /opt/airflow/project
COPY --chown=airflow:root src /opt/airflow/project/src
COPY --chown=airflow:root scripts /opt/airflow/project/scripts
COPY --chown=airflow:root dags /opt/airflow/project/dags
COPY --chown=airflow:root docs /opt/airflow/project/docs
COPY --chown=airflow:root README.md /opt/airflow/project/README.md

ENV PYTHONPATH=/opt/airflow/project/src \
    CTF_PROJECT_ROOT=/opt/airflow/project \
    MLFLOW_TRACKING_URI=http://mlflow:5000

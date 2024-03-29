# FlinkSQL with MLFlow and PyTorch 

This repository illustrates how to use an event streaming platform ([Apache Kafka](https://kafka.apache.org)) in combination with a realtime event stream processor ([Apache Flink](https://flink.apache.org)) using a SQL syntax to predict on trained [PyTorch](https://pytorch.org) models which are distributed via a S3-compatible object store ([MinIO](https://min.io)) and tracked/served with a platform for model lifecycle management ([MLflow](https://mlflow.org)).

# Prerequisites:
- Docker installed

# Running:

1. Run the `setup` profile to download the fruit360 dataset and build a Flink image with Python installed

`docker compose --profile setup build`

2. Run the `dataset` container to intialize the shared volume:

`docker compose run dataset`

3. Run the `mlflow` profile to start MLFlow and its dependencies (Minio, PostgreSQL)

`docker compose --profile mlflow up -d`

4. Run the `training` profile to train the PyTorch when mlflow becomes available (check http://localhost:5001 in the browser)

`docker compose --profile training up`

5. Run the `fullstack` profile to run the remainder of the stack (zookeeper, kafka, jobmanager, taskmanager, sqlclient)

`docker compose --profile fullstack up -d`

tail the logs of the model_deployment container to see when it is ready to receive connections:

`docker compose logs -f model_deployment`

if the `model_deployment` container fails to install its dependencies because of a connection reset (unstable WiFi) just rerun the command `docker compose up -d model_deployment`

6. Now that the full stack is running, visit in the browser:

mlflow: http://localhost:5001
model_deployment: http://localhost:5002 (or run something from the invocation.http file)
minio: http://localhost:9000
jobmanager: http://localhost:8091
kafka-ui: http://localhost:9003

7. Go to the kafka-ui in the browser and select the image-stream topic. Produce a message which is a base64 encoded image, e.g.:

`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\n`

Go in the kafka-ui to the output topic and see the result.




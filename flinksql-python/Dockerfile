FROM flink:1.17-scala_2.12-java11

# Add the PyFlink dependencies
RUN apt-get update && apt-get install -y curl git build-essential make libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget llvm libncurses5-dev xz-utils tk-dev liblzma-dev lzma

# Switch to flink user
USER 9999:9999

# Install pyenv
RUN curl https://pyenv.run | bash
ENV PYENV_ROOT="/opt/flink/.pyenv"
ENV PATH="${PYENV_ROOT}/bin:${PATH}"
ENV PATH="${PYENV_ROOT}/shims:${PATH}"
RUN eval "$(pyenv init -)"

# Install Python 3.8
RUN pyenv install 3.8.16
RUN pyenv global 3.8.16
RUN pip install apache-flink==1.17.0

# Add contents of /app directory to PYTHONPATH
ENV PYTHONPATH "${PYTHONPATH}:/sql-python-job-config/app"

# Add additional connectors to lib
RUN curl -o /opt/flink/lib/flink-sql-connector-kafka-1.17.0.jar https://repo.maven.apache.org/maven2/org/apache/flink/flink-sql-connector-kafka/1.17.0/flink-sql-connector-kafka-1.17.0.jar
RUN curl -o /opt/flink/lib/flink-sql-avro-confluent-registry-1.17.0.jar https://repo.maven.apache.org/maven2/org/apache/flink/flink-sql-avro-confluent-registry/1.17.0/flink-sql-avro-confluent-registry-1.17.0.jar
RUN curl -o /opt/flink/lib/flink-sql-avro-1.17.0.jar https://repo.maven.apache.org/maven2/org/apache/flink/flink-sql-avro/1.17.0/flink-sql-avro-1.17.0.jar

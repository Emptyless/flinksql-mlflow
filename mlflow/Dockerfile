FROM ghcr.io/mlflow/mlflow:v2.3.1

RUN apt-get update && apt-get install -y curl git build-essential make libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget llvm libncurses5-dev xz-utils tk-dev liblzma-dev lzma

# Install pip dependencies
RUN pip install psycopg2-binary awscli boto3 virtualenv

# Install pyenv
RUN curl https://pyenv.run | bash
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="${PYENV_ROOT}/bin:${PATH}"
RUN eval "$(pyenv init -)"
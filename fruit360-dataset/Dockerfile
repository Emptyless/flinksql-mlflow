FROM debian:bookworm

RUN apt-get update && apt-get install -y wget

COPY cache /cache
WORKDIR /cache
RUN [ -e "test_X_10.tsv" ] && echo "already exists" || wget -O test_X_10.tsv https://zenodo.org/record/5702887/files/test_X_10.tsv?download=1
RUN [ -e "test_y_10.tsv" ] && echo "already exists" || wget -O test_y_10.tsv https://zenodo.org/record/5702887/files/test_y_10.tsv?download=1
RUN [ -e "train_X_10.tsv" ] && echo "already exists" || wget -O train_X_10.tsv https://zenodo.org/record/5702887/files/train_X_10.tsv?download=1
RUN [ -e "train_y_10.tsv" ] && echo "already exists" || wget -O train_y_10.tsv https://zenodo.org/record/5702887/files/train_y_10.tsv?download=1

WORKDIR /dataset
RUN cp /cache/test_X_10.tsv test_X_10.tsv
RUN cp /cache/test_X_10.tsv test_X_10.tsv
RUN cp /cache/test_y_10.tsv test_y_10.tsv
RUN cp /cache/train_X_10.tsv train_X_10.tsv
RUN cp /cache/train_y_10.tsv train_y_10.tsv

VOLUME /dataset

CMD ["ls"]






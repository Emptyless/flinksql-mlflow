FROM debian:bookworm

RUN apt-get update && apt-get install -y wget

VOLUME /dataset
WORKDIR /dataset
RUN wget -O test_X_10.tsv https://zenodo.org/record/5702887/files/test_X_10.tsv?download=1
RUN wget -O test_y_10.tsv https://zenodo.org/record/5702887/files/test_y_10.tsv?download=1
RUN wget -O train_X_10.tsv https://zenodo.org/record/5702887/files/train_X_10.tsv?download=1
RUN wget -O train_y_10.tsv https://zenodo.org/record/5702887/files/train_y_10.tsv?download=1

CMD ["ls"]






FROM python:3.9-buster

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN mkdir models
COPY src src

ENV PYTHONPATH="${PYTHONPATH}:/app/src"

ENTRYPOINT ["python"]
CMD ["-m", "training"]
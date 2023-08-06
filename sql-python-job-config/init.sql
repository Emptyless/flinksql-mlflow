CREATE FUNCTION predict AS 'model.predict' LANGUAGE PYTHON;

CREATE TABLE ImageStream
(
    log STRING
) WITH (
      'connector' = 'kafka',
      'topic' = 'image-stream',
      'properties.bootstrap.servers' = 'kafka:29092',
      'properties.group.id' = 'image-stream-group',
      'scan.startup.mode' = 'earliest-offset',
      'format' = 'raw'
      );

CREATE TABLE PredictionStream
(
    class       STRING,
    label       STRING,
    probability STRING
) WITH (
  'connector' = 'kafka',
  'topic' = 'output',
  'properties.bootstrap.servers' = 'kafka:29092',
  'properties.group.id' = 'image-stream-group',
  'format' = 'json'
  );


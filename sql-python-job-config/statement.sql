INSERT INTO PredictionStream
SELECT predict_result['class'], predict_result['label'], predict_result['probability']
FROM (SELECT predict(ImageStream.log) as predict_result FROM ImageStream);

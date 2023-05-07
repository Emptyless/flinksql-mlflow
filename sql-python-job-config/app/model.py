import json

from pyflink.table import DataTypes
from pyflink.table.udf import udf
from urllib3 import request


@udf(result_type=DataTypes.MAP(key_type=DataTypes.STRING(), value_type=DataTypes.STRING()))
def predict(base64_encoded_image: str):
    r = request(method="POST", url="http://model_deployment:5002/invocations",
                headers={"Content-Type": "application/json"},
                json={
                    "inputs": {
                        "base64 encoded image": [
                            base64_encoded_image
                        ]
                    }
                })

    if r.status == 200:
        values = json.loads(r.data)
    else:
        values = {
            'predictions': {
                'class': "-1",
                'label': "request failed",
                'probability': "-1"
            }
        }

    return {
        "class": f"{values['predictions']['class']}",
        "label": f"{values['predictions']['label']}",
        "probability": f"{values['predictions']['probability']}"
    }

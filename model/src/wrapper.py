import base64
import os
from io import BytesIO

import mlflow
import torch
from PIL import Image
from mlflow.environment_variables import MLFLOW_DEFAULT_PREDICTION_DEVICE
from torch import softmax
from torchvision import transforms


class ModelWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        if not "CLASSES" in os.environ:
            raise ValueError("CLASSES env var not set. Configure as class1,class2,class3")
        else:
            self.classes = os.environ["CLASSES"].split(",")

        model_path = context.artifacts["model_path"]
        self.device = torch.device(MLFLOW_DEFAULT_PREDICTION_DEVICE.get())
        self.model_ft = mlflow.pytorch.load_model(model_path)
        self.model_ft.eval()
        self.model_ft.to(self.device)
        self.input_size = 224
        self.compose = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, context, input_model):
        print(input_model)
        imgdata = input_model["base64 encoded image"][0]
        img = Image.open(BytesIO(base64.b64decode(imgdata)))
        img = self.compose(img).to(self.device)
        img = torch.reshape(img, (1, 3, self.input_size, self.input_size))
        res = self.model_ft(img)
        idx = torch.argmax(res, dim=1).item()
        label = self.classes[idx]
        return {
            "class": idx,
            "label": label,
            "probability": softmax(res, dim=1).tolist()[0][idx]
        }

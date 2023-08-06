from __future__ import division
from __future__ import print_function

import copy
import math
import multiprocessing
import os
import time
import uuid

import cv2
import mlflow
import numpy as np
import torch
import torchvision
from mlflow.models import ModelSignature
from mlflow.types import Schema, ColSpec
from torch import nn, optim
from torchvision import models, transforms, datasets

from wrapper import ModelWrapper

if "MLFLOW_DEFAULT_PREDICTION_DEVICE" in os.environ:
    device = torch.device(os.environ["MLFLOW_DEFAULT_PREDICTION_DEVICE"])
elif torch.has_cuda:
    device = torch.device("cuda:0")
elif torch.has_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    torch.set_num_threads(multiprocessing.cpu_count() - 1)


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)
            if phase == 'train':
                mlflow.log_metric('train_loss', epoch_loss, epoch)
                mlflow.log_metric('train_acc', float(epoch_acc), epoch)
            if phase == 'val':
                mlflow.log_metric('val_loss', epoch_loss, epoch)
                mlflow.log_metric('val_acc', float(epoch_acc), epoch)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    mlflow.log_metric('training_time', time_elapsed)
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    mlflow.log_metric('best_val_acc', float(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(num_classes, feature_extract, use_pretrained=True):
    model_ft = models.squeezenet1_0(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model_ft.num_classes = num_classes
    input_size = 224

    return model_ft, input_size


if __name__ == "__main__":
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    if not "MLFLOW_TRACKING_URI" in os.environ:
        mlflow.set_tracking_uri("http://localhost:5001")
    if not "AWS_ACCESS_KEY_ID" in os.environ:
        os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
    if not "AWS_SECRET_ACCESS_KEY" in os.environ:
        os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
    if not "MLFLOW_S3_ENDPOINT_URL" in os.environ:
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"

    # Top level data directory. Here we assume the format of the directory conforms
    #   to the ImageFolder structure
    if "DATA_DIR" in os.environ:
        data_dir = os.environ["DATA_DIR"]
    else:
        data_dir = "./dataset"

    if "CLASSES" in os.environ:
        classes = os.environ["CLASSES"].split(",")
    else:
        classes = ["Apple_Red_Delicious", "Banana", "Corn", "Orange", "Peach", "Pepper_Green", "Pineapple",
                   "Potato_White",
                   "Strawberry", "Tomato_1"]

    if "MODEL_DIR" in os.environ:
        model_dir = os.environ["MODEL_DIR"]
    else:
        model_dir = "./models"

    model_name = "squeezenet"

    # Number of classes in the dataset
    num_classes = len(classes)

    # Batch size for training (change depending on how much memory you have)
    batch_size = 32

    # Number of epochs to train for
    num_epochs = 1

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = False

    # Initialize the model for this run
    model_ft, input_size = initialize_model(num_classes, feature_extract, use_pretrained=True)

    # Print the model we just instantiated
    print(model_ft)

    # create dir if not exists and convert tsv files back to jpg
    for dir in ["train", "val"]:
        dataset_dir = os.path.join(data_dir, dir)
        if not os.path.exists(dataset_dir):
            os.mkdir(dataset_dir)

            stage = "train" if dir == "train" else "test"
            with open(os.path.join(data_dir, f"{stage}_y_10.tsv"), 'r') as f:
                lines = f.readlines()[1:]
                labels = [line.split("\t")[0] for line in lines]
                filenames = [os.path.basename(line.split("\t")[1]) for line in lines]

            # 3x100x100 (RGB * Width * Height)
            with open(os.path.join(data_dir, f"{stage}_X_10.tsv"), 'r') as f:
                lines = f.readlines()
                for num, line in enumerate(lines):
                    imgdata = np.empty([100, 100, 3])
                    values = line.split("\t")
                    for i, value in enumerate(values):
                        x = math.floor((i % 300) / 3)
                        y = math.floor(i / 300)
                        rgb = i % 3
                        imgdata[x][y][rgb] = int(value)

                    img_dir = os.path.join(dataset_dir, labels[num])
                    if not os.path.exists(img_dir):
                        os.mkdir(img_dir)
                    cv2.imwrite(os.path.join(img_dir, filenames[num]), imgdata)

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in
        ['train', 'val']}

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    with mlflow.start_run() as run:
        mlflow.log_param('dataset', "fruit360")
        mlflow.log_param('dataset_paper', "https://arxiv.org/abs/1712.00580")
        mlflow.log_param('model_name', model_name)
        mlflow.log_param('num_classes', num_classes)
        mlflow.log_param('batch_size', batch_size)
        mlflow.log_param('num_epochs', num_epochs)
        mlflow.log_param('feature_extract', feature_extract)
        mlflow.log_param('pretrained', True)

        # Setup the loss fxn
        criterion = nn.CrossEntropyLoss()

        # Train and evaluate
        model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

        # Save and log model
        model_path = os.path.join(model_dir, f"{model_name}_{str(uuid.uuid4())[0:6]}")
        mlflow.pytorch.save_model(model_ft, model_path)

        artifacts = {
            "model_path": model_path,
        }

        with open(os.path.join(os.path.dirname(__file__), "input_example.txt"), 'r') as f:
            base64_encoded_image = f.read()

        input_schema = Schema([ColSpec("string", "base64 encoded image")])
        input_example = {"base64 encoded image": base64_encoded_image}
        output_schema = Schema(
            [ColSpec("integer", "class"), ColSpec("string", "label"), ColSpec("float", "probability")])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        mlflow.pyfunc.log_model(artifact_path="models", python_model=ModelWrapper(), artifacts=artifacts,
                                registered_model_name=model_name, signature=signature,
                                input_example=input_example, code_path=["./src/wrapper.py"],
                                pip_requirements=os.path.join(model_path, "requirements.txt"))

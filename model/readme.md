# Model

The `src/training.py` code is the result of following the [finetuning_torchvision_models_tutorial](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html). Minor modifications are made to store metrics and parameters to MLFlow:

- See the `mlflow.log_param` invocations
- Logging of performance per epoch:
```python
if phase == 'train':
    mlflow.log_metric('train_loss', epoch_loss, epoch)
    mlflow.log_metric('train_acc', float(epoch_acc), epoch)
if phase == 'val':
    mlflow.log_metric('val_loss', epoch_loss, epoch)
    mlflow.log_metric('val_acc', float(epoch_acc), epoch)
```
- Saving the model artifacts to Minio and logging the model to MLFlow
- Wrapping the model in a `src/wrapper.py` for the mlflow serve functionality


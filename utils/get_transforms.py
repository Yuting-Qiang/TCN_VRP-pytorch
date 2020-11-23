import torchvision.transforms as transforms
import numpy as np

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def get_train_transform():

    return transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std = std)
    ])

def get_test_transform():

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

class Target_Transformer():

    def __init__(self):
        pass

    def __call__(self, input):
        return np.array((input>0), dtype=np.float32)


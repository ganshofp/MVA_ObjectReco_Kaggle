import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models
import torch.nn as nn

nclasses = 20 

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,32,kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(64,128,kernel_size=3, stride=1, padding=1, bias=True)
        self.fc1 = nn.Linear(8192,50)
        self.fc2 = nn.Linear(50,nclasses)
        self.dp1 = nn.Dropout(p=0.1)
        self.dp2 = nn.Dropout(p=0.1)

    def forward(self, x):
        
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = dp1(x)
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = dp2(x)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = x.view(-1,8192)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return x


class ResNext(nn.Module):
    def __init__(self):
        super(ResNext, self).__init__()
        self.model = models.resnext50_32x4d(pretrained=True)
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, nclasses)
        

    def forward(self, x):
        return self.model(x)
    

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        set_parameter_requires_grad(self.model, feature_extracting=True)
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, 256)
        self.fc1 = nn.Linear(256,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,20)
        self.dp1 = nn.Dropout(p=0.4)
        
        
        

    def forward(self, x):
        x = F.relu(self.model(x))
        x = self.dp1(x)
        x = F.relu(self.fc1(x))
        x = self.dp1(x)
        x = F.relu(self.fc2(x))
        
        return self.fc3(x)

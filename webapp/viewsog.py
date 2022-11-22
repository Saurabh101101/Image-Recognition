from distutils.command.config import config
import enum
from fileinput import filename
from typing import final
from django.shortcuts import render,HttpResponse
from django.core.files.storage import FileSystemStorage
import torch
#PYTORCH
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable
from torchvision.models import squeezenet1_1
import torch.functional as F
from io import open
import os
from PIL import Image
import pathlib
import glob
import cv2




# Create your views here.
def home(request):
    context={'a':1}
    return render(request,"home.html")
def about(request):
    return render(request,"about.html")



def predictImage(request):
    print(request)
    print(request.POST.dict())
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
  
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)
    context={'filePathName':filePathName}
    train_path = 'D:/WORK/Project/Final/train'
    pred_path = 'media'

    # initial="."+filePathName
    # final="media/predict_img.jpg"
    # os.rename(initial,final)

    # categories
    root = pathlib.Path(train_path)
    classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
    # CNN Network
    


    class ConvNet(nn.Module):
        def __init__(self, num_classes=6):
            super(ConvNet, self).__init__()

            # Output size after convolution filter
            # ((w-f+2P)/s) +1

            

            self.conv1 = nn.Conv2d(
                in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
            
            self.bn1 = nn.BatchNorm2d(num_features=12)
            
            self.relu1 = nn.ReLU()
            

            self.pool = nn.MaxPool2d(kernel_size=2)
            # Reduce the image size be factor 2
            

            self.conv2 = nn.Conv2d(
                in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
            
            self.relu2 = nn.ReLU()
            

            self.conv3 = nn.Conv2d(
                in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
            
            self.bn3 = nn.BatchNorm2d(num_features=32)
            
            self.relu3 = nn.ReLU()
            

            self.fc = nn.Linear(in_features=32*75*75, out_features=num_classes)

            # Feed forwad function

        def forward(self, input):
            output = self.conv1(input)
            output = self.bn1(output)
            output = self.relu1(output)

            output = self.pool(output)

            output = self.conv2(output)
            output = self.relu2(output)

            output = self.conv3(output)
            output = self.bn3(output)
            output = self.relu3(output)

            # Above output will be in matrix form, with shape (256,32,75,75)

            output = output.view(-1, 32*75*75)

            output = self.fc(output)

            return output


    checkpoint = torch.load('D:/WORK/Project/Final/best_checkpoint.model')
    model = ConvNet(num_classes=6)
    model.load_state_dict(checkpoint)
    model.eval()
    # Transforms
    transformer = transforms.Compose([
        transforms.Resize((150,150)),
        transforms.ToTensor(),  # 0-255 to 0-1, numpy to tensors
        transforms.Normalize([0.5, 0.5, 0.5],  # 0-1 to [-1,1] , formula (x-mean)/std
                            [0.5, 0.5, 0.5])
    ])
    # prediction function


    def prediction(img_path, transformer):

        image = Image.open(img_path)

        image_tensor = transformer(image).float()

        image_tensor = image_tensor.unsqueeze_(0)

        if torch.cuda.is_available():
            image_tensor.cuda()

        input = Variable(image_tensor)

        output = model(input)

        index = output.data.numpy().argmax()

        pred = classes[index]

        return pred


    images_path = glob.glob(pred_path+'/*.jpg')

    pred_dict=''

    
        

    for i in images_path:
        pred_dict = prediction(i, transformer)
    context={ 'mynum': pred_dict}
    
    if pred_dict=="aphids":
            return render(request,"Aphids.html",context)
            
        
    elif pred_dict=="Thrips":
            return render(request,"Thrips.html",context)
        

    elif pred_dict=="Leaf hopper":
            return render(request,"LeafHopper.html",context)
        

    elif pred_dict=="Leaf roller":
            return render(request,"LeafRoller.html",context)
        
    
    elif pred_dict=="Stem weevil":
            return render(request,"StemWeevil.html",context)
        

    elif pred_dict=="Whitefly":
            return render(request,"whitefly.html",context)
        
    
    else:
            return render(request,"default.html",context)
            

    
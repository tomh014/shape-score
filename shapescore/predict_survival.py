import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import pandas as pd
import cv2
import numpy as np 
import shutil
import os
from tqdm import tqdm



import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory 


def load_model(model_path, num_classes):
    model = resnet18()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model


def get_image_scores(path,device, dataloaders, bf, image_datasets):
    #NOTE: 1=alive, 2=dead
    
    model  = load_model(path, 2)
    model.to(device)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    
    score2_softmax=[]
    score1_softmax=[]
    
    samples=image_datasets[bf].samples

    with torch.no_grad():
        for (inputs,labels) in tqdm(dataloaders[bf], leave=True,position=0):
            inputs = inputs.to(device)

            outputs = model(inputs)
            score2_softmax.append(float(outputs.cpu().softmax(1).numpy()[0][1]))
            score1_softmax.append(float(outputs.cpu().softmax(1).numpy()[0][0]))
    
    filename=[x[0].split('\\')[-1] for x in samples]
    res = pd.DataFrame({'1_softmax':score1_softmax, '2_softmax':score2_softmax, 'filename':filename})
    return res
        

def sum_channels(img):
    summed_img = img[:,:,0]+img[:,:,1]+img[:,:,2]
    return summed_img

def sig(x):
    return 1/(1 + np.exp(-x))

def make_mask_df(in_dir):
    mask_data = {'time':[],
                 'well':[],
                 'oc':[],
                 'plate':[],
                 'path':[]}

    for f in os.listdir(in_dir):
        mask_id=f.replace('.jpg','')
        l = mask_id.split('_')
        mask_data['time'].append(l[0])
        mask_data['plate'].append(l[1])
        mask_data['well'].append(l[2])
        mask_data['oc'].append(l[3])
        mask_data['path'].append(in_dir+'/'+f)
    df = pd.DataFrame(mask_data)
    df['filename']=[x.split('/')[-1] for x in df['path']]
    return df





def predict_survival(project_id , model='model/combined_5-21.combined_model_5-20-25.deadalive.pth',
                     bf='Brightfield2and4ul'):

    res_folder = 'project/'+project_id
    os.makedirs(res_folder+'/res',exist_ok=True)
    
    data_transforms = {
        bf: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    
    data_dir = res_folder
    image_datasets = {x: datasets.ImageFolder(data_dir+'/processed_mask/mask_raw',
                                              data_transforms[x])
                      for x in [bf]}
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                                 shuffle=False, num_workers=1)
                  for x in [bf]}
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    res = get_image_scores(model, device, dataloaders, bf, image_datasets) 
    
    
    
    maskdf = make_mask_df(data_dir+'/processed_mask/mask_raw/mask_raw')
    maskdf['filename']=[x.split('/')[-1] for x in maskdf['path']]
    
    os.makedirs(data_dir+'/res',exist_ok=True)
    maskdf['alive_softmax']=list(res.set_index('filename').loc[list(maskdf['filename'])]['1_softmax'])
    maskdf['dead_softmax']=list(res.set_index('filename').loc[list(maskdf['filename'])]['2_softmax'])
    maskdf.to_csv(data_dir+'/res/maskpreds.csv')

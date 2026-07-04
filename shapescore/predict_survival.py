import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.transforms import transforms
import pandas as pd
from .utils import make_mask_df
import os
from tqdm import tqdm
from torchvision import datasets



def load_model(model_path, num_classes):
    model = resnet18()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model


def get_image_scores(path,device, dataloaders, image_datasets):
    #NOTE: 1=alive, 2=dead
    model  = load_model(path, 2)
    model.to(device)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    
    score2_softmax=[]
    score1_softmax=[]
    
    samples=image_datasets[0].samples

    with torch.no_grad():
        for (inputs,labels) in tqdm(dataloaders[0], leave=True,position=0):
            inputs = inputs.to(device)

            outputs = model(inputs)
            score2_softmax.append(float(outputs.cpu().softmax(1).numpy()[0][1]))
            score1_softmax.append(float(outputs.cpu().softmax(1).numpy()[0][0]))
    
    paths2=[x[0].replace('/','').replace('\\','') for x in samples]
    filename=[x.split('mask_raw')[-1] for x in paths2]
    res = pd.DataFrame({'1_softmax':score1_softmax, '2_softmax':score2_softmax, 'filename':filename})
    return res
        

def predict_survival(proj_dir , model='model/combined_5-21.combined_model_5-20-25.deadalive.pth'):

    in_dir=proj_dir+'/processed_mask/mask_raw/mask_raw'
    os.makedirs(proj_dir+'/res',exist_ok=True)
    
    data_transforms = {
        0: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    

    image_datasets = {0: datasets.ImageFolder(os.path.abspath(os.getcwd())+'/'+proj_dir +'/processed_mask/mask_raw',
                                              data_transforms[0])}
    dataloaders = {0: torch.utils.data.DataLoader(image_datasets[0], batch_size=1,
                                                 shuffle=False, num_workers=1)}
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print('Predicting survival...')
    res = get_image_scores(model, device, dataloaders, image_datasets) 
    
    
    
    maskdf = make_mask_df(in_dir,proj_dir)
    
    os.makedirs(proj_dir+'/res',exist_ok=True)
    maskdf['alive_softmax']=list(res.set_index('filename').loc[list(maskdf['filename'])]['1_softmax'])
    maskdf['dead_softmax']=list(res.set_index('filename').loc[list(maskdf['filename'])]['2_softmax'])
    maskdf.to_csv(proj_dir+'/res/maskpreds.csv')

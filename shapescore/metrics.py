import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
from .utils import make_mask_df, sum_channels


"""
NAMING CONVENTION ***pay attention to this when converting ND2 files***
timepoint_plateid_well_opticalconfig_maskno
"""


def get_metrics(mask_df): #dfs should have same wells
    """
    get stats for every single well. If n bf worms detected > n sytox worms,
    fluorescent worms are imputed with the std dev and mean of the detected
    fluorescent worms to match the number of bf worms. If n bf < n sytox,
    bf data is not used
    """
    res= mask_df.copy().reset_index()
    size=[];intensity=[];std=[];med=[];ar=[]
    for ix in tqdm(res.index,leave=True):
        path = res.loc[ix]['path']
        im = cv2.imread(path)
        im2 = sum_channels(im)
        aspect_ratio=im2.shape[1]/im2.shape[0]
        im2 = im2[im2 > 0]
        intensity.append(np.mean(im2))
        std.append(np.std(im2))
        med.append(np.median(im2))
        size.append(len(im2))   
        ar.append(aspect_ratio)
    res['size']=size;res['intensity_avg']=intensity;res['std']=std
    res['intensity_med']=med
    
    return res




def metrics(project, upper=600,lower=100):
    in_fol = 'project/'+project+'/processed_mask/mask_raw/mask_raw'
    proj_dir='project/'+project
    
    mask_df = make_mask_df(in_fol, proj_dir)
    print('Calculating metrics...')
    mask_metrics = get_metrics(mask_df)

    
    preds=pd.read_csv(proj_dir+'/res/maskpreds.csv')
    mask_metrics.drop(columns=['index']).to_csv(proj_dir+'/res/mask_metrics.csv',index=False)
    mask_metrics['filename']=[x.split('/')[-1] for x in mask_metrics['path']]
    mask_metrics=mask_metrics.set_index('filename')
    preds=preds.set_index('filename')
    full=pd.concat([preds,mask_metrics],axis=1)
    full['project']=[project]*len(full)
    full['count']=[1]*len(full)
    full.to_csv(proj_dir+'/res/res.csv',index=False)

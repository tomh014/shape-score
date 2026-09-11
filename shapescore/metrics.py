import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
from .utils import make_mask_df, sum_channels


def get_metrics(mask_df): 

    res= mask_df.copy().reset_index()
    size=[];intensity=[];std=[];ar=[]
    for ix in tqdm(res.index,leave=True):
        path = res.loc[ix]['path']
        im = cv2.imread(path)
        im2 = sum_channels(im)
        aspect_ratio=im2.shape[1]/im2.shape[0]
        im2 = im2[im2 > 0]
        intensity.append(np.mean(im2))
        std.append(np.std(im2))
        size.append(len(im2))   
        ar.append(aspect_ratio)
    res['size']=size;res['intensity_avg']=intensity;res['std']=std
    
    return res




def metrics(proj_dir, project, upper=600,lower=100):
    in_fol = proj_dir+'/processed_mask/mask_raw/mask_raw'
   
    mask_df = make_mask_df(in_fol, proj_dir)
    print('Calculating metrics...')
    mask_metrics = get_metrics(mask_df)

    
    preds=pd.read_csv(proj_dir+'/res/maskpreds.csv',index_col=0)
    mask_metrics.drop(columns=['index']).to_csv(proj_dir+'/res/mask_metrics.csv',index=False)
    mask_metrics['filename']=[x.split('/')[-1] for x in mask_metrics['path']]
    mask_metrics=mask_metrics.set_index('filename')
    preds=preds.set_index('filename')
    full=preds.merge(mask_metrics)
    full['project']=[project]*len(full)
    full['count']=[1]*len(full)
    full.to_csv(proj_dir+'/res/res.csv',index=False)

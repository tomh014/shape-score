import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import shutil


project='combined_model_5-20-25'

def makeLayoutDf(project_dir):
    files =os.listdir(project_dir+'/plate_layouts')
    dfs=[pd.read_csv(project_dir+'/plate_layouts/'+f) for f in files]
    plates=[s.replace('.csv','') for s in files]
    for n, df in dict(zip(plates,dfs)).items():
        df['plate']=[n]*len(df)
    layout_all = pd.concat(dfs).set_index(['well','plate'])
    return layout_all
    
    
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
    return df

#################
#thresholded 6 and 10 hour training set
#################
prefix='combined_5-21'

out_dir='model/project/'+project+'/'+prefix
os.makedirs(out_dir+'/data/train/train/alive', exist_ok=True);os.makedirs(out_dir+'/data/train/train/dead', exist_ok=True)

masks=pd.DataFrame()

proj_dir='../../project'
in_dir=proj_dir+'/2505041600bl_hs/processed_mask/mask_raw/mask_raw'
maskinfo=make_mask_df(in_dir)
maskinfo['project']=[in_dir.split('/')[2]]*len(maskinfo)
masks=pd.concat([masks,maskinfo],axis=0)

in_dir=proj_dir+'/2505031600bl_hs/processed_mask/mask_raw/mask_raw'
maskinfo=make_mask_df(in_dir)
maskinfo['project']=[in_dir.split('/')[2]]*len(maskinfo)
masks=pd.concat([masks,maskinfo],axis=0)

in_dir=proj_dir+'/2505111600bl_hs/processed_mask/mask_raw/mask_raw'
maskinfo=make_mask_df(in_dir)
maskinfo['project']=[in_dir.split('/')[2]]*len(maskinfo)
masks=pd.concat([masks,maskinfo],axis=0)

#set n (two half plates get about 5000 worms, which will be the smallest dataset size, normalizing size of all models from here)
#select timepoints
alive_plates=['6hrhsCu','6hrhsNocu']
dead_plates=['10hrhsCu','12hrhsCu']
maskdf=masks.set_index('plate').loc[alive_plates+dead_plates].reset_index()

labels=np.array([0]*len(maskdf))
labels[[x in alive_plates for x in maskdf['plate']]]=1
labels[[x in dead_plates for x in maskdf['plate']]]=0
maskdf['label']=labels


#older project
alive_plates=['4hrhsCu']
dead_plates=['14hrhsCu']
in_dir=proj_dir+'/2503251800bl_hs_4ul/processed_mask/mask_raw/mask_raw'
masks=make_mask_df(in_dir)
masks['project']=[in_dir.split('/')[2]]*len(masks)
masks=masks.set_index('plate').loc[alive_plates+dead_plates].reset_index()

labels=np.array([0]*len(masks))
labels[[x in alive_plates for x in masks['plate']]]=1
labels[[x in dead_plates for x in masks['plate']]]=0
masks['label']=labels

maskdf=pd.concat([maskdf,masks],axis=0).reset_index()

print('class alive: '+str(sum(maskdf['label'])))
print('class dead: '+str(len(maskdf)-sum(maskdf['label'])))
maskdf.to_csv(out_dir+'/mask_info.csv')


maskdf_balanced=maskdf

for ix in tqdm(maskdf_balanced.index):
    label = 'alive' if maskdf_balanced.loc[ix]['label']==1 else 'dead'
    in_path=maskdf_balanced.loc[ix]['path']
    proj=maskdf_balanced.loc[ix]['project'].split('_')[0]
    filename=in_path.split('/')[-1].replace('mask',proj)
    out_path=out_dir+'/'+'data/train/train/'+label+'/'+filename
    shutil.copy(in_path, out_path)
    



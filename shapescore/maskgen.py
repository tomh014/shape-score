"""
from
https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb
"""
import os
import torch
import numpy as np
import cv2
from PIL import Image, ImageFile
from tqdm import tqdm
import pandas as pd
import glob
import gc
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor



def bbox2(img): #https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def rm(path):
    try:
        os.remove(path)
    except:
        pass
    

def predict_segmentation(proj_dir, sam2_checkpoint = "model/sam2_hiera_tiny.pt",bf=None,
                  upper=600,lower=50,batch_size=4, save_tiff=False):
    
    print('Generating masks...')
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    
    in_folder = proj_dir+'/input_raw'
    indir=proj_dir+'/processed_mask'


    out_folder =indir+'/mask_raw/mask_raw';out_folder2 =indir+'/mask_raw_uncropped/mask_raw_uncropped'
    os.makedirs(indir,exist_ok=True)
    os.makedirs(out_folder,exist_ok=True);
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    
    model_cfg = "configs/sam2/sam2_hiera_t.yaml"
    
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    
    if bf == None:
        files=os.listdir(in_folder)
    else:
        files = [x  for x in os.listdir(in_folder) if bf in x]
    order_df=pd.DataFrame({'img':files})
    order_df.to_csv(indir+'/maskgen_order.csv')
    
    
    #make new filelist to start from point where left off
    made_masks = os.listdir(indir+'/mask_raw/mask_raw')
    made_well_imgs=list(set([x.split('_mask')[0] for x in made_masks]))
    all_well_imgs = files
    unmasked_well_imgs = list(set(all_well_imgs)-set(made_well_imgs))
    #delete masks from last image done in case it wasn't completed
    
    list_of_files = glob.glob(indir+'/mask_raw/mask_raw/*') # * means all if need specific format then *.csv
    try:
        latest_file = max(list_of_files, key=os.path.getctime)
        latest_file=latest_file.split('/')[-1]
    except:
        latest_file=''
    if latest_file!='':
        latest_well_im=latest_file.split('_mask')[0]
        del_masks=[x for x in made_masks if x.split('_mask')[0]==latest_well_im]
        for mask in del_masks:
            rm(indir+'/mask_raw/mask_raw/'+mask)
            rm(indir+'/mask_uncropped/'+mask.replace('jpg','tiff'))
        unmasked_well_imgs.append(latest_well_im)
        unmasked_well_imgs=list(set(unmasked_well_imgs))
        files=unmasked_well_imgs
    
    
    """bounding box prompts"""
    
    
    iters=len(files)//batch_size
    if len(files)%batch_size == 0:
        iters+=1
    boxdf=pd.read_csv(proj_dir+'/bounding_boxes.csv')
    
    ix=0
    
    pbar=tqdm(total=iters,leave=True)
    for ix in range(0, len(files), batch_size):
        batch_filenames = files[ix:ix+batch_size]
        batch_filenames=[x.split('\\')[-1] for x in batch_filenames]
        boxes_batch = []
        img_batch = []
        valid_filenames = []
        
        for f in batch_filenames:
            image = Image.open(proj_dir+'/input_raw/'+f)
            image = np.array(image.convert("RGB"))
        
            bs = np.array(
                boxdf[boxdf['filename'] == f][['x_min','y_min','x_max','y_max']]
            )
        
            if len(bs) != 0:
                boxes_batch.append(bs)
                img_batch.append(image)
                valid_filenames.append(f)
        try:
            predictor.set_image_batch(img_batch)
            masks_batch, scores_batch, _ = predictor.predict_batch(
                None,
                None, 
                box_batch=boxes_batch, 
                multimask_output=False
            )
            ix+=batch_size
    
            
            gc.collect()
            torch.cuda.empty_cache()
    
            for masks, img_name in zip(masks_batch, valid_filenames):
                for i in range(len(masks)):
                    
                    mask2=masks[i]
                    
                    try:
                        binary_masked_image=mask2[0,:,:]
                        binary_masked_image[binary_masked_image!=0]=255
                    except:
                        binary_masked_image[binary_masked_image!=0]=255
            
                    #zoom in on mask
                    min_y,max_y,min_x,max_x=bbox2(binary_masked_image)
                    mask_im=binary_masked_image[min_y:max_y,min_x:max_x]
                    # Calculate the width and height of the bounding box
                    width = max_x - min_x
                    height = max_y - min_y
                    if (width<upper)&(height<upper)&(width>lower)&(height>lower):
                        #add either extra height or width to make crop square again
                        if width > height:
                            height_to_add = width - height
                            add_array=np.zeros((height_to_add,width))
                            mask_im_square=np.concatenate((mask_im,add_array),axis=0)
                        else:
                            width_to_add = height - width
                            add_array=np.zeros((height,width_to_add))
                            mask_im_square=np.concatenate((mask_im,add_array), axis=1)
                        
                        mask_im_square[mask_im_square!=0]=255
                        cv2.imwrite(out_folder+'/'+img_name+'_mask'+str(i)+'.jpg', mask_im_square)
                    if save_tiff: 
                            os.makedirs(out_folder2,exist_ok=True)
                            os.makedirs(out_folder2+'/'+img_name,exist_ok=True)
                            cv2.imwrite(out_folder2+'/'+img_name+'/'+img_name+'_mask'+str(i)+'.tiff', binary_masked_image)
        except Exception as e: print(e)
        pbar.update()
    
        



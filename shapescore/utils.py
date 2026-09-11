import os
import pandas as pd
from PIL import Image
import numpy as np
import colorsys

def sum_channels(img):
    summed_img = img[:,:,0].astype(np.int64)+img[:,:,1].astype(np.int64)+img[:,:,2].astype(np.int64)
    return summed_img

def make_mask_df(in_dir, proj_dir):
    fields=pd.read_csv(proj_dir+'/fields.csv')
    fdict=dict(zip(list(fields['field']),list(fields['name'])))
    mask_data = {x:[] for x in fields['name']}
    mask_data['path']=[];mask_data['filename']=[]

    for f in os.listdir(in_dir):
        mask_id=f.replace('.jpg','')
        l = mask_id.split('_')
        for field, name in fdict.items():
            mask_data[name].append(l[field])
        mask_data['path'].append(in_dir+'/'+f);mask_data['filename'].append(f)
    df = pd.DataFrame(mask_data)
    return df


def make_layout_df(proj_dir):
    files =os.listdir(proj_dir+'/plate_layouts')
    dfs=[pd.read_csv(proj_dir+'/plate_layouts/'+f) for f in files]
    plates=[s.replace('.csv','') for s in files]
    for n, df in dict(zip(plates,dfs)).items():
        df['plate']=[n]*len(df)
    layout_all = pd.concat(dfs).set_index(['well','plate'])
    return layout_all

def overlay_masks_on_image(
    original_image_path,
    masks_folder,
    output_path,
    alpha=0.4,
    saturation=0.75,
    value=0.95
    ):
        """
        Overlays multiple binary masks onto an original image using visually pleasing colors.
        """

        base_img = Image.open(original_image_path).convert("RGB")
        base = np.array(base_img).astype(np.float32)

        h, w, _ = base.shape

        mask_files = sorted([
            f for f in os.listdir(masks_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        ])

        if not mask_files:
            raise ValueError("No mask images found in folder.")

        n = len(mask_files)

        def get_pretty_color(i, n):
            # Evenly spaced hues around the color wheel
            hue = i / max(n, 1)
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            return np.array([r * 255, g * 255, b * 255])

        overlay = base.copy()

        for i, mask_file in enumerate(mask_files):
            mask_path = os.path.join(masks_folder, mask_file)

            mask_img = Image.open(mask_path).convert("L").resize((w, h))
            mask = np.array(mask_img)

            mask = (mask > 127).astype(np.float32)

            color = get_pretty_color(i, n)

            # Blend mask color onto image
            for c in range(3):
                overlay[..., c] = np.where(
                    mask == 1,
                    (1 - alpha) * overlay[..., c] + alpha * color[c],
                    overlay[..., c]
                )

        result = np.clip(overlay, 0, 255).astype(np.uint8)
        Image.fromarray(result).save(output_path)

        print(f"Saved overlay image to: {output_path}")


def summarize_to_well(proj):
    
    mask_df = pd.read_csv('project/'+proj+'/res/res.csv',index_col=0).set_index(['well','plate'])
    layout = make_layout_df('project/'+proj)
    layout['project']=[proj]*len(layout)

    mask2=mask_df[[(x in list(set(layout.index))) for x in mask_df.index]]
    mdata=pd.concat([mask2, layout.loc[mask2.index]],axis=1).dropna(subset=['alive_softmax']).reset_index()
    mdata['count']=[1]*len(mdata)
    agg={c:'first' for c in mdata.columns}
    agg['alive_softmax']='mean'
    agg['count']='sum'
    agg['intensity_avg']='mean'
    agg['size']='mean'
    mdata=mdata.loc[:,~mdata.columns.duplicated()].copy()
    mdata['wpp']=mdata['well'].values+mdata['plate'].values+mdata['project'].values
    surv=mdata.groupby('wpp').agg(agg).reset_index().drop(columns=['wpp'])

    return surv

def normalize(surv, ctrl_name='DMSO'): #cutoff - std deviations above mean
    #normalize for ctrls
    surv2=pd.DataFrame()
    for plate in list(set(surv['plate'])):
        df=surv[surv['plate']==plate]
        ctrl_avg=np.mean(df[df['drug']==ctrl_name]['alive_softmax'])
        df['norm SAP']=df['alive_softmax']/ctrl_avg
        ctrl_avg_size=np.mean(df[df['drug']==ctrl_name]['size'])
        df['norm size']=df['size']/ctrl_avg_size
        surv2=pd.concat([surv2,df],axis=0)
    return surv2


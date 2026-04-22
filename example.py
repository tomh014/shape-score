import os
os.chdir('G:/image_projects/auto_worm_analysis_versions/shape-score')

proj='2510101700bl_hs'

from shapescore.detection import predict_bb
predict_bb(proj)

from shapescore.maskgen import predict_segmentation
predict_segmentation(proj)

from shapescore.predict_survival import predict_survival
predict_survival(proj)

from shapescore.metrics import metrics
metrics(proj)

import pandas as pd
import seaborn as sns
res=pd.read_csv('project/'+proj+'/res/res.csv')

sns.violinplot(data=res,x='plate',y='alive_softmax')
sns.violinplot(data=res,x='plate',y='size')
sns.violinplot(data=res[['plate','well','count']].groupby(['well','plate']).sum(),x='plate',y='count')

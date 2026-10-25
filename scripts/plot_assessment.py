import os
from pathlib import Path
root=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(root/'output/.mplconfig')
import json
import sys
sys.path.insert(0,str(root))
from pipeline_common import config
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=root/config()['output_dir'];dest=out/'assessment'
df=pd.read_csv(out/'deviation_points.csv');good=df.quality_mask.to_numpy(bool)
nz=np.abs(df.normal_z.to_numpy());dist=df.euclidean_distance_mm.to_numpy()
fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
for label,mask,color in [('All 1 mm voxels',np.ones(len(df),bool),'#888888'),('Quality + |normal Z| >= 0.9',good&(nz>=.9),'#277da1'),('Quality + |normal Z| < 0.9',good&(nz<.9),'#d76b35')]:
 d=np.sort(dist[mask]);axes[0].plot(d,np.arange(1,len(d)+1)/len(d),label=label,color=color)
axes[0].set(xlim=(0,2.5),ylim=(0,1),xlabel='Euclidean point-to-surface distance (mm)',ylabel='Cumulative fraction',title='Broad horizontal regions fit better than side / sloped regions')
axes[0].legend(loc='lower right',fontsize=8);axes[0].grid(alpha=.2)
r=json.load(open(dest/'spatial_holdout.json'))['results'];s=json.load(open(out/'registration_sensitivity.json'))['cases']
median=[x['median_displacement_mm'] for x in s[-2:]]+[x['median_displacement_mm'] for x in r]
p95=[x['p95_displacement_mm'] for x in s[-2:]]+[x['p95_displacement_mm'] for x in r]
x=np.arange(6);axes[1].bar(x-.18,median,.36,label='Median displacement',color='#277da1');axes[1].bar(x+.18,p95,.36,label='P95 displacement',color='#f1a340')
axes[1].set(xticks=x,xticklabels=['Alternating\nhalf A','Alternating\nhalf B','Leave out\nX− Y−','Leave out\nX+ Y−','Leave out\nX− Y+','Leave out\nX+ Y+'],ylabel='Movement relative to saved fit (mm)',title='Fit depends on the spatial regions used')
axes[1].grid(axis='y',alpha=.2);axes[1].legend()
fig.suptitle('Additional diagnosis — spatial sensitivity is not calibrated absolute uncertainty',fontsize=13)
fig.savefig(dest/'assessment_summary.png',dpi=170)

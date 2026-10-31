"""All panels derive from acquired arrays and persisted analysis outputs."""
from pathlib import Path
import json,io
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon,Rectangle
from matplotlib.collections import PatchCollection
from pyproj import Transformer
R=Path(__file__).resolve().parents[1];D=R/'data';F=R/'figures';F.mkdir(exist_ok=True)
C={'siargao':'#007F86','leyte':'#CF7925','palawan':'#7563A6'};L={'siargao':'Siargao','leyte':'Southern Leyte','palawan':'Palawan'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def save(fig,name):
 for ext in ['pdf','png']:
  b=io.BytesIO();fig.savefig(b,format=ext,dpi=240,bbox_inches='tight');(F/(name+'.'+ext)).write_bytes(b.getvalue())
 plt.close(fig)
def main():
 grids=[json.load(open(p)) for p in sorted(D.glob('*_grid.json'))];a=pd.read_csv(D/'analysis_table_all.csv');ref=pd.read_csv(D/'gedi_mangrove_reference.csv');cal=pd.read_csv(D/'canopy_calibration_predictions.csv');s=json.load(open(D/'canopy_calibration_status.json'))
 fig,axs=plt.subplots(2,2,figsize=(8,7));ax=axs[0,0];land=json.load(open(D/'land_boundaries.geojson'));patches=[]
 for f in land['features']:
  geom=f['geometry'];polys=[geom['coordinates']] if geom['type']=='Polygon' else geom['coordinates'] if geom['type']=='MultiPolygon' else []
  patches += [Polygon(po[0],closed=True) for po in polys]
 for axis in axs.ravel():axis.add_collection(PatchCollection([Polygon(p.get_xy(),closed=True) for p in patches],facecolor='#E7EBE7',edgecolor='#919A94',linewidth=.4))
 tr=pd.read_csv(D/'rai_2021_track.csv');ax.plot(tr.LON,tr.LAT,c='#444',lw=1,label='Rai best track')
 for reg in C:
  gg=[g for g in grids if g['region']==reg];xx=[(g['bbox'][0]+g['bbox'][2])/2 for g in gg];yy=[(g['bbox'][1]+g['bbox'][3])/2 for g in gg];ax.scatter(xx,yy,color=C[reg],s=12,label=L[reg])
 ax.set(xlim=(117,128),ylim=(5,15),xlabel='Longitude (°E)',ylabel='Latitude (°N)',title='(a) Study regions and Rai track');ax.legend(fontsize=7,loc='lower left');ax.set_aspect('equal')
 for axis,reg,letter in zip(axs.ravel()[1:],C,['b','c','d']):
  gg=[g for g in grids if g['region']==reg];bb=np.array([g['bbox'] for g in gg]);axis.set(xlim=(bb[:,0].min()-.04,bb[:,2].max()+.04),ylim=(bb[:,1].min()-.04,bb[:,3].max()+.04),xlabel='Longitude (°E)',ylabel='Latitude (°N)',title=f'({letter}) {L[reg]}');axis.set_aspect('equal')
  for g in gg:
   b=g['bbox'];axis.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],edgecolor=C[reg],facecolor=C[reg],alpha=.7));axis.annotate(g['site'].split('_')[1],((b[0]+b[2])/2,(b[1]+b[3])/2),xytext=(5,2),textcoords='offset points',fontsize=8)
 fig.tight_layout();save(fig,'fig01_locations')
 fig,axs=plt.subplots(3,4,figsize=(9,7.2))
 for row,reg in enumerate(C):
  for col,k in enumerate(range(1,5)):
   site=f'{reg}_{k:02}';g=next(g for g in grids if g['site']==site);m=np.load(D/f'{site}_grid.npz')['mask'];axis=axs[row,col];axis.imshow(m,extent=(0,2.4,0,2.4),origin='upper',cmap=ListedColormap(['#F1F2EB',C[reg]]));r=ref[ref.site==site]
   if len(r):axis.scatter((r.x-g['transform'][2])/1000,(r.y-(g['transform'][5]-2400))/1000,s=5,facecolor='#D9344B',edgecolor='white',linewidth=.2)
   axis.set(title=site,xlim=(0,2.4),ylim=(0,2.4),xticks=[0,1.2,2.4],yticks=[0,1.2,2.4]);axis.tick_params(labelsize=7)
   if row==2:axis.set_xlabel('km')
   if col==0:axis.set_ylabel('km')
 fig.suptitle('Fixed 2020 mapped mangroves and quality/purity-qualified GEDI footprints',fontsize=11);fig.tight_layout();save(fig,'fig02_landscape_masks')
 sites=[f'{r}_{k:02}' for r in C for k in range(1,5)];counts=a.groupby('site').primary_eligible.agg(['sum','count']).reindex(sites);gd=pd.read_csv(D/'gedi_attrition.csv').set_index('landscape').reindex(sites)
 fig,axs=plt.subplots(1,2,figsize=(8,5.5));y=np.arange(12);axs[0].barh(y,counts['count'],color='#DDE3DE',label='Candidate cells');axs[0].barh(y,counts['sum'],color=[C[x.split('_')[0]] for x in sites],label='Forecast eligible');axs[0].set(yticks=y,yticklabels=sites,xlabel='100 m cells',title='(a) Forecast cohort retention');axs[0].invert_yaxis();axs[0].legend(fontsize=7)
 axs[1].barh(y,gd.matched_prestorm,color='#647D87',label='2019–2021');axs[1].barh(y,gd.matched_poststorm,left=gd.matched_prestorm,color='#D49742',label='Oct 2022–Mar 2023');axs[1].set(yticks=y,yticklabels=[],xlabel='GEDI shots matched within 30 days',title='(b) Independent-sensor coverage');axs[1].invert_yaxis();axs[1].legend(fontsize=7);fig.tight_layout();save(fig,'fig03_coverage')
 met=pd.read_csv(D/'primary_landscape_metrics.csv');nest=pd.read_csv(D/'primary_nested_development.csv');fig,axs=plt.subplots(1,2,figsize=(8,4));mm=met[met.role=='new_transfer'];xx=np.arange(3)
 for i,(name,color) in enumerate([('selected','#007F86'),('median','#7563A6'),('persistence','#CF7925')]):
  vals=mm[mm.model==name].set_index('landscape').reindex(['palawan_02','palawan_03','palawan_04']).mae;axs[0].bar(xx+(i-1)*.25,vals,width=.24,color=color,label=name)
 axs[0].set(xticks=xx,xticklabels=['Palawan 02','Palawan 03','Palawan 04'],ylabel='MAE (NDVI deficit)',title='(a) New geographic transfer');axs[0].legend(fontsize=7)
 yy=np.arange(len(nest));axs[1].scatter(nest.mae,yy,c='#007F86',label='Nested selected');axs[1].scatter(nest.median_mae,yy,c='#7563A6',marker='x',label='Training median');axs[1].set(yticks=yy,yticklabels=nest.landscape,xlabel='MAE (NDVI deficit)',title='(b) Outer omitted landscape');axs[1].invert_yaxis();axs[1].legend(fontsize=7);fig.tight_layout();save(fig,'fig04_forecast_performance')
 fig,axs=plt.subplots(1,3,figsize=(9,3.5));markers={'prestorm':'o','poststorm_2022_23':'^'}
 for axis,reg in zip(axs,C):
  dd=cal[cal.region==reg]
  for period,m in markers.items():
   pp=dd[dd.period==period];axis.scatter(pp.ndvi,pp.cover,s=13,alpha=.5,c=C[reg],marker=m,edgecolors='none',label=period.replace('poststorm_2022_23','2022–23').replace('prestorm','2019–21'))
  x=np.linspace(.2,1,100);axis.plot(x,np.clip(s['intercept']+s['slope']*x,0,1),'--',c='#444',lw=1,label='Development calibration');axis.set(xlim=(.2,1),ylim=(0,1),xlabel='Concurrent footprint NDVI',title=L[reg]);axis.text(.04,.96,f'n = {len(dd)} shots',transform=axis.transAxes,va='top',fontsize=8)
 axs[0].set_ylabel('GEDI canopy-cover fraction');axs[1].legend(fontsize=6,loc='lower left');fig.tight_layout();save(fig,'fig05_canopy_reference')
 pred=pd.read_csv(D/'primary_predictions.csv');fig,axs=plt.subplots(3,3,figsize=(8,7));cm=plt.get_cmap('RdBu_r').copy();cm.set_bad('#F4F4ED');im=None
 for ri,site in enumerate(['palawan_02','palawan_03','palawan_04']):
  dd=pred[pred.site==site];eligible=np.load(D/f'{site}_grid.npz')['eligible']
  for ci,name in enumerate(['observed','selected','residual']):
   v=np.full((24,24),np.nan);val=dd.selected-dd.observed if name=='residual' else dd[name];v[dd.row,dd.col]=val;axis=axs[ri,ci];axis.imshow(np.where(eligible,1,np.nan),cmap=ListedColormap(['#BBBFB9']),extent=(0,2.4,0,2.4));im=axis.imshow(v,cmap=cm,vmin=-.2,vmax=.2,extent=(0,2.4,0,2.4));axis.set(xticks=[0,1.2,2.4],yticks=[0,1.2,2.4],title=(['Observed deficit','Predicted deficit','Prediction − observation'][ci] if ri==0 else ''))
   if ci==0:axis.set_ylabel(site+'\nkm')
   if ri==2:axis.set_xlabel('km')
 fig.subplots_adjust(right=.87,hspace=.3,wspace=.15);cax=fig.add_axes([.90,.2,.025,.6]);fig.colorbar(im,cax=cax,label='NDVI units');save(fig,'fig06_transfer_maps')
if __name__=='__main__':main()
"""Fivefold validation on the SAME pilot 2054-gene dataset as the original report.
Each transformer is fitted inside each CV training fold. Results are exploratory.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import gzip,json,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest,f_classif,VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
ROOT=Path(__file__).resolve().parent
EXPR=Path(os.environ.get('TCGA_EXPRESSION','/mnt/data/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz'))
META=Path(os.environ.get('TCGA_METADATA','/mnt/data/TCGA_phenotype_denseDataOnlyDownload.tsv.gz'))
OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True)
class_map={'breast invasive carcinoma':'BRCA','kidney clear cell carcinoma':'KIRC','colon adenocarcinoma':'COAD','lung adenocarcinoma':'LUAD','prostate adenocarcinoma':'PRAD'}
meta=pd.read_csv(META,sep='\t',compression='gzip')
meta=meta[(meta['sample_type']=='Primary Tumor') & meta['_primary_disease'].isin(class_map)].drop_duplicates('sample')
labels=dict(zip(meta['sample'],meta['_primary_disease'].map(class_map)))
print('Reading compressed expression data',flush=True)
with gzip.open(EXPR,'rt') as fp:
 header=fp.readline().rstrip('\n').split('\t')
 positions=[i+1 for i,s in enumerate(header[1:]) if s in labels]
 seen=set();positions_unique=[];samples=[]
 for p in positions:
  s=header[p]
  if s not in seen: seen.add(s);positions_unique.append(p);samples.append(s)
 rows=[]
 for row_idx,line in enumerate(fp):
  if row_idx%10:continue
  a=line.rstrip('\n').split('\t')
  rows.append([float(a[k]) if a[k] not in ('','NA') else np.nan for k in positions_unique])
X=np.asarray(rows,dtype=np.float32).T
y=np.asarray([labels[s] for s in samples]);patients=['-'.join(s.split('-')[:3]) for s in samples]
assert len(set(patients))==len(patients), 'Need patient-level group cross-validation'
idx_train,idx_test=train_test_split(np.arange(len(y)),test_size=.2,random_state=42,stratify=y)
Xtr,ytr=X[idx_train],y[idx_train]
print('train shape',Xtr.shape,'test',len(idx_test),flush=True)
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
models={'majority':DummyClassifier(strategy='most_frequent')}
for k in [20,50,100,250,500]:
 models[f'ANOVA-{k} + logistic']=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('variance',VarianceThreshold()),('select',SelectKBest(f_classif,k=k)),('scale',StandardScaler()),('clf',LogisticRegression(max_iter=1000,random_state=42))])
for k in [20,50,100]:
 models[f'PCA-{k} + logistic']=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler()),('pca',PCA(n_components=k,svd_solver='randomized',random_state=42)),('clf',LogisticRegression(max_iter=1000,random_state=42))])
results=[]
for name,model in models.items():
 t=time.time()
 try:
  with __import__('warnings').catch_warnings():
   __import__('warnings').simplefilter('ignore',RuntimeWarning)
   scores=cross_validate(model,Xtr,ytr,cv=cv,scoring={'accuracy':'accuracy','macro_f1':'f1_macro'},n_jobs=1,error_score='raise')
  row={'model':name,'fold_accuracy':scores['test_accuracy'].tolist(),'fold_macro_f1':scores['test_macro_f1'].tolist(),'accuracy_mean':float(scores['test_accuracy'].mean()),'accuracy_sd':float(scores['test_accuracy'].std(ddof=1)),'macro_f1_mean':float(scores['test_macro_f1'].mean()),'macro_f1_sd':float(scores['test_macro_f1'].std(ddof=1)),'runtime_s':round(time.time()-t,2)}
  results.append(row)
  print(f"{name}: accuracy {row['accuracy_mean']:.4f} +/- {row['accuracy_sd']:.4f}; macro F1 {row['macro_f1_mean']:.4f} +/- {row['macro_f1_sd']:.4f}, time {row['runtime_s']}s",flush=True)
 except Exception as ex:
  print(name,'FAILED',str(ex),flush=True)
output={'design':'Exploratory fivefold CV on the original 80% training split; original 20% holdout was already evaluated in earlier pilot and not reused here. All imputation, variance filtering, supervised selection, standardization, and PCA fitted within each CV training fold.','n_total':len(y),'n_train_cv':len(ytr),'n_original_heldout':len(idx_test),'n_pilot_genes':X.shape[1],'n_missing_values':int(np.isnan(X).sum()),'n_unique_patients':len(set(patients)),'cv_splits':5,'seed':42,'results':results}
(OUT/'cv_results.json').write_text(json.dumps(output,indent=2))
pd.DataFrame(results).drop(columns=['fold_accuracy','fold_macro_f1']).to_csv(OUT/'cv_metrics.csv',index=False)

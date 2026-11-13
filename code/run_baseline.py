"""Reproducible preliminary TCGA 5-class classifier for CS/SE 4AL3 Group 57.
Streaming subset of genes for manageable initial experiments; NOT a final full-feature result.
"""
import gzip, csv, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ROOT=Path(__file__).resolve().parent
EXPR=Path(os.environ.get('TCGA_EXPRESSION','/mnt/data/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz'))
META=Path(os.environ.get('TCGA_METADATA','/mnt/data/TCGA_phenotype_denseDataOnlyDownload.tsv.gz'))
OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
class_map={'breast invasive carcinoma':'BRCA','kidney clear cell carcinoma':'KIRC','colon adenocarcinoma':'COAD','lung adenocarcinoma':'LUAD','prostate adenocarcinoma':'PRAD'}
meta=pd.read_csv(META,sep='\t',compression='gzip')
meta=meta[(meta['sample_type']=='Primary Tumor') & (meta['_primary_disease'].isin(class_map))].drop_duplicates('sample')
labels=dict(zip(meta['sample'],meta['_primary_disease'].map(class_map)))
print('metadata primary tumour rows',len(meta),flush=True)
# Read header then stream every tenth gene. Full input stream is processed only once.
with gzip.open(EXPR,'rt') as fp:
    header=fp.readline().rstrip('\n').split('\t')
    positions=[i for i,s in enumerate(header[1:]) if s in labels]
    samples=[header[i+1] for i in positions]
    # In rare cases duplicated sample columns, use only one per sample.
    seen=set(); selected=[]; unique_samples=[]
    for i,s in zip(positions,samples):
        if s not in seen: selected.append(i+1); unique_samples.append(s); seen.add(s)
    rows=[]; genes=[]
    for line_number,line in enumerate(fp):
        if line_number%10: continue
        a=line.rstrip('\n').split('\t')
        try: vals=[float(a[k]) if a[k] not in ('','NA') else np.nan for k in selected]
        except (ValueError,IndexError): continue
        rows.append(vals); genes.append(a[0])
        if line_number%5000==0: print('scanned gene row',line_number,flush=True)
X=np.asarray(rows,dtype=np.float32).T
classes=np.asarray([labels[s] for s in unique_samples])
patients=np.asarray(['-'.join(s.split('-')[:3]) for s in unique_samples])
if len(set(patients))!=len(patients):
    raise ValueError('Multiple patient records detected; implement group-aware split before training')
print('missing cell count',int(np.isnan(X).sum()),flush=True)
print('matrix',X.shape,'classes',pd.Series(classes).value_counts().to_dict(),flush=True)
idx_train,idx_test=train_test_split(np.arange(len(classes)),test_size=.2,random_state=42,stratify=classes)
Xtr,Xte=X[idx_train],X[idx_test]; ytr,yte=classes[idx_train],classes[idx_test]
models={'majority':DummyClassifier(strategy='most_frequent'),
        'anova100_logistic':Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('select',SelectKBest(f_classif,k=min(100,Xtr.shape[1]))),('scale',StandardScaler()),('logistic',LogisticRegression(max_iter=1000,random_state=42))])}
results={}; lab=['BRCA','KIRC','COAD','LUAD','PRAD']
for name, model in models.items():
    print('training',name,flush=True)
    model.fit(Xtr,ytr); yp=model.predict(Xte)
    results[name]={'accuracy':float(accuracy_score(yte,yp)), 'macro_f1':float(f1_score(yte,yp,average='macro')),
                   'classification_report':classification_report(yte,yp,output_dict=True,zero_division=0),
                   'confusion_matrix':confusion_matrix(yte,yp,labels=lab).tolist()}
    print(name,'accuracy',round(results[name]['accuracy'],4),'macro_f1',round(results[name]['macro_f1'],4),flush=True)
output={'dataset':{'samples':len(classes),'sample_classes':pd.Series(classes).value_counts().to_dict(),'gene_features_original':20532,'gene_features_in_experiment':len(genes),'selection':'every 10th gene row as a temporary reproducible computationally light pilot; supervised ANOVA features selected only after split','patients_unique':len(set(patients)),'missing_cells_in_pilot':int(np.isnan(X).sum())},'experiment':{'test_fraction':0.2,'random_state':42,'train_n':len(idx_train),'test_n':len(idx_test),'labels':lab},'results':results}
(OUT/'results.json').write_text(json.dumps(output,indent=2))
pd.DataFrame([{'model':k,'accuracy':v['accuracy'],'macro_f1':v['macro_f1']} for k,v in results.items()]).to_csv(OUT/'metrics.csv',index=False)
print('saved',OUT/'results.json',flush=True)

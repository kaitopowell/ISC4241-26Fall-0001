"""Diagnostic figures use development/out-of-fold data unless explicitly labeled test."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
from sklearn.tree import plot_tree
from .models import CLASSES, LABELS, vif_table
from .data import FULL, REDUCED
sns.set_theme(style='whitegrid',context='notebook',palette='colorblind')
FIG = Path(__file__).resolve().parents[1]/'reports/figures'

def save(name):
    plt.tight_layout()
    plt.savefig(FIG/f'{name}.png',dpi=165,bbox_inches='tight')
    plt.close()

def heatmap(X,name,title):
    plt.figure(figsize=(12,10))
    sns.heatmap(X.corr(),vmin=-1,vmax=1,cmap='vlag',annot=True,fmt='.2f',annot_kws={'size':7},square=True)
    plt.title(title)
    save(name)

def exploration(fixtures, dev, splits):
    FIG.mkdir(parents=True,exist_ok=True)
    # Training/development only: avoid using held-out outcomes to choose changes.
    counts=pd.crosstab(fixtures.season,fixtures.outcome).reindex(columns=CLASSES)
    counts.plot.bar(figsize=(8,4)); plt.title('Development season class counts'); plt.ylabel('Matches'); save('01_class_counts')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    sns.histplot(fixtures.hg,bins=np.arange(0,11)-.5,ax=axes[0],label='Home',color='tab:blue')
    sns.histplot(fixtures.ag,bins=np.arange(0,11)-.5,ax=axes[0],label='Away',color='tab:orange',alpha=.5)
    axes[0].legend(); axes[0].set(title='Observed goals (development)',xlabel='Goals')
    sns.boxplot(data=dev[['ppg_diff','gf_diff','ga_diff','win_diff']],ax=axes[1]); axes[1].set_title('Form differences: extremes are not deleted'); save('02_distributions')
    heatmap(dev[FULL],'03_full_correlation','Full initial predictors: development data only')
    heatmap(dev[REDUCED],'04_difference_correlation','Candidate difference predictors: development data only')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,c in zip(axes,['ppg_diff','elo_diff']):
        sns.boxplot(data=dev,x='outcome',y=c,order=CLASSES,ax=ax); ax.set_title(f'{c} by observed result')
    save('05_features_by_result')
    plt.figure(figsize=(10,4))
    for fold,tr,va in splits:
        plt.plot([dev.iloc[tr].date.min(),dev.iloc[tr].date.max()],[fold,fold],lw=6,color='tab:blue')
        plt.plot([dev.iloc[va].date.min(),dev.iloc[va].date.max()],[fold,fold],lw=6,color='tab:orange')
    plt.title('10 expanding-window folds: blue=train, orange=validation');plt.ylabel('Fold');save('06_cv_timeline')

def comparison(summary,folds):
    best=summary.loc[summary.groupby('family').cv_log_loss.idxmin()].sort_values('cv_log_loss')
    shown=best[best.family!='majority']
    plt.figure(figsize=(10,5));plt.barh(shown.name,shown.cv_log_loss,xerr=shown.fold_log_loss_sd,color='steelblue');plt.xlabel('Pooled out-of-fold log loss (lower is better)');plt.title('Model families; error bars = fold SD, not confidence intervals');save('07_model_comparison')
    names=shown.name.tolist()
    subset=folds[folds.name.isin(names)]
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name,g in subset.groupby('name'):
        axes[0].plot(g.fold,g.log_loss,marker='.',label=name)
    axes[0].set(title='Validation loss by time block',xlabel='Fold',ylabel='Log loss');axes[0].legend(fontsize=6)
    agg=subset.groupby('name')[['train_log_loss','log_loss']].mean().loc[names]
    agg.plot.bar(ax=axes[1]);axes[1].tick_params(axis='x',labelsize=6);axes[1].set_title('Training vs validation: overfitting check');save('08_overfitting')

def probability_diagnostics(frame,p,prefix,title):
    y=frame.outcome.to_numpy(); pred=CLASSES[p.argmax(axis=1)]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    cm=confusion_matrix(y,pred,labels=CLASSES)
    sns.heatmap(cm,annot=True,fmt='d',xticklabels=LABELS,yticklabels=LABELS,ax=axes[0],cmap='Blues')
    sns.heatmap(cm/cm.sum(axis=1,keepdims=True),annot=True,fmt='.2f',xticklabels=LABELS,yticklabels=LABELS,ax=axes[1],vmin=0,vmax=1,cmap='Blues')
    for ax in axes:ax.set(xlabel='Predicted',ylabel='Actual')
    fig.suptitle(title+' — confusion matrices (counts / row recall)');save(prefix+'_confusion')
    fig,axes=plt.subplots(1,3,figsize=(13,4)); bins=[]
    for j,(c,label) in enumerate(zip(CLASSES,LABELS)):
        truth=(y==c).astype(int)
        # Fixed bins report support; empty bins omitted rather than fabricated.
        group=np.minimum((p[:,j]*5).astype(int),4)
        for k in range(5):
            mask=group==k
            if mask.any():bins.append({'class':c,'bin':k,'n':int(mask.sum()),'mean_probability':p[mask,j].mean(),'observed_rate':truth[mask].mean()})
        tab=pd.DataFrame([r for r in bins if r['class']==c])
        axes[j].plot([0,1],[0,1],'--',color='gray')
        axes[j].plot(tab.mean_probability,tab.observed_rate,'o-')
        for r in tab.itertuples():axes[j].annotate(str(r.n),(r.mean_probability,r.observed_rate),fontsize=7)
        axes[j].set(xlim=(0,1),ylim=(0,1),title=label,xlabel='Mean predicted probability',ylabel='Observed fraction')
    fig.suptitle(title+' — calibration (labels show bin counts)');save(prefix+'_calibration')
    pd.DataFrame(bins).to_csv(FIG.parent/f'{prefix}_calibration_bins.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for j,label in enumerate(LABELS):
        truth=(y==CLASSES[j]).astype(int)
        fpr,tpr,_=roc_curve(truth,p[:,j]);axes[0].plot(fpr,tpr,label=label)
        precision,recall,_=precision_recall_curve(truth,p[:,j]);axes[1].plot(recall,precision,label=f'{label} AP={average_precision_score(truth,p[:,j]):.2f}')
        axes[1].axhline(truth.mean(),ls=':',alpha=.25)
    axes[0].plot([0,1],[0,1],'--',color='gray');axes[0].set(xlabel='False positive rate',ylabel='True positive rate',title='One-vs-rest ROC')
    axes[1].set(xlabel='Recall',ylabel='Precision',title='One-vs-rest precision–recall');axes[0].legend();axes[1].legend(fontsize=8);fig.suptitle(title);save(prefix+'_roc_pr')
    onehot=(y[:,None]==CLASSES).astype(float)
    loss=-np.log(np.clip(p[np.arange(len(y)),[list(CLASSES).index(c) for c in y]],1e-15,1))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for j,label in enumerate(LABELS):
        group=np.minimum((p[:,j]*5).astype(int),4)
        xs=[];rs=[]
        for k in range(5):
            mask=group==k
            if mask.any():xs.append(p[mask,j].mean());rs.append((onehot[mask,j]-p[mask,j]).mean())
        axes[0].plot(xs,rs,'o-',label=label)
    axes[0].axhline(0,ls='--',color='gray');axes[0].set(xlabel='Mean predicted probability',ylabel='Mean indicator minus probability',title='Binned probability residuals');axes[0].legend(fontsize=8)
    monthly=pd.DataFrame({'date':frame.date.to_numpy(),'loss':loss}).set_index('date').resample('MS').loss.mean()
    axes[1].plot(monthly.index,monthly.values,'o-');axes[1].set(title='Monthly prediction loss',ylabel='Log loss');axes[1].tick_params(axis='x',rotation=40)
    fig.suptitle(title);save(prefix+'_residuals')
    errors=frame[['match_id','date','home','away','outcome']].copy();errors['predicted']=pred;errors['log_loss']=loss
    errors.to_csv(FIG.parent/f'{prefix}_errors.csv',index=False)

def interpretation(logit, tree, dev):
    features=logit.named_steps['features'].kept_
    coef=logit.named_steps['model'].coef_
    tab=pd.DataFrame(coef,index=CLASSES,columns=features)
    tab.to_csv(FIG.parent/'standardized_logistic_coefficients.csv')
    plt.figure(figsize=(10,4));sns.heatmap(tab,annot=True,fmt='.2f',cmap='vlag',center=0);plt.title('Multinomial L2 coefficients per training SD (not causal)');save('15_logistic_coefficients')
    plt.figure(figsize=(17,8));plot_tree(tree.named_steps['model'],feature_names=tree.named_steps['features'].kept_,class_names=LABELS,filled=True,max_depth=2,fontsize=7)
    plt.title('First levels of selected decision tree; thresholds use standardized features');save('16_tree')

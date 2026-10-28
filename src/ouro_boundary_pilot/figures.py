"""Reproducible scientific diagrams and figures from frozen numerical evidence.

All figures are sourced from raw arrays and validated transformations; no simulation.
To regenerate: python scripts/generate_figures.py
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from .artifacts import read_raw, project_root
from .statistics import calculate, paired_question_effects, bootstrap_indices, ci

NAVY='#12253e'; MID='#386999'; TEAL='#117b75'; BLUE='#4a75bd'; PURPLE='#8063a9'; GRAY='#6d7785'; LIGHT='#e7edf3'; ORANGE='#bd7338'; RED='#a94949'
PALETTE=[GRAY,BLUE,TEAL,PURPLE]
DIRECTIONS=['seed42 / 2913','seed123 / 2437']
MODES=['Windows FP16','DGX FP16','FP32 realized','FP32 intended']


def plot_theme():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':14,
        'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,
        'axes.edgecolor':'#ccd4df','xtick.color':'#46576a','ytick.color':'#46576a',
        'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
        'svg.fonttype':'none'})


def save(fig, folder:Path, name:str):
    folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/(name+'.png'),dpi=175,bbox_inches='tight',pad_inches=0.18)
    fig.savefig(folder/(name+'.svg'),bbox_inches='tight',pad_inches=0.18)
    plt.close(fig)


def load_and_effects():
    old,new=read_raw()
    w=paired_question_effects(old['margins'],old['alphas'])
    p=paired_question_effects(new['margins'],float(new['alpha']))
    combined=np.concatenate([w[:,:,:1],p],axis=2)
    return old,new,w,p,combined


def effect_forest(out, w,p):
    all_p=np.concatenate([w[:,:,:1],p],axis=2)
    draws=bootstrap_indices()
    boot=all_p[draws].mean(axis=1)
    fig,ax=plt.subplots(figsize=(9.5,5.1))
    positions=[]; labels=[]
    for d in range(2):
        for m in range(4):
            y=7-(d*4+m)
            x=all_p[:,d,m].mean()
            coverage=0.9875 if m in (2,3) else 0.975
            lo,hi=ci(boot[:,d,m],coverage)
            ax.errorbar(x,y,xerr=[[x-lo],[hi-x]],fmt='o',markersize=6,capsize=3,elinewidth=2,
                        color=PALETTE[m],zorder=3)
            positions.append(y)
            labels.append(('seed42' if d==0 else 'seed123')+'  ·  '+MODES[m])
    ax.axvline(0,color=RED,linestyle='--',linewidth=1.2)
    ax.set_yticks(positions, labels)
    ax.set_xlabel('Mean candidate − matched-control boundary contrast T')
    ax.set_title('Finite-difference causal contrasts across execution modes',loc='left',pad=17,fontweight='bold',color=NAVY)
    ax.text(0,1.01,'97.5% individual CIs for Windows/DGX FP16; 98.75% for joint four FP32 tests',
        transform=ax.transAxes,va='bottom',fontsize=8.8,color=GRAY)
    ax.grid(axis='x',alpha=.17)
    save(fig,out,'01_effect_forest')


def question_level(out,p,w):
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.65),sharex=True,sharey=True)
    for d,ax in enumerate(axes):
        for j in range(30):
            ax.plot([j+1,j+1],[p[j,d,0],p[j,d,2]],color=LIGHT,zorder=1)
        ax.scatter(np.arange(1,31)-0.1,p[:,d,0],label='DGX FP16',s=20,c=BLUE,zorder=2)
        ax.scatter(np.arange(1,31)+0.1,p[:,d,2],label='FP32 intended',s=20,c=TEAL,zorder=3,marker='D')
        ax.axhline(0,color=RED,ls='--',lw=1)
        ax.set_title(DIRECTIONS[d],fontweight='bold',color=NAVY)
        ax.set_xlabel('Frozen SciQ question index')
        ax.set_xlim(.3,30.7)
        ax.grid(axis='y',alpha=.14)
        if d==0: ax.set_ylabel('Question-level paired contrast P(q)')
    axes[1].legend(frameon=False,fontsize=9,loc='upper right')
    fig.suptitle('Question-level numerical sensitivity (same 30 prompts)',fontweight='bold',color=NAVY,x=.03,ha='left',y=1.04,fontsize=14)
    save(fig,out,'02_question_level')


def precision_delta(out,p):
    fig,ax=plt.subplots(figsize=(9.2,5))
    draws=bootstrap_indices()
    boots=p[draws].mean(axis=1)
    j=0
    ticks=[]
    for d in range(2):
        for arm in (1,2):
            delta=p[:,d,arm]-p[:,d,0]
            lower,upper=ci(boots[:,d,arm]-boots[:,d,0],.95)
            x=delta.mean()
            y=3-j
            ax.errorbar(x,y,xerr=[[x-lower],[upper-x]],fmt='o',capsize=5,color=[TEAL,PURPLE][arm-1],markersize=8,lw=2)
            ticks.append(('seed42' if d==0 else 'seed123')+'  ·  '+('realized' if arm==1 else 'intended'))
            j+=1
    ax.axvspan(-.005,.005,color=TEAL,alpha=.11,label='Proposed ±0.005 equivalence band')
    ax.axvline(0,color=GRAY,lw=1)
    ax.set_yticks([3,2,1,0],ticks)
    ax.set_xlabel('FP32 − DGX FP16 mean paired contrast discrepancy ΔT')
    ax.set_title('Proposed numerical-equivalence criterion NOT established',loc='left',color=NAVY,fontweight='bold',pad=14)
    ax.grid(axis='x',alpha=.18)
    ax.legend(frameon=False,loc='lower right',fontsize=9)
    save(fig,out,'03_precision_equivalence')


def endpoint_rounding(out,p):
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.5))
    for d,ax in enumerate(axes):
        xs=p[:,d,1];ys=p[:,d,2]
        ax.scatter(xs,ys,c=TEAL,s=25,alpha=.78)
        lo=min(xs.min(),ys.min())-.012;hi=max(xs.max(),ys.max())+.012
        ax.plot([lo,hi],[lo,hi],color=GRAY,lw=1,ls='--')
        ax.set_xlim(lo,hi);ax.set_ylim(lo,hi)
        ax.set_xlabel('FP32 realized endpoint P(q)')
        if d==0:ax.set_ylabel('FP32 intended endpoint P(q)')
        ax.set_title(DIRECTIONS[d],fontweight='bold',color=NAVY)
        ax.grid(alpha=.12)
    fig.suptitle('Rounding endpoint comparison: not end-to-end FP32',fontweight='bold',color=NAVY,x=.03,ha='left',y=1.05,fontsize=14)
    save(fig,out,'04_endpoint_rounding')


def magnitude(out,w):
    boot=w[bootstrap_indices()].mean(axis=1)
    fig,ax=plt.subplots(figsize=(8.5,4.65))
    x=np.array([0.08,0.16]);off=[-.006,.006]
    for d in range(2):
        means=w[:,d,:].mean(axis=0)
        lows=[];ups=[]
        for a in range(2):
            lo,hi=ci(boot[:,d,a],.975);lows.append(means[a]-lo);ups.append(hi-means[a])
        ax.errorbar(x+off[d],means,yerr=[lows,ups],fmt='o-',capsize=4,lw=1.6,markersize=7,
                    color=[BLUE,PURPLE][d],label=DIRECTIONS[d])
    ax.axhline(0,color=RED,ls='--',lw=1)
    ax.set_xticks(x, ['0.08 (primary)','0.16 (secondary)'])
    ax.set_xlabel('Original Windows FP16 perturbation magnitude α')
    ax.set_ylabel('Mean paired contrast T')
    ax.set_title('Original frozen magnitude sensitivity (not replication)',loc='left',color=NAVY,fontweight='bold',pad=14)
    ax.grid(axis='y',alpha=.2);ax.legend(frameon=False)
    save(fig,out,'05_magnitude_sensitivity')


def mode_shift(out,p,w):
    fig,ax=plt.subplots(figsize=(8.7,4.8))
    for d in range(2):
        x=w[:,d,0];y=p[:,d,0]
        ax.scatter(x,y,marker=['o','^'][d],s=36,alpha=.7,label=DIRECTIONS[d],color=[BLUE,PURPLE][d])
    lo=min(w[:,:,0].min(),p[:,:,0].min())-.02
    hi=max(w[:,:,0].max(),p[:,:,0].max())+.02
    ax.plot([lo,hi],[lo,hi],ls='--',color=GRAY)
    ax.set_xlim(lo,hi);ax.set_ylim(lo,hi)
    ax.set_xlabel('Archived Windows FP16 question effect P(q)')
    ax.set_ylabel('DGX FP16 question effect P(q)')
    ax.set_title('Cross-hardware FP16: not question-level identical',loc='left',color=NAVY,fontweight='bold',pad=14)
    ax.grid(alpha=.13);ax.legend(frameon=False)
    save(fig,out,'06_cross_hardware')


def diagram(out,kind):
    fig,ax=plt.subplots(figsize=(12.8,3.65 if kind=='arch' else 4.6))
    ax.set_xlim(0,100);ax.set_ylim(0,100);ax.axis('off')
    def block(x,y,w,h,title,subtitle='',accent=MID):
        patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.5,rounding_size=3',lw=1.05,
                            facecolor='#f3f7fb',edgecolor='#c8d6e5')
        ax.add_patch(patch)
        ax.text(x+2,y+h-9,title,ha='left',va='top',fontweight='bold',fontsize=10.3,color=NAVY)
        if subtitle: ax.text(x+2,y+h-26,subtitle,ha='left',va='top',fontsize=8.6,color=GRAY,linespacing=1.6)
        ax.plot([x+2,x+10],[y+h-4,y+h-4],color=accent,lw=3,solid_capstyle='round')
    def arrow(x0,y0,x1,y1):
        ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle='-|>',mutation_scale=15,lw=1.5,color=MID))
    if kind=='arch':
        ax.text(2,97,'Ouro-1.4B · observed causal intervention interfaces',color=NAVY,fontsize=15,fontweight='bold',va='top')
        block(2,30,20,47,'Frozen prompt','Full-sequence token\nembeddings')
        block(28,30,20,47,'Recurrent model','Normalize at loop\nboundary 2 or 3',accent=PURPLE)
        block(54,30,20,47,'State patch','Last prompt token\n± α · vector',accent=ORANGE)
        block(80,30,18,47,'Readout','One more loop;\ngold − wrong logit',accent=TEAL)
        for x0,x1 in [(22,28),(48,54),(74,80)]:arrow(x0,53,x1-1,53)
        ax.text(64,17,'Candidate SAE direction / 16 matched controls  ·  no weight updates',ha='center',fontsize=9,color=GRAY)
    else:
        ax.text(2,97,'Three-arm DGX precision validation (original Windows FP16 archived separately)',color=NAVY,fontsize=14,fontweight='bold',va='top')
        block(2,34,22,42,'DGX FP16 state','Same full-sequence\nboundary state')
        block(34,67,29,22,'Native FP16','Original hook = split replay',accent=BLUE)
        block(34,39,29,22,'FP32 realized','FP16-rounded endpoint',accent=TEAL)
        block(34,11,29,22,'FP32 intended','Unrounded FP32 endpoint',accent=PURPLE)
        for yy in (78,50,22):arrow(24,55,33,yy)
        block(73,37,25,40,'Question-paired T','95% ΔT CI not fully\nwithin ±0.005',accent=ORANGE)
        for yy in (78,50,22):arrow(63,yy,72,56)
    save(fig,out,'00_architecture' if kind=='arch' else '00_validation_pipeline')


def batch_note(out):
    fig,ax=plt.subplots(figsize=(10,3.85))
    ax.set_xlim(0,17);ax.set_ylim(0,4.3);ax.axis('off')
    ax.text(0,4.15,'DGX v2: seed123 trailing-batch control comparability limitation',fontsize=13,fontweight='bold',color=NAVY,va='top')
    for j in range(17):
        x=j+.03
        rect=FancyBboxPatch((x,1.3),.89,1.15,boxstyle='round,pad=0.015,rounding_size=.15',
                            facecolor=BLUE if j<15 else ORANGE,edgecolor='white',lw=1)
        ax.add_patch(rect)
        ax.text(x+.445,1.86,'C' if j==0 else str(j),color='white',ha='center',va='center',fontsize=10,fontweight='bold')
    ax.text(7.55,.86,'Candidate + controls 1–14 · batch size 8',ha='center',color=BLUE,fontsize=10)
    ax.text(15.95,.86,'controls\n15–16 · batch 4',ha='center',color=ORANGE,fontsize=9)
    ax.text(0,.1,'Only seed123 affected. Existing saved T was not corrected; size and direction of possible bias are unknown.',fontsize=9.4,color=GRAY)
    save(fig,out,'07_batch_shape_limitation')


def make_all():
    plot_theme()
    _,_,w,p,_=load_and_effects()
    out=project_root()/'figures'/'rendered'
    diagram(out,'arch');diagram(out,'val')
    effect_forest(out,w,p)
    question_level(out,p,w)
    precision_delta(out,p)
    endpoint_rounding(out,p)
    magnitude(out,w)
    mode_shift(out,p,w)
    batch_note(out)
    model_core_arch(out)
    print('Generated figures:', len(list(out.glob('*.png'))), 'PNG /',len(list(out.glob('*.svg'))),'SVG')


def model_core_arch(out: Path):
    """Schematic from pinned modeling_ouro.py (SHA256 c5c68fbb...)."""
    plot_theme()
    fig, ax = plt.subplots(figsize=(12.2, 4.3))
    ax.set_xlim(0,100);ax.set_ylim(0,100);ax.axis('off')
    ax.text(2,97,'Pinned OuroModel: shared decoder layers across recurrent steps', fontsize=14.1,
            color=NAVY,fontweight='bold',va='top')
    def node(x,y,w,h,head,sub,stroke=BLUE):
        patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.35,rounding_size=3',
                            facecolor='#f3f7fb',edgecolor='#cbd7e5',lw=1.2)
        ax.add_patch(patch)
        ax.plot([x+2,x+11],[y+h-4,y+h-4],color=stroke,lw=3,solid_capstyle='round')
        ax.text(x+2,y+h-11,head,fontsize=10,fontweight='bold',color=NAVY,va='top')
        ax.text(x+2,y+h-27,sub,fontsize=8.7,color=GRAY,va='top',linespacing=1.5)
    def arrow(a,b,c,d,color=MID,rad=0):
        ax.add_patch(FancyArrowPatch((a,b),(c,d),arrowstyle='-|>',mutation_scale=16,
            connectionstyle=f'arc3,rad={rad}',color=color,lw=1.5))
    node(2,28,18,42,'Token embed','Input IDs → hidden\nsequence')
    node(26,28,24,42,'Shared decoder stack','Configured layers run\nin each UT step',PURPLE)
    node(56,28,17,42,'RMSNorm','Per-step full\nsequence state',TEAL)
    node(79,28,19,42,'LM head','Selected step state\n→ token logits',ORANGE)
    arrow(20,49,25,49);arrow(50,49,55,49);arrow(73,49,78,49)
    arrow(64,28,64,12,PURPLE)
    arrow(64,12,36,12,PURPLE)
    arrow(36,12,36,27,PURPLE)
    ax.text(50,4,'Loop over shared decoder layers, then normalize (total_ut_steps = 4 in archived config)',
            ha='center',fontsize=8.9,color=PURPLE)
    ax.text(63,84,'Early-exit gate is computed at each step; fixed-step readout is used in this pilot',
            ha='center',fontsize=9,color=GRAY)
    save(fig,out,'00_model_core_schematic')

if __name__ == '__main__':
    make_all()

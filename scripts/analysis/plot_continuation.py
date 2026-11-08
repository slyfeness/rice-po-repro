"""Plot redacted review metrics only; never imports a model or reads trajectories."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean


def render(evidence_path,retrieval_path,output):
    evidence=json.loads(Path(evidence_path).read_text('utf8'))
    retrieval=json.loads(Path(retrieval_path).read_text('utf8'))
    rows={m:sorted(evidence['training'][m],key=lambda r:r['global_update']) for m in ('grpo','ricepo')}
    for method,rs in rows.items():
        if [r['global_update'] for r in rs]!=[1,2,3] or any(not r['optimizer_step_committed'] for r in rs):
            raise ValueError('three actual committed updates required for each method')
        if rs[-1]['development']['status']!='complete':raise ValueError('complete final development required')
    initial=retrieval['initial']['summary']['mean_task_score']
    data={m:[dict(update=r['global_update'],mean_training_utility=fmean(r['original_training_utilities']),
        grad_norm_before_clip=r['update']['optimizer']['grad_norm_before_clip'],
        development_utility=(r['development']['summary']['macro'] if r['development']['status']=='complete' else None))
        for r in rs] for m,rs in rows.items()}
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    names=['CURVES.png','CURVES.svg','CURVE_DATA.json','CURVE_PROVENANCE.json']
    if any((out/name).exists() for name in names):raise FileExistsError('figure outputs already exist')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles={'grpo':('#0072B2','s'),'ricepo':('#D55E00','o')}
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.hashsalt':'ricepo-three-update-review'}):
        fig,axes=plt.subplots(2,2,figsize=(10,6.8),layout='constrained')
        train,dev,credit,gradient=axes.flat
        for m,rs in data.items():
            color,marker=styles[m];x=[r['update'] for r in rs]
            train.plot(x,[r['mean_training_utility'] for r in rs],color=color,marker=marker,label=m.upper())
            gradient.plot(x,[r['grad_norm_before_clip'] for r in rs],color=color,marker=marker,label=m.upper())
            # There is no development observation at update two. Do not connect over it.
            dx=[0]+[r['update'] for r in rs if r['development_utility'] is not None]
            dy=[initial]+[r['development_utility'] for r in rs if r['development_utility'] is not None]
            dev.scatter(dx,dy,color=color,marker=marker,s=38,label=m.upper(),zorder=3)
        train.set(title='A  Training batches (different fixed queries)',ylabel='Mean failure-adjusted training utility',
                  xlabel='Global update',xticks=[1,2,3],ylim=(0,None))
        dev.axvspan(1.75,2.25,color='#EEEEEE',zorder=0)
        dev.text(2,.60,'No evaluation\nat update 2',ha='center',va='center',fontsize=8)
        d0=retrieval['initial']['summary']['mean_d0']
        dev.axhline(d0,color='#555555',linestyle='--',linewidth=1,label='Fixed D0 NDCG@10')
        dev.set(title='B  Frozen complete development set (32 queries)',ylabel='Task utility / D0 reference [0, 1]',
                xlabel='Global update (0 = initial model)',xticks=[0,1,2,3],ylim=(0,.70))
        for j,(key,label,hatch,color) in enumerate([
                ('selected','Selected','', '#CCCCCC'),('estimable','Estimable','//','#0072B2'),
                ('nonzero_local','Nonzero local','..','#D55E00'),('propagation_count','Propagated (incl. last round)','xx','#009E73')]):
            credit.bar([r['global_update']+(j-1.5)*.18 for r in rows['ricepo']],
                       [r['credit'][key] for r in rows['ricepo']],width=.18,label=label,
                       hatch=hatch,color=color,edgecolor='#333333',linewidth=.4)
        credit.set(title='C  RICE counterfactual credit (actual counts)',ylabel='Anchors',xlabel='Global update',
                   xticks=[1,2,3],ylim=(0,20))
        gradient.set(title='D  Combined objective gradient (before clipping)',ylabel='L2 gradient norm',
                     xlabel='Global update',xticks=[1,2,3],ylim=(0,None))
        for ax in (train,dev,gradient):ax.legend(fontsize=7,frameon=False)
        credit.legend(fontsize=7,frameon=False,ncol=2)
        for ax in axes.flat:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        fig.suptitle('Independent two-stage / failure-zero variant; one training seed',fontsize=12)
        fig.supxlabel('Update 2 for RICE uses verified recovery. No smoothing, missing-point interpolation, or replicate uncertainty estimates.',fontsize=8)
        fig.savefig(out/'CURVES.png',dpi=180,metadata={'Software':'Matplotlib; redacted RICE-PO review metrics'})
        fig.savefig(out/'CURVES.svg',metadata={'Creator':'RICE-PO review plot','Date':None})
        plt.close(fig)
    svg=out/'CURVES.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text('utf8').splitlines())+'\n',encoding='utf8',newline='\n')
    (out/'CURVE_DATA.json').write_text(json.dumps(dict(initial_utility=initial,d0=d0,training=data),indent=2)+'\n',encoding='utf8',newline='\n')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    provenance=dict(inputs={Path(p).name:sha(p) for p in (evidence_path,retrieval_path)},
        script_sha256=sha(__file__),matplotlib=matplotlib.__version__,
        units='Raw [0,1] utility; missing development at update 2 is null, not zero.',
        scope='Public redacted aggregate presentation only; no model, dataset, private trajectory or GPU computation.',
        files={name:sha(out/name) for name in names[:-1]})
    (out/'CURVE_PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf8',newline='\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',required=True);parser.add_argument('--retrieval',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    render(args.evidence,args.retrieval,args.output)

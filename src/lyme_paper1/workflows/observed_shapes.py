'Calculate conditional output ranges using observed host-burden distributions.'
import collections
import csv
import datetime
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

BASE=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(BASE/'.mplconfig'))
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch

ROOT=BASE.parents[2]
SOURCE=ROOT/'data/raw/cary_related/Larvae_onHosts_91-23.csv'
F=np.linspace(.05,.95,31);P=F.copy();EX=np.array([.2,.5,.8]);ETAS=[0.,.5]
SF=['reciprocal','exponential','inverse_sqrt'];QF=['rational','exponential']
VERSIONS=['animal_augsep','capture_clean_augsep','capture_source_augsep','animal_julaugsep']

def writer(path,fields):
    handle=(gzip.open(path,'wt',encoding='utf-8-sig',newline='') if str(path).endswith('.gz')
            else path.open('w',encoding='utf-8-sig',newline=''))
    out=csv.DictWriter(handle,fieldnames=fields);out.writeheader();return handle,out

def feeding(family,z,reference):
    f=reference[:,None]
    if family=='reciprocal':return 1/(1+(1/f-1)*z[None,:])
    if family=='exponential':return np.exp(np.log(f)*z[None,:])
    if family=='inverse_sqrt':return 1/np.sqrt(1+(f**-2-1)*z[None,:])
    raise ValueError(family)

def infection(family,z,eta,reference):
    p=reference[:,None];floor=eta*p
    if family=='rational':
        b=(p-floor)/(1-p);return floor+(1-floor)*b*z[None,:]/(1+b*z[None,:])
    b=-np.log((1-p)/(1-floor))
    return floor+(1-floor)*(-np.expm1(-b*z[None,:]))

raw=list(csv.DictReader(SOURCE.open(encoding='utf-8-sig')))
all_identity_counts=collections.Counter(tuple(r[k].strip() for k in ['Year','Grid','Day1','Spp','Tag']) for r in raw)
records=[];identity_counts=collections.Counter();full_duplicate=collections.Counter(tuple(r.items()) for r in raw)
for i,r in enumerate(raw,2):
    d={**r,'source_row':i}
    try:
        day=datetime.date.fromisoformat(r['Day1']);year=int(r['Year'])
        d['date_valid']=day.year==year;d['month']=day.month
    except ValueError:d['date_valid']=False;d['month']=-1;year=int(r['Year'])
    try:
        count=float(r['Larvae']);d['count_valid']=math.isfinite(count) and count>=0 and count.is_integer()
    except ValueError:count=float('nan');d['count_valid']=False
    d['count']=count;d['year']=year;d['species_valid']=r['Spp'] in ['PL','TS']
    d['tag_valid']=r['Tag'].strip().lower() not in ['', 'na','nan']
    key=(year,r['Grid'],r['Day1'],r['Spp'],r['Tag'].strip())
    d['identity_date_key']=key
    if d['tag_valid']:identity_counts[key]+=1
    records.append(d)

qc_fields=['source_row','Year','Grid','Day1','Spp','Tag','Larvae','species_valid','date_valid','count_valid',
           'tag_valid','duplicate_identity_date','primary_included','primary_exclusion_reason']
qh,qw=writer(BASE/'qc_record_audit.csv.gz',qc_fields)
groups={v:collections.defaultdict(list) for v in VERSIONS}
inventory=set();exclusions=collections.Counter()
for d in records:
    duplicate=identity_counts[d['identity_date_key']]>1
    basic=d['species_valid'] and d['date_valid'] and d['count_valid']
    clean=basic and d['tag_valid'] and not duplicate
    augsep=d['month'] in [8,9];julaugsep=d['month'] in [7,8,9]
    key=(d['year'],d['Grid'],d['Spp'])
    if d['species_valid']:inventory.add(key)
    reason='included'
    for condition,label in [(not d['species_valid'],'invalid_species'),(not d['date_valid'],'invalid_date_or_year'),
                            (not d['count_valid'],'invalid_count'),(not augsep,'outside_augsep'),
                            (not d['tag_valid'],'missing_tag'),(duplicate,'duplicate_identity_date')]:
        if condition:reason=label;break
    included=clean and augsep;exclusions[reason]+=1
    qw.writerow({**{k:d[k] for k in qc_fields[:7]},**{k:d[k] for k in ['species_valid','date_valid','count_valid','tag_valid']},
                 'duplicate_identity_date':duplicate,'primary_included':included,'primary_exclusion_reason':reason})
    if included:
        groups['animal_augsep'][key].append(d);groups['capture_clean_augsep'][key].append(d)
    if basic and augsep:groups['capture_source_augsep'][key].append(d)
    if clean and julaugsep:groups['animal_julaugsep'][key].append(d)
qh.close()

profiles={v:{} for v in VERSIONS};host_rows=[]
for version in VERSIONS:
    for key,rs in groups[version].items():
        if version.startswith('animal'):
            animals=collections.defaultdict(list)
            for r in rs:animals[r['Tag'].strip()].append(r)
            units=[]
            for tag,rr in sorted(animals.items()):
                mean=float(np.mean([r['count'] for r in rr]));units.append((tag,mean))
                host_rows.append({'version':version,'year':key[0],'grid':key[1],'species':key[2],'tag':tag,
                    'n_captures':len(rr),'mean_observed_burden':mean,'first_date':min(r['Day1'] for r in rr),
                    'last_date':max(r['Day1'] for r in rr)})
        else:units=[('source_row:'+str(r['source_row']),r['count']) for r in rs]
        values=np.array([v for u,v in units]);profiles[version][key]={'units':units,'values':values,
            'mean':float(values.mean()),'n':len(values),'n_captures':len(rs)}

hh,hw=writer(BASE/'animal_observation_means.csv',list(host_rows[0]))
hw.writerows(host_rows);hh.close()
qualification=[];eligible=[]
for key in sorted(inventory):
    profile=profiles['animal_augsep'].get(key)
    if profile is None:reason='no_clean_augsep_records'
    elif profile['n']<10:reason='fewer_than_10_identified_animals'
    elif profile['mean']==0:reason='all_zero_captured_burdens'
    else:reason='eligible';eligible.append(key)
    qualification.append({'year':key[0],'grid':key[1],'species':key[2],
        'primary_n_animals':profile['n'] if profile else 0,
        'primary_n_captures':profile['n_captures'] if profile else 0,
        'primary_mean':profile['mean'] if profile else '', 'qualification':reason})
qh,qw=writer(BASE/'group_qualification.csv',list(qualification[0]));qw.writerows(qualification);qh.close()

shape_rows=[];pooled={};normalized_fields=['version','species','year','grid','unit','observed_burden',
    'group_mean','normalized_z','within_group_weight','pooled_year_grid_weight','pooled_weight']
nh,nw=writer(BASE/'normalized_observed_shapes.csv.gz',normalized_fields)
for version in VERSIONS:
    for species in ['PL','TS']:
        keys=[k for k in eligible if k[2]==species];by_year=collections.Counter(k[0] for k in keys)
        zs=[];ws=[]
        for key in keys:
            profile=profiles[version][key];values=profile['values'];mu=profile['mean']
            assert mu>0
            z=values/mu;assert abs(z.mean()-1)<1e-12
            group_weight=1/len(by_year)/by_year[key[0]]
            weight=group_weight/profile['n'];zs.extend(z);ws.extend(np.repeat(weight,profile['n']))
            order=np.sort(values);n=len(values)
            gini=float(np.dot(2*np.arange(1,n+1)-n-1,order)/(n*order.sum()))
            shape_rows.append({'version':version,'year':key[0],'grid':key[1],'species':species,
                'n_shape_units':n,'n_captures':profile['n_captures'],'mean_observed_burden':mu,
                'normalized_cv':float(z.std()),'zero_fraction':float(np.mean(values==0)),
                'gini_observed_shape':gini,'top_decile_share':float(order[-math.ceil(.1*n):].sum()/values.sum()),
                'max_normalized_z':float(z.max()),'pooled_year_grid_weight':group_weight})
            for (unit,value),zz in zip(profile['units'],z):
                nw.writerow({'version':version,'species':species,'year':key[0],'grid':key[1],
                    'unit':unit,'observed_burden':value,'group_mean':mu,'normalized_z':float(zz),
                    'within_group_weight':1/n,'pooled_year_grid_weight':group_weight,'pooled_weight':weight})
        zs=np.asarray(zs);ws=np.asarray(ws);assert abs(ws.sum()-1)<1e-12
        assert abs(np.dot(zs,ws)-1)<1e-12
        unique,inverse=np.unique(zs,return_inverse=True);weights=np.bincount(inverse,weights=ws)
        pooled[(version,species)]={'z':unique,'w':weights,'n_groups':len(keys),'n_years':len(by_year),
            'n_shape_units':len(zs),'normalization_mean':float(np.dot(unique,weights)),
            'zero_fraction':float(weights[unique==0].sum())}
nh.close();sh,sw=writer(BASE/'shape_group_summary.csv',list(shape_rows[0]));sw.writerows(shape_rows);sh.close()

def evaluate(z,w,f,p,eta):
    assert abs(w.sum()-1)<1e-12 and abs(np.dot(z,w)-1)<1e-12
    result=[]
    for sf in SF:
        s=feeding(sf,z,f);h=s*z[None,:];eh=h@w;rn=eh/f
        control=(h*w[None,:])@np.broadcast_to(p[:,None],(len(p),len(z))).T/(f[:,None]*p[None,:])
        assert np.allclose(control,rn[:,None],rtol=1e-12,atol=1e-12)
        for qf in QF:
            q=infection(qf,z,eta,p)
            assert np.all((q>=0)&(q<=1)) and np.all((s>=0)&(s<=1))
            ehq=(h*w[None,:])@q.T;ri=ehq/(f[:,None]*p[None,:])
            assert np.all(ri<=rn[:,None]/p[None,:]+1e-10)
            eq=q@w;cov=((h-eh[:,None])*w[None,:])@(q-eq[:,None]).T
            assert np.allclose(ehq,eh[:,None]*eq[None,:]+cov,rtol=1e-10,atol=1e-12)
            result.append({'sf':sf,'qf':qf,'RI':ri,'RN':rn})
    return result

def codes(results):
    ri=np.stack([r['RI'] for r in results]);rn=np.stack([np.broadcast_to(r['RN'][:,None],r['RI'].shape) for r in results])
    increase=np.all(ri>1.01,axis=0);decrease=np.all(ri<.99,axis=0)
    conflict=np.any(ri>1.01,axis=0)&np.any(ri<.99,axis=0)
    code=np.zeros(increase.shape,dtype=int);code[increase]=1;code[decrease]=-1;code[conflict]=2
    return code,ri,rn

cache={};grid_rows=[];fixed_pooled=[]
for version in VERSIONS:
    for species in ['PL','TS']:
        shape=pooled[(version,species)]
        for eta in ETAS:
            result=evaluate(shape['z'],shape['w'],F,P,eta);code,ri,rn=codes(result)
            cache[(version,species,eta)]={'results':result,'code':code,'RI':ri,'RN':rn}
            for m,r in enumerate(result):
                for i,f in enumerate(F):
                    for j,p in enumerate(P):
                        grid_rows.append({'version':version,'species':species,'floor_fraction':eta,
                            'feeding_family':r['sf'],'infection_family':r['qf'],'f':float(f),'p':float(p),
                            'R_infected':float(ri[m,i,j]),'R_all_nymphs':float(rn[m,i,j]),
                            'model_set_consensus_code':int(code[i,j])})
            for f in EX:
                for p in EX:
                    i=int(np.argmin(abs(F-f)));j=int(np.argmin(abs(P-p)))
                    fixed_pooled.append({'version':version,'species':species,'floor_fraction':eta,
                        'f':float(f),'p':float(p),'min_R_infected':float(ri[:,i,j].min()),
                        'max_R_infected':float(ri[:,i,j].max()),'min_R_all_nymphs':float(rn[:,i,j].min()),
                        'max_R_all_nymphs':float(rn[:,i,j].max()),'consensus_code':int(code[i,j]),
                        'all_six_infected_up_all_nymphs_down':bool(np.all((ri[:,i,j]>1.01)&(rn[:,i,j]<.99)))})
    print('Completed pooled shape:',version,flush=True)
gh,gw=writer(BASE/'pooled_full_grid.csv.gz',list(grid_rows[0]));gw.writerows(grid_rows);gh.close()
fh,fw=writer(BASE/'pooled_fixed_nine_scenarios.csv',list(fixed_pooled[0]));fw.writerows(fixed_pooled);fh.close()

cell_summary=[];detail_fields=['version','year','grid','species','floor_fraction','feeding_family','infection_family','f','p','R_infected','R_all_nymphs']
dh,dw=writer(BASE/'per_cell_fixed_models.csv.gz',detail_fields)
for version in VERSIONS:
    for key in eligible:
        profile=profiles[version][key];z=profile['values']/profile['mean'];w=np.ones(len(z))/len(z)
        for eta in ETAS:
            result=evaluate(z,w,EX,EX,eta);code,ri,rn=codes(result)
            for i,f in enumerate(EX):
                for j,p in enumerate(EX):
                    cell_summary.append({'version':version,'year':key[0],'grid':key[1],'species':key[2],
                        'floor_fraction':eta,'f':float(f),'p':float(p),'min_R_infected':float(ri[:,i,j].min()),
                        'max_R_infected':float(ri[:,i,j].max()),'min_R_all_nymphs':float(rn[:,i,j].min()),
                        'max_R_all_nymphs':float(rn[:,i,j].max()),'consensus_code':int(code[i,j]),
                        'all_six_infected_up_all_nymphs_down':bool(np.all((ri[:,i,j]>1.01)&(rn[:,i,j]<.99)))})
                    for m,r in enumerate(result):
                        dw.writerow({'version':version,'year':key[0],'grid':key[1],'species':key[2],
                            'floor_fraction':eta,'feeding_family':r['sf'],'infection_family':r['qf'],
                            'f':float(f),'p':float(p),'R_infected':float(ri[m,i,j]),'R_all_nymphs':float(rn[m,i,j])})
    print('Completed individual cells:',version,flush=True)
dh.close();ch,cw=writer(BASE/'per_cell_consensus_fixed.csv',list(cell_summary[0]));cw.writerows(cell_summary);ch.close()

version_audit=[]
for species in ['PL','TS']:
    for eta in ETAS:
        primary=cache[('animal_augsep',species,eta)]['code']
        for version in VERSIONS[1:]:
            alternative=cache[(version,species,eta)]['code']
            version_audit.append({'species':species,'floor_fraction':eta,'alternative_version':version,
                'matched_grid_cells':primary.size,'same_consensus_code':int((primary==alternative).sum()),
                'opposite_unanimous_sign':int(((primary==1)&(alternative==-1)|(primary==-1)&(alternative==1)).sum()),
                'lost_unanimous_direction':int(((abs(primary)==1)&(abs(alternative)!=1)).sum())})
vh,vw=writer(BASE/'pooled_representation_sensitivity.csv',list(version_audit[0]));vw.writerows(version_audit);vh.close()

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
    'font.size':10.5,'axes.labelsize':10.5,'axes.titlesize':10.5,'xtick.labelsize':9.5,'ytick.labelsize':9.5,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8,'svg.fonttype':'none',
    'pdf.fonttype':42,'legend.fontsize':9.5,'figure.facecolor':'white'})
figure_checks=[]
def savefig(fig,stem):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[];texts=[]
    for obj in fig.findobj(matplotlib.text.Text):
        if obj.get_visible() and obj.get_text():
            bb=obj.get_window_extent(renderer)
            if bb.x0<fig.bbox.x0-1 or bb.y0<fig.bbox.y0-1 or bb.x1>fig.bbox.x1+1 or bb.y1>fig.bbox.y1+1:outside.append(obj.get_text())
    for ax in fig.axes:
        texts.extend(ax.get_xticklabels()+ax.get_yticklabels()+list(ax.texts)+[ax.xaxis.label,ax.yaxis.label,ax.title])
        if ax.get_legend():texts.extend(ax.get_legend().get_texts())
    texts.extend(fig.texts)
    for lg in fig.legends:texts.extend(lg.get_texts())
    texts=[t for t in texts if t.get_visible() and t.get_text()];collisions=[]
    for i,t in enumerate(texts):
        bb=t.get_window_extent(renderer).padded(-.5)
        for other in texts[i+1:]:
            if bb.overlaps(other.get_window_extent(renderer).padded(-.5)):collisions.append([t.get_text(),other.get_text()])
    assert not outside,(stem,outside)
    assert not collisions,(stem,collisions)
    for ext in ['pdf','svg','png']:fig.savefig(BASE/f'{stem}.{ext}',dpi=600)
    figure_checks.append({'figure':stem,'width_mm':173,'outside_text':outside,'text_collision_pairs':collisions})
    plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(173/25.4,145/25.4),sharex=True,sharey=True)
fig.subplots_adjust(left=.15,right=.985,bottom=.225,top=.91,wspace=.19,hspace=.43)
cmap=ListedColormap(['#3775BA','#D9D9D9','#B64342','#9A4D8E']);norm=BoundaryNorm([-1.5,-.5,.5,1.5,2.5],4)
names={'PL':'P. leucopus','TS':'T. striatus'}
for i,species in enumerate(['PL','TS']):
    for j,eta in enumerate(ETAS):
        ax=axes[i,j];code=cache[('animal_augsep',species,eta)]['code']
        ax.pcolormesh(P,F,code,cmap=cmap,norm=norm,shading='nearest',rasterized=True)
        ax.set_title(names[species]+f': q(0) = {eta:g} × p',pad=8)
        ax.text(-.17,1.16,chr(97+i*2+j),transform=ax.transAxes,fontweight='bold',fontsize=11.5)
        ax.set_xlim(.05,.95);ax.set_ylim(.05,.95);ax.set_xticks([.1,.5,.9]);ax.set_yticks([.1,.5,.9])
fig.supxlabel('Effective infection probability at mean input, p',y=.135,fontsize=10.5)
fig.supylabel('Feeding probability at mean input, f',x=.026,fontsize=10.5)
fig.legend([Patch(facecolor=c) for c in ['#3775BA','#B64342','#9A4D8E','#D9D9D9']],
    ['All six decrease','All six increase','Opposite signs','Mixed / near boundary'],
    loc='lower center',bbox_to_anchor=(.5,.015),ncol=2,frameon=False)
savefig(fig,'observed_shape_function_consensus')

fig,axes=plt.subplots(1,2,figsize=(173/25.4,92/25.4),sharey=True)
fig.subplots_adjust(left=.105,right=.975,bottom=.22,top=.855,wspace=.25)
colors=['#0F4D92','#B64342','#42949E','#9A4D8E']
for species,color in zip(['PL','TS'],colors):
    d=pooled[('animal_augsep',species)];axes[0].step(np.log1p(d['z']),np.cumsum(d['w']),where='post',color=color,label=names[species],lw=1.4)
labels=['Animal means (Aug–Sep)','Capture records (clean)','Capture records (source)','Animal means (Jul–Sep)']
for version,color,label in zip(VERSIONS,colors,labels):
    d=pooled[(version,'PL')];axes[1].step(np.log1p(d['z']),np.cumsum(d['w']),where='post',color=color,label=label,lw=1.2)
for ax in axes:
    ticks=np.array([0.,1.,3.,10.,30.]);ax.set_xticks(np.log1p(ticks));ax.set_xticklabels(['0','1','3','10','30'])
    ax.set_xlim(0,np.log1p(30));ax.set_ylim(0,1.01);ax.set_yticks([0,.5,1]);ax.set_xlabel('Normalized observed burden, z')
    ax.legend(frameon=False,loc='lower right',fontsize=9)
axes[0].set_ylabel('Weighted cumulative proportion')
axes[0].set_title('Species profiles',pad=8);axes[1].set_title('Representation sensitivity',pad=8)
for ax,letter in zip(axes,['a','b']):ax.text(-.13,1.12,letter,transform=ax.transAxes,fontweight='bold',fontsize=11.5)
savefig(fig,'observed_input_shape_profiles')

summary={'analysis_type':'Conditional scenarios using observed host-burden distributions',
    'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'exact_prior_kernel_match':'pass','raw_records':len(raw),
    'raw_missing_tags':sum(not d['tag_valid'] for d in records),
    'raw_exact_duplicate_excess':sum(n-1 for n in full_duplicate.values()),
    'raw_all_identity_date_duplicate_rows':sum(all_identity_counts[tuple(r[k].strip() for k in ['Year','Grid','Day1','Spp','Tag'])]>1 for r in raw),
    'raw_known_tag_duplicate_identity_date_rows':sum(identity_counts[d['identity_date_key']]>1 for d in records if d['tag_valid']),
    'primary_exclusion_counts':dict(exclusions),'qualification_counts':dict(collections.Counter(r['qualification'] for r in qualification)),
    'pooled_shapes':[{k:v for k,v in d.items() if k not in ['z','w']}|{'version':version,'species':species}
        for (version,species),d in pooled.items()],
    'pooled_model_grid_rows':len(grid_rows),'per_cell_consensus_rows':len(cell_summary),
    'fixed_pooled_scenarios':fixed_pooled,'representation_sensitivity':version_audit,
    'figure_checks':figure_checks,'constant_q_and_probability_bound_checks':'pass',
    'normalized_shape_mean_checks':'pass','weighted_covariance_identity_checks':'pass',
    'author_contact':False,'new_predictive_fit':False,'parameters_fitted_from_observations':False}
(BASE/'results.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='fixed_pooled_scenarios'},indent=2,ensure_ascii=False))

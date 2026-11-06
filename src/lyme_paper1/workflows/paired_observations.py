'Summarize Norwegian host records, Cary burden distributions and Jacquet nymph assays.'
from pathlib import Path
import csv, collections, gzip, hashlib, json, math, shutil
import openpyxl

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
KIT=OUT/'组图素材'; KIT.mkdir(exist_ok=True)
SHAPE=ROOT/'lyme/reviews/2026-10-14/empirical_shape_public_v1'
NOR=ROOT/'lyme/data/raw/external/paper3_Lindso_open_host_2026-10-24/Lindso2024/data.csv'
JAC=next((ROOT/'lyme/literature/paper3_infection_age_2026-10-26').rglob('*.xlsx'))
sources=[NOR,JAC,SHAPE/'normalized_observed_shapes.csv.gz',SHAPE/'pooled_fixed_nine_scenarios.csv',SHAPE/'per_cell_consensus_fixed.csv']
fingerprints={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
(OUT/'SOURCE_SHA256.json').write_text(json.dumps(fingerprints,indent=2))
def read(p,delimiter=','):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f,delimiter=delimiter))
def write(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
nor=read(NOR,';')
assert len(nor)==1391 and len({r['ID_long'] for r in nor})==1391
summary=[];strata_rows=[]
for variant in ['raw','author_cap']:
    pairs=[(float(r['NN_larva']),int(r['Bburg2'])) for r in nor]
    if variant=='author_cap':pairs=[(158. if x>300 else x,y) for x,y in pairs]
    n=len(pairs);b=math.fsum(x for x,y in pairs);k=sum(y for x,y in pairs)
    by=math.fsum(x*y for x,y in pairs);eb=b/n;ey=k/n;cov=by/n-eb*ey
    grouped=collections.defaultdict(list)
    for r,pair in zip(nor,pairs):grouped[(r['year_cat'],r['sp_eng'],r['season'])].append(pair)
    within=[];between=[]
    for (year,sp,season),v in sorted(grouped.items()):
        m=len(v);mb=math.fsum(x for x,y in v)/m;my=sum(y for x,y in v)/m
        cc=math.fsum(x*y for x,y in v)/m-mb*my
        within.append(m/n*cc);between.append(m/n*(mb-eb)*(my-ey))
        strata_rows.append(dict(version=variant,year=year,species=sp,season=season,n=m,
            mean_burden=mb,tissue_positive_fraction=my,covariance=cc,weight=m/n))
    w=math.fsum(within);v=math.fsum(between)
    assert abs(cov-w-v)<1e-12
    summary.append(dict(version=variant,hosts=n,positive_hosts=k,larval_count=b,
        larvae_on_positive_hosts=by,host_positive_fraction=ey,
        larval_weighted_tissue_positive_fraction=by/b,joint_mean=by/n,
        product_of_marginals=eb*ey,joint_to_product_ratio=(by/n)/(eb*ey),
        covariance=cov,within_covariance=w,between_covariance=v,strata=len(grouped)))
write(OUT/'Norway_summary.csv',summary);write(OUT/'Norway_group_decomposition.csv',strata_rows)
plotrows=[dict(weighting='Hosts equally',numerator=k,denominator=len(nor),percent=100*summary[0]['host_positive_fraction']),
          dict(weighting='Observed larval counts',numerator=summary[0]['larvae_on_positive_hosts'],denominator=summary[0]['larval_count'],percent=100*summary[0]['larval_weighted_tissue_positive_fraction'])]
write(KIT/'panel_b_Norway_weighting.csv',plotrows)

# ECDF retains the original full support and original hierarchical weights.
atoms=collections.defaultdict(lambda:collections.defaultdict(list))
with gzip.open(SHAPE/'normalized_observed_shapes.csv.gz','rt',encoding='utf-8-sig',newline='') as f:
    for r in csv.DictReader(f):
        if r['version']=='animal_augsep':atoms[r['species']][float(r['normalized_z'])].append(float(r['pooled_weight']))
ecdf=[]
for sp,values in atoms.items():
    cum=0.
    for z,ww in sorted(values.items()):
        mass=math.fsum(ww);cum+=mass
        ecdf.append(dict(species=sp,normalized_burden=z,probability_mass=mass,cumulative_probability=cum))
    assert abs(cum-1)<1e-11
write(KIT/'panel_a_Cary_weighted_ECDF.csv',ecdf)
descriptors=ROOT/'lyme/reviews/2026-10-26/paper1_supplied_review_revision_v1/observed_shape_descriptor_summary.csv'
shutil.copy2(descriptors,KIT/descriptors.name)
scenarios=[r for r in read(SHAPE/'pooled_fixed_nine_scenarios.csv') if r['version']=='animal_augsep']
write(OUT/'Cary_archived_primary_scenarios.csv',scenarios)
cells=[r for r in read(SHAPE/'per_cell_consensus_fixed.csv') if r['version']=='animal_augsep' and float(r['f'])==.8 and float(r['p'])==.2 and float(r['floor_fraction'])==0]
counts={sp:dict(n=sum(r['species']==sp for r in cells),all_six_separated=sum(r['species']==sp and r['all_six_infected_up_all_nymphs_down']=='True' for r in cells)) for sp in ['PL','TS']}
assert counts=={'PL':{'n':181,'all_six_separated':169},'TS':{'n':141,'all_six_separated':105}}

w=openpyxl.load_workbook(JAC,read_only=True,data_only=True);it=w['Sheet2'].iter_rows(values_only=True);head=next(it)
j=[dict(zip(head,row)) for row in it if any(v is not None for v in row)]
g=collections.Counter((r['mouse.id'],int(r['age3'])) for r in j)
assert len(j)==1174 and len(g)==113
byhost={r['mouse.id']:r for r in j}
ja=dict(ticks=len(j),unique_samples=len({r['sample.name'] for r in j}),hosts=len(byhost),
        observed_host_time_groups=len(g),five_possible_times=True,
        missing_host_time_groups=[dict(host=h,day=a) for h in sorted(byhost) for a in sorted({int(r['age3']) for r in j}) if (h,a) not in g],
        treatment_host_counts=dict(collections.Counter(r['treatment'] for r in byhost.values())),
        endpoint='PCR positivity in selected post-moult nymphs from infected study mice',
        burden_or_feeding_denominator_available=False)
write(OUT/'Jacquet_observation_inventory.csv',[dict(observed_mice=23,selected_nymph_assays=1174,observed_host_time_groups=113,possible_host_time_groups=115)])
out=dict(Norway=summary,Jacquet=ja,Cary_counts=counts,
    scope='Norwegian host-count summaries, Cary distribution summaries and the Jacquet assay inventory.')
(OUT/'EVIDENCE_QA.json').write_text(json.dumps(out,indent=2))
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in fingerprints.items())
print(json.dumps(out,indent=2))

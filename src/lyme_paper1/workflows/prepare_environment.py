'Aggregate Daymet observations by season.'
from pathlib import Path
import hashlib, json, io
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/paper1_environment_benchmark_finite_v1_2026-10-16'
OUT.mkdir(parents=True,exist_ok=True)
RAW=ROOT/'data/raw/external/climate_feasibility_2026-10-16/daymet_station_point_1988_2024.csv'
lines=RAW.read_text().splitlines();start=next(i for i,x in enumerate(lines) if x.startswith('year,yday'))
d=pd.read_csv(io.StringIO('\n'.join(lines[start:])))
d['date']=pd.to_datetime(d.year.astype(str)+'-01-01')+pd.to_timedelta(d.yday-1,unit='D')
assert not d.duplicated(['year','yday']).any() and d.date.is_unique
required=['tmax (deg c)','tmin (deg c)','vp (Pa)','prcp (mm/day)']
assert d[required].notna().all().all()
d['valid']=(d[required].notna().all(axis=1)&d['tmax (deg c)'].ge(d['tmin (deg c)'])
            &d['vp (Pa)'].ge(0)&d['prcp (mm/day)'].ge(0))
def es(t):return .6108*np.exp(17.27*t/(t+237.3))
unclipped=(es(d['tmax (deg c)'])+es(d['tmin (deg c)']))/2-d['vp (Pa)']/1000
d['approx_vpd_kPa']=np.maximum(unclipped,0)
d['in_window']=d.date.dt.month.isin([7,8,9])
rows=[]
for year in range(1988,2025):
    g=d[d.year.eq(year)&d.in_window].copy()
    dates=pd.date_range(f'{year}-07-01',f'{year}-09-30')
    complete=len(g)==92 and set(g.date)==set(dates) and g.valid.all()
    rows.append(dict(source_year=year,first_date=str(g.date.min().date()),last_date=str(g.date.max().date()),
                     n_days=len(g),n_valid=int(g.valid.sum()),complete_92=bool(complete),
                     summer_vpd_kPa=float(g.approx_vpd_kPa.mean()) if complete else np.nan,
                     summer_precip_mm=float(g['prcp (mm/day)'].sum()) if complete else np.nan))
s=pd.DataFrame(rows);s.to_csv(OUT/'seasonal_environment.csv',index=False)
d[d.in_window][['year','yday','date']+required+['valid','approx_vpd_kPa']].to_csv(OUT/'summer_daily_source.csv',index=False)
assert s[s.source_year.between(1991,2023)].complete_92.all()
audit=dict(source_path=str(RAW.relative_to(ROOT)),source_sha256=hashlib.sha256(RAW.read_bytes()).hexdigest(),
           raw_rows=len(d),
           citation_version='Daymet V4 R1',software_banner='4.0',all_required_years_complete=True,
           summer_negative_unclipped_vpd_days=int(((unclipped<0)&d.in_window).sum()),
           matched_source_years=list(range(1991,2024)),
           station_point=[41.785823,-73.741447],ecological_outcomes_loaded=False)
(OUT/'ENVIRONMENT_QA.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))

'Run supplementary analyses with separate input and result directories.'
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,os
SOURCES=Path(__file__).with_name('workflows')
JOBS={
 'build-annual-inputs':[('scripts/fit_cary_compensation_forecast.py','fit_cary_compensation_forecast.py')],
 'annual-relationships':[('scripts/annual_relationships.py','annual_relationships.py')],
 'counts': [('scripts/extend_cary_regression_models.py','extend_cary_regression_models.py'),('scripts/decompose_cary_burden_and_gains.py','decompose_cary_burden_and_gains.py')],
 'count-path-models': [('scripts/extend_cary_regression_models.py','extend_cary_regression_models.py')],
 'dispersion-bootstrap':[('scripts/bootstrap_cary_density_dispersion.py','bootstrap_cary_density_dispersion.py')],
 'matched-forecasts':[('scripts/benchmark_cary_published_don.py','benchmark_cary_published_don.py')],
 'forecast-controls':[('scripts/run_paper1_major_revision.py','run_paper1_major_revision.py')],
 'weather-ann':[('scripts/run_paper1_weather_ANN_revision.py','run_paper1_weather_ANN_revision.py')],
 'alignment':[('manuscript/paper1/submission_cn_2026-10-13/run_additional_evidence.py','additional_evidence.py')],
 'pcr-scales':[('scripts/check_pcr_scale_uncertainty.py','check_pcr_scale_uncertainty.py')],
 'shapes':[('reviews/2026-10-14/empirical_shape_public_v1/run_empirical_shapes.py','observed_shapes.py')],
 'environment':[('reviews/2026-10-16/paper1_environment_benchmark_finite_v1/prepare_environment.py','prepare_environment.py'),('reviews/2026-10-16/paper1_environment_benchmark_finite_v1/run_benchmark.py','environment.py')],
 'measurement':[('reviews/2026-11-02/paper1_measurement_design_v1/run.py','measurement_design.py'),('reviews/2026-11-02/paper1_measurement_design_v1/verify.py','verify_measurement.py')],
 'anchor-sensitivity':[('reviews/2026-11-02/paper1_anchor_robustness_v1/run.py','anchor_sensitivity.py'),('reviews/2026-11-02/paper1_anchor_robustness_v1/verify.py','verify_anchors.py')],
 'paired-observations':[('reviews/2026-10-26/paper1_empirical_bridge_integration_v1/prepare_evidence.py','paired_observations.py')],
 'descriptors':[('reviews/2026-10-26/paper1_supplied_review_revision_v1/describe_and_recompose.py','descriptors.py')],
 'infection-forecasts':[('scripts/infection_reproduction.py','infection_reproduction.py')],
}

def _sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(job,inputs,output):
 inputs=inputs.resolve();output=output.resolve()
 if output==inputs or inputs in output.parents or output in inputs.parents:raise ValueError('Output and input directories must not contain one another')
 if output.exists() and any(output.iterdir()):raise ValueError('Choose an empty output directory')
 manifest=json.loads((inputs/'supplementary_manifest.json').read_text())
 for row in manifest['files']:
  rel=Path(row['path'])
  if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe manifest path')
  p=inputs/rel
  if not p.is_file() or _sha(p)!=row['sha256']:raise ValueError('Missing or changed input: '+str(rel))
 output.mkdir(parents=True,exist_ok=True)
 work=output/'workspace';shutil.copytree(inputs/'supplementary',work)
 lyme=work/'lyme';(lyme/'results').mkdir(exist_ok=True)
 companions = {
  'weather-ann': ['benchmark_cary_published_don.py', 'extend_cary_regression_models.py'],
  'infection-forecasts': ['infection_functions.py'],
 }
 for name in companions.get(job, []):
  target = lyme/'scripts'/name
  target.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(SOURCES/name, target)
 stages=JOBS[job]
 for rel,name in stages:
  target=lyme/rel;target.parent.mkdir(parents=True,exist_ok=True)
  text=(SOURCES/name).read_text()
  if job=='counts':
   text=text.split('def vars_frame')[0] if name.startswith('extend_') else text.split("p=pd.read_csv(OUT/'chronological_predictions.csv')")[0]
  target.write_text(text)
  if '/reviews/' in str(target):
   (target.parent/'results').mkdir(exist_ok=True)
  env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','MPLCONFIGDIR':str(output/'matplotlib-cache')}
  with (output/(name+'.log')).open('w') as log:
   proc=subprocess.run([sys.executable,str(target)],cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT)
  if proc.returncode:raise RuntimeError('Analysis failed; see '+str(output/(name+'.log')))
 changed=[]
 for p in work.rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:
   rel=p.relative_to(work);old=inputs/'supplementary'/rel
   if not old.exists() or _sha(p)!=_sha(old):changed.append(str(rel))
 report={'analysis':job,'completed':True,'output_files':changed,'input_files_verified':len(manifest['files']),'scope':'Analysis definitions and required data are listed in docs/COVERAGE.md.'}
 (output/'RUN_REPORT.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2));return 0

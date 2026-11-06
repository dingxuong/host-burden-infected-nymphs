"""Build primary forecast inputs from provider tables."""
from pathlib import Path
import hashlib, json
import pandas as pd

def build(long_table, burdens, data):
    if data.exists() and any(data.iterdir()):
        raise ValueError("Use an empty destination for primary-data preparation")
    raw = pd.read_csv(long_table)
    records = pd.read_csv(burdens)
    grid = {"Green Control":"GC", "Green Experimental":"GE", "Henry Control":"HC",
            "Henry Experimental":"HE", "Tea Control":"TC", "Tea Experimental":"TE"}
    records["Plot"] = records.Grid.map(grid)
    # Calculate mean burden across captures, using all recorded months.
    agg = records.groupby(["Year","Plot","Spp"]).Larvae.mean().unstack("Spp").reset_index()
    agg = agg.rename(columns={"PL":"burden_m", "TS":"burden_c"})
    annual = raw[raw.Plot.notna()].copy()
    source = annual[["Year","Plot","PL_t","TS_t","larval_peak_t","cum_gdd0_oct_sep_t"]].merge(
        agg[["Year","Plot","burden_m","burden_c"]], on=["Year","Plot"], validate="one_to_one")
    source = source.rename(columns={"Year":"source_year", "PL_t":"mouse", "TS_t":"chip",
                                    "larval_peak_t":"larvae", "cum_gdd0_oct_sep_t":"past_gdd"})
    source = source[source.source_year.between(1991,2023)].copy()
    source["outcome_year"] = source.source_year+1
    paired = source.merge(annual[["Year","Plot","nymph_peak_t"]],
                          left_on=["outcome_year","Plot"], right_on=["Year","Plot"],validate="one_to_one")
    paired = paired.dropna(subset=["mouse","chip","larvae","burden_m","burden_c","past_gdd","nymph_peak_t"]).copy()
    paired["fed_m"] = paired.mouse*paired.burden_m
    paired["fed_c"] = paired.chip*paired.burden_c
    paired["outcome"] = paired.nymph_peak_t*450
    cols = ["source_year","outcome_year","Plot","mouse","chip","larvae","past_gdd","fed_m","fed_c","outcome"]
    units = paired[cols].sort_values(["outcome_year","Plot"])
    if len(units)!=174:
        raise ValueError(f"'Expected 174 complete plot-years; found '{len(units)}")
    folder = data/"forecast"; folder.mkdir(parents=True,exist_ok=True)
    out = folder/"units.csv";units.to_csv(out,index=False)
    sha = lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest = {"schema_version":1,"scope":'Primary forecast data from the two source tables',
                "files":[{"path":"forecast/units.csv","sha256":sha(out),"bytes":out.stat().st_size}],
                "provider_files":[{"name":Path(x).name,"sha256":sha(x)} for x in [long_table,burdens]]}
    (data/"manifest.json").write_text(json.dumps(manifest,indent=2))
    return {"n_units":len(units),"source_years":[int(units.source_year.min()),int(units.source_year.max())],
            "input_kind":'Plot-years paired with next-year nymph observations; complete cases only'}

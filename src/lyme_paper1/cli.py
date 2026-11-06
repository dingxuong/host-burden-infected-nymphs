'Commands for forecasting, site selection and response-function bounds.'
import argparse, json, subprocess, sys, shutil
from pathlib import Path

ENGINE = Path(__file__).resolve().parent / "_engines"

def _main():
    parser = argparse.ArgumentParser(prog="lyme-paper1")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help='Calculate the exact A/B example')
    demo.add_argument("--output", type=Path)
    prep = sub.add_parser("prepare", help="Copy a verified prepared input pack")
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--data", type=Path, default=Path("data"))
    raw = sub.add_parser("prepare-primary", help='Prepare primary forecast data from the two source CSVs')
    raw.add_argument("--long-table", type=Path, required=True)
    raw.add_argument("--burdens", type=Path, required=True)
    raw.add_argument("--data", type=Path, default=Path("data"))
    chk = sub.add_parser("check-data", help="Check required pack checksums")
    chk.add_argument("--data", type=Path, default=Path("data"))
    for name in ["primary", "diagnostics", "selection", "certificates", "interval-bounds", "verify-selection"]:
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, default=Path("data"))
        p.add_argument("--output", type=Path, default=Path("outputs") / name)
        if name == "verify-selection":
            p.add_argument("--results", type=Path, required=True)
    from .supplementary import JOBS
    repro = sub.add_parser("reproduce", help='Run a supplementary analysis')
    repro.add_argument("analysis", choices=sorted(JOBS))
    repro.add_argument("--inputs",type=Path,required=True)
    repro.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.command == "reproduce":
        from .supplementary import run
        return run(args.analysis,args.inputs,args.output)
    if args.command == "demo":
        from .exact_theory import counterexample
        text = json.dumps(counterexample(), indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(text)
        print(text); return 0
    if args.command == "prepare-primary":
        from .prepare_primary import build
        print(json.dumps(build(args.long_table, args.burdens, args.data), indent=2)); return 0
    from .prepare import verify, prepare
    if args.command == "prepare":
        print(json.dumps(dict(files=prepare(args.source, args.data), data=str(args.data)))); return 0
    verify(args.data)
    if args.command == "check-data":
        print("Input checksums passed"); return 0
    data = args.data.resolve(); output = args.output.resolve()
    if output == data or data in output.parents:
        raise ValueError("Output must be outside the input-data directory")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "primary":
        from .primary import run
        print(json.dumps(run(data / "forecast", output), indent=2)); return 0
    commands = {
        "diagnostics": [str(ENGINE / "run_diagnostics.py"), "--inputs", str(data/"forecast"), "--output", str(output)],
        "selection": [str(ENGINE/"run_selection.py"), "--inputs", str(data/"selection/inputs"), "--output", str(output)],
        "verify-selection": [str(ENGINE/"verify_selection.py"), "--inputs", str(data/"selection/inputs"), "--results", str(args.results.resolve())] if args.command=="verify-selection" else [],
        "certificates": [str(ENGINE/"verify_theory_certificates.py"), "--inputs", str(data/"theory"), "--report", str(output/"verification.json")],
    }
    if args.command == "interval-bounds":
        work = output / "workspace"
        if work.exists():
            raise ValueError('Results already exist here; choose a new output directory')
        shutil.copytree(data/"interval_workspace", work)
        script = work/"lyme/reviews/2026-10-14/mathematical_strengthening_v1/run_strengthening.py"
        # Use the bound-calculation script included with the analysis code.
        script.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ENGINE/"run_strengthening_original.py", script)
        return subprocess.call([sys.executable, str(script)], cwd=work)
    return subprocess.call([sys.executable, *commands[args.command]])

def main():
    try:
        return _main()
    except (FileNotFoundError, ValueError) as error:
        print(f"Input error: {error}. See docs/DATA.md for required inputs.", file=sys.stderr)
        return 2

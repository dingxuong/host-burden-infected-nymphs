'Copy input data after checking file hashes.'
from pathlib import Path
import hashlib, json, shutil

def verify(data):
    manifest = json.loads((data / "manifest.json").read_text())
    for entry in manifest["files"]:
        rel = Path(entry["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Unsafe input path")
        path = data / rel
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Missing or changed input: {rel}")
    return len(manifest["files"])

def prepare(source, destination):
    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination must differ")
    count = verify(source)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Choose an empty destination directory')
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "manifest.json", destination / "manifest.json")
    manifest = json.loads((source / "manifest.json").read_text())
    for entry in manifest["files"]:
        rel = Path(entry["path"]); target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / rel, target)
    verify(destination)
    return count

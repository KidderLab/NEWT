import sys, subprocess, shutil
from importlib import resources
from pathlib import Path

def get_script_path(name: str) -> str | None:
    try:
        with resources.as_file(resources.files("newt.scripts") / name) as p:
            return str(p)
    except Exception:
        return None

def run_vendored(name: str, args: list[str]) -> int:
    script = get_script_path(name)
    if script is None or not Path(script).exists():
        raise FileNotFoundError(f"Vendored script not found: {name}")
    cmd = [sys.executable, "-u", script] + list(args)
    return subprocess.call(cmd)

def run_external(script_name: str, args: list[str]) -> int:
    p = Path.cwd() / script_name
    if p.exists():
        return subprocess.call([sys.executable, "-u", str(p)] + list(args))
    which = shutil.which(script_name)
    if which:
        return subprocess.call([which] + list(args))
    raise FileNotFoundError(f"Script not found in current directory or PATH: {script_name}")

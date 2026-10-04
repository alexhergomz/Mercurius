"""Install and test commands per repository, taken from SWE-smith's profiles.

SWE-smith (github.com/SWE-bench/SWE-smith, MIT) records, for ~847 repositories
across 11 languages, the shell commands that actually build and test each one:
`install_cmds`, `test_cmd`, a Python version, and a pinned commit. Someone has
already fought each of those repositories into building. Re-deriving them by
hand is how we spent an afternoon discovering that one project needs
`pip install -e .[dev]`.

Only the SHELL COMMANDS are taken, which is MIT-licensed code. Their problem
statements are written by claude-3-7-sonnet and their repository pool includes
GPL projects; neither is used here (docs/data_policy.md S10, decisions D7). The
repository list is filtered by our own licence policy first -- 616 of 849
survive -- and the tasks are mined from each repository's real history by us.

Their `test_cmd` activates a conda environment, which we do not use; the conda
prefix is stripped and the runner invocation kept.

    from mercurius.data.recipes import load_recipes
    r = load_recipes()["marshmallow-code/apispec"]      # -> install_cmds, test_cmd
"""
import json
import os
import re

from mercurius.paths import ROOT

CACHE = os.path.join(ROOT, "data/repos/recipes.json")
BASE = "https://raw.githubusercontent.com/SWE-bench/SWE-smith/main/swesmith/profiles"
LANGS = ("python", "golang", "typescript", "java", "rust", "cpp", "c",
         "javascript", "ruby", "php", "csharp")

# language defaults, mirroring their *Profile base classes
DEFAULTS = {
    "python": {"install_cmds": ["python -m pip install -e ."],
               "test_cmd": "pytest --disable-warnings --color=no --tb=no --verbose"},
    "golang": {"install_cmds": ["go mod tidy"], "test_cmd": "go test -v ./..."},
    "rust": {"install_cmds": ["cargo build"], "test_cmd": "cargo test"},
    "javascript": {"install_cmds": ["npm install"], "test_cmd": "npm test"},
    "typescript": {"install_cmds": ["npm install"], "test_cmd": "npm test"},
    "java": {"install_cmds": ["mvn -q -DskipTests install"], "test_cmd": "mvn test"},
    "ruby": {"install_cmds": ["bundle install"], "test_cmd": "bundle exec rake test"},
    "cpp": {"install_cmds": ["cmake -S . -B build", "cmake --build build -j"],
            "test_cmd": "ctest --test-dir build --output-on-failure"},
    "c": {"install_cmds": ["cmake -S . -B build", "cmake --build build -j"],
          "test_cmd": "ctest --test-dir build --output-on-failure"},
    "php": {"install_cmds": ["composer install"], "test_cmd": "vendor/bin/phpunit"},
    "csharp": {"install_cmds": ["dotnet restore"], "test_cmd": "dotnet test"},
}

_CLASS = re.compile(r"@dataclass\s*\nclass\s+(\w+)\(([\w]+)\):\s*\n((?:\s{4}.*\n|\s*\n)*)")
_FIELD = re.compile(r"^\s{4}(\w+)\s*:\s*[^=]+=\s*(.+?)\s*$", re.M)


def _literal(text):
    """field(default_factory=lambda: [...]) -> the list; plain "x" -> the string."""
    # greedy to the LAST bracket: a command like `pip install -e .[dev]` contains
    # a "]" of its own, and a lazy/negated-class match truncates mid-command
    m = re.search(r"default_factory\s*=\s*lambda:\s*(\[.*\])\s*\)", text, re.S)
    if m:
        try:
            return json.loads(m.group(1).replace("'", '"'))
        except json.JSONDecodeError:
            return re.findall(r"['\"]([^'\"]+)['\"]", m.group(1))
    m = re.match(r"^['\"](.*)['\"]$", text.strip())
    return m.group(1) if m else None


def strip_conda(cmd):
    """Their test_cmd activates conda; we run in a venv, so keep only the runner."""
    if not cmd:
        return cmd
    parts = [p.strip() for p in cmd.split(";")]
    keep = [p for p in parts
            if p and not p.startswith(("source ", "conda activate", "conda "))]
    return "; ".join(keep) or cmd


def fetch_recipes(langs=LANGS):
    import requests
    out = {}
    for lang in langs:
        try:
            text = requests.get(f"{BASE}/{lang}.py", timeout=40).text
        except Exception:
            continue
        for _name, _base, body in _CLASS.findall(text):
            f = dict(_FIELD.findall(body))
            owner, repo = _literal(f.get("owner", "")), _literal(f.get("repo", ""))
            if not owner or not repo:
                continue
            d = dict(DEFAULTS.get(lang, {}))
            d["lang"] = lang
            d["commit"] = _literal(f.get("commit", "")) or None
            if "install_cmds" in f:
                v = _literal(f["install_cmds"])
                if v:
                    d["install_cmds"] = v
            if "test_cmd" in f:
                v = _literal(f["test_cmd"])
                if v:
                    d["test_cmd"] = strip_conda(v)
            if "python_version" in f:
                d["python_version"] = _literal(f["python_version"])
            out[f"{owner}/{repo}"] = d
    return out


def load_recipes(refresh=False):
    if os.path.exists(CACHE) and not refresh:
        return json.load(open(CACHE))
    r = fetch_recipes()
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(r, open(CACHE, "w"), indent=0)
    return r


if __name__ == "__main__":
    r = load_recipes(refresh=True)
    from collections import Counter
    print(f"{len(r)} recipes; by language {dict(Counter(v['lang'] for v in r.values()).most_common())}")
    custom = sum(1 for v in r.values()
                 if v.get("install_cmds") != DEFAULTS.get(v["lang"], {}).get("install_cmds"))
    print(f"{custom} have a non-default install command")
    for k in list(r)[:4]:
        print(f"  {k}: {r[k].get('install_cmds')}")

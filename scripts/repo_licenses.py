"""Look up the license and basic metadata of every candidate repository.

Candidates are the repositories behind the open SWE task sets (SWE-rebench-V2,
R2E-Gym, SWE-smith, SWE-Gym): real projects, many with executable test
environments. The license is read from the GitHub API (the repo's detected
SPDX id), not from any dataset's own tag -- a dataset being CC-BY says nothing
about the code inside it.

Policy (docs/data_policy.md): code is admitted only under a permissive
license that allows commercial use without share-alike terms. Repos that
SWE-bench Verified draws from are excluded regardless, so that evaluation
stays clean.

    python scripts/repo_licenses.py            # writes data/repos/licenses.json
"""
import concurrent.futures as cf
import json
import os
import subprocess
import time

import requests

ROOT = os.path.join(os.path.dirname(__file__), "..")
LISTS = os.path.join(ROOT, "data/repos/task_repo_lists.json")
OUT = os.path.join(ROOT, "data/repos/licenses.json")

PERMISSIVE = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC",
              "Unlicense", "CC0-1.0", "0BSD", "Zlib", "PSF-2.0", "BSL-1.0",
              "MIT-0", "BSD-3-Clause-Clear", "UPL-1.0"}
SWE_BENCH_VERIFIED = {"astropy/astropy", "django/django", "matplotlib/matplotlib",
                      "mwaskom/seaborn", "pallets/flask", "psf/requests",
                      "pydata/xarray", "pylint-dev/pylint", "pytest-dev/pytest",
                      "scikit-learn/scikit-learn", "sphinx-doc/sphinx", "sympy/sympy"}
R2E_OWNERS = {"orange3": "biolab/orange3", "coveragepy": "nedbat/coveragepy",
              "numpy": "numpy/numpy", "datalad": "datalad/datalad",
              "pyramid": "Pylons/pyramid", "aiohttp": "aio-libs/aiohttp",
              "scrapy": "scrapy/scrapy", "tornado": "tornadoweb/tornado",
              "pillow": "python-pillow/Pillow", "pandas": "pandas-dev/pandas"}


def normalise(name):
    if name.startswith("swesmith/"):
        name = name.split("/", 1)[1].rsplit(".", 1)[0].replace("__", "/", 1)
    return R2E_OWNERS.get(name, name)


def main():
    lists = json.load(open(LISTS))
    tasks = {}
    for ds, v in lists.items():
        for r, n in v["repos"].items():
            k = normalise(r)
            tasks.setdefault(k, {"tasks": 0, "sources": set()})
            tasks[k]["tasks"] += n
            tasks[k]["sources"].add(ds)
    token = subprocess.run(["gh", "auth", "token"], capture_output=True,
                           text=True).stdout.strip()
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {token}",
                      "Accept": "application/vnd.github+json"})
    cache = json.load(open(OUT)) if os.path.exists(OUT) else {}

    def fetch(name):
        if name in cache and "error" not in cache[name]:
            return name, cache[name]
        for attempt in range(3):
            r = s.get(f"https://api.github.com/repos/{name}", timeout=30)
            if r.status_code == 200:
                j = r.json()
                lic = (j.get("license") or {}).get("spdx_id")
                return name, {"full_name": j["full_name"], "license": lic,
                              "stars": j["stargazers_count"], "archived": j["archived"],
                              "fork": j["fork"], "size_kb": j["size"],
                              "language": j.get("language"),
                              "default_branch": j["default_branch"]}
            if r.status_code in (403, 429):
                time.sleep(30 * (attempt + 1))
                continue
            return name, {"error": r.status_code}
        return name, {"error": "rate-limited"}

    with cf.ThreadPoolExecutor(8) as ex:
        for i, (name, meta) in enumerate(ex.map(fetch, sorted(tasks))):
            meta.update(tasks=tasks[name]["tasks"], sources=sorted(tasks[name]["sources"]))
            cache[name] = meta
            if (i + 1) % 500 == 0:
                json.dump(cache, open(OUT, "w"), indent=0)
                print(f"  {i + 1}/{len(tasks)}", flush=True)
    for name, m in cache.items():
        m["verified_eval_repo"] = (m.get("full_name", name).lower()
                                   in {x.lower() for x in SWE_BENCH_VERIFIED})
        m["admitted"] = (m.get("license") in PERMISSIVE and not m["verified_eval_repo"]
                         and "error" not in m)
    json.dump(cache, open(OUT, "w"), indent=0)
    from collections import Counter
    lic = Counter(m.get("license") or f"error:{m.get('error')}" for m in cache.values())
    adm = [m for m in cache.values() if m["admitted"]]
    print(f"{len(cache)} repos; licenses: {lic.most_common(14)}")
    print(f"admitted: {len(adm)} repos, {sum(m['tasks'] for m in adm)} tasks; "
          f"excluded as SWE-bench Verified: "
          f"{sum(m['verified_eval_repo'] for m in cache.values())}")


if __name__ == "__main__":
    main()

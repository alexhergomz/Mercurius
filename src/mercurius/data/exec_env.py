"""Build a repository at a commit and run its tests. The fail-to-pass gate.

This is the strongest verification available to us, and D1 wrongly ruled it out.
The blocker was never running tests -- it was that SWE-smith ships prebuilt
x86 Docker images and this machine is aarch64. Building the environment
OURSELVES works fine: pre-commit-hooks installs and runs 417 tests in 1.8 s on
ARM.

Scale is not the objection either. Only 11 of our clones are installable Python
with a test suite, which sounded disqualifying until you notice SWE-Gym's entire
dataset is 11 repositories, and that is where its +12-14pp came from. A small
execution-verified core plus a broad unverified corpus is the shape the field
converged on.

The gate, per SWE-bench: at the PARENT of a fix commit, apply only the TEST part
of the diff. Those tests must FAIL. Apply the source part too; they must PASS.
Tests that make that transition are the fail-to-pass set, and they are what makes
a generated patch checkable rather than merely plausible.

Isolation matters more than it looks. The system Python here carries ROS on
PYTHONPATH, which leaks into a fresh venv and breaks imports, and pytest walks
upwards to find a config file and picks up the mercurius pyproject.toml. Both
are stripped below; without that, results are silently about the wrong code.
"""
import os
import re
import shutil
import subprocess
import tempfile

from mercurius.paths import ROOT

GIT_DIR = os.path.join(ROOT, "data/repos/git")
VENVS = os.path.join(ROOT, "data/repos/venvs")
WORK = os.path.join(ROOT, "data/repos/work")
TEST_PATH = re.compile(r"(^|/)(tests?|spec)/|_test\.py$|(^|/)test_[^/]*\.py$", re.I)

# a clean environment: no ROS, no user site, no inherited config
CLEAN = {k: v for k, v in os.environ.items()
         if k not in ("PYTHONPATH", "PYTHONHOME", "ROS_DISTRO", "AMENT_PREFIX_PATH",
                      "CMAKE_PREFIX_PATH", "LD_LIBRARY_PATH", "PYTHONSTARTUP")}
CLEAN["PYTHONDONTWRITEBYTECODE"] = "1"
CLEAN["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"


def _run(args, cwd=None, timeout=900, env=None):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout, env=env or CLEAN)


def bare_of(repo):
    return os.path.join(GIT_DIR, repo.replace("/", "__") + ".git")


def workdir_for(repo):
    """One PERSISTENT working tree per repository.

    Not a temporary directory: the virtualenv installs the package EDITABLE
    against this path, so if the tree is deleted between commits the venv points
    at nothing and every later test run silently measures the wrong code. That
    bug produced a clean-looking 0/4 on the first attempt.
    """
    return os.path.join(WORK, repo.replace("/", "__"))


def checkout(repo, sha, dest):
    """Materialise the tree at `sha`. The bare clones are treeless, so file
    contents are fetched on demand -- a few seconds the first time, cached after."""
    bare = bare_of(repo)
    if not os.path.isdir(bare):
        return False, "no_history"
    if not os.path.isdir(os.path.join(dest, ".git")):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.rmtree(dest, ignore_errors=True)
        # Clone from GITHUB, not from our local bare mirror. The bare clones are
        # treeless promisors, and a clone OF a promisor cannot fetch the blobs it
        # is missing through it -- checkout dies with "unable to read sha1 file".
        # Cloning from the origin gives this tree its own promisor.
        r = _run(["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout",
                  f"https://github.com/{repo}.git", dest], timeout=1800)
        if r.returncode:
            return False, f"clone:{r.stderr.strip()[:90]}"
    _run(["git", "-C", dest, "checkout", "--quiet", "--force", "."])
    _run(["git", "-C", dest, "clean", "-qfd"])
    r = _run(["git", "-C", dest, "checkout", "--quiet", "--force", sha], timeout=900)
    if r.returncode:
        f = _run(["git", "-C", dest, "fetch", "--quiet", "upstream", sha], timeout=900)
        r = _run(["git", "-C", dest, "checkout", "--quiet", "--force", sha], timeout=900)
        if r.returncode:
            return False, f"checkout:{r.stderr.strip()[:90]}"
    if not os.listdir(dest):
        return False, "empty_tree"
    return True, "ok"


def venv_for(repo, src, rebuild=False, recipe=None):
    """One cached virtualenv per repository, with the package installed editable.

    Editable so that a later `git checkout` of a different commit changes the
    code under test without reinstalling -- which is what makes running hundreds
    of commits affordable.
    """
    path = os.path.join(VENVS, repo.replace("/", "__"))
    py = os.path.join(path, "bin", "python")
    if os.path.exists(py) and not rebuild:
        return py, "cached"
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    r = _run(["python3", "-m", "venv", path], timeout=300)
    if r.returncode:
        return None, f"venv:{r.stderr.strip()[:90]}"
    pip = os.path.join(path, "bin", "pip")
    # pip >= 25.1 for PEP 735 `--group`: modern projects declare test deps under
    # [dependency-groups], NOT as extras, and an older pip resolves `.[dev]` to
    # nothing and exits 0 -- installing nothing while reporting success. environs
    # scored 0/5 for exactly this: `tests` and `dev` were groups, `django` was
    # the only extra, and the test file could not import dj_database_url.
    _run([pip, "install", "-q", "-U", "pip"], timeout=600)
    r = _run([pip, "install", "-q", "pytest"], timeout=600)
    if r.returncode:
        return None, f"pytest:{r.stderr.strip()[:120]}"
    # SWE-smith recorded the command that actually works for this repository
    # (MIT, mercurius.data.recipes). Guessing extras by hand is how the first
    # attempt burned an afternoon.
    cmds = (recipe or {}).get("install_cmds") or ["python -m pip install -e ."]
    note = "ok"
    for cmd in cmds:
        args = cmd.replace("python -m pip", pip).replace("pip install", f"{pip} install")
        r = _run(["bash", "-lc", f"cd {src} && {args}"], timeout=1800)
        if r.returncode:
            for extra in ("[test]", "[tests]", "[dev]", ""):
                r2 = _run([pip, "install", "-q", "-e", src + extra], timeout=1800)
                if r2.returncode == 0:
                    note = f"fallback{extra or '(plain)'}"
                    break
            else:
                return None, f"install:{r.stderr.strip()[-160:]}"
    # TEST dependencies, always -- not only when the base install fails. A
    # successful `pip install -e .` routinely omits them, and the test file then
    # fails to IMPORT, which pytest reports as a collection error. environs died
    # exactly this way: `import dj_database_url` -> ModuleNotFoundError -> the
    # gate read it as the bug reproducing, and as still reproducing after the
    # fix. A whole repository scored zero for a missing test dependency.
    # PEP 735 dependency-groups first, then extras: a project may use either,
    # and several use groups for tests while keeping unrelated extras.
    for grp in ("tests", "test", "dev", "testing"):
        if _run([pip, "install", "-q", "--group", f"{os.path.join(src, 'pyproject.toml')}:{grp}"],
                timeout=1800).returncode == 0:
            note += f" +group:{grp}"
            break
    for extra in ("[dev]", "[test]", "[tests]", "[testing]"):
        if _run([pip, "install", "-q", "-e", src + extra], timeout=1800).returncode == 0:
            note += f" +{extra}"
            break
    # tox/setup.cfg testenv deps: where many projects actually put test-only
    # requirements, invisible to `pip install -e .[extra]`
    td = tox_deps(src)
    if td:
        if _run([pip, "install", "-q", *td], timeout=1800).returncode == 0:
            note += f" +tox({len(td)})"
        else:
            for d in td:                       # one bad pin should not sink the rest
                _run([pip, "install", "-q", d], timeout=600)
            note += f" +tox-partial({len(td)})"
    _run([pip, "install", "-q", *COMMON_TEST_DEPS], timeout=1800)
    note += " +common"
    for req in ("requirements-dev.txt", "dev-requirements.txt", "test-requirements.txt",
                "requirements-test.txt", "requirements/dev.txt", "requirements/test.txt"):
        rp = os.path.join(src, req)
        if os.path.exists(rp):
            if _run([pip, "install", "-q", "-r", rp], timeout=1800).returncode == 0:
                note += f" +{req}"
            break
    return py, note


# Test helpers that projects routinely use and routinely fail to declare in a
# place pip can see -- they live in tox.ini `deps`, in setup.py `tests_require`,
# or nowhere at all. Missing any one of them is a COLLECTION error, which takes
# the whole file down and cost us 27% of outcomes. Cheap to preinstall once per
# venv; `torch` is deliberately absent (2 GB, repo-specific).
COMMON_TEST_DEPS = [
    "pytest-asyncio", "pytest-mock", "pytest-timeout", "pytest-subtests",
    "hypothesis", "freezegun", "async-timeout", "mock", "responses",
    "requests-mock", "parameterized", "pytest-randomly", "coverage",
]


def tox_deps(src):
    """Dependencies declared in tox.ini / setup.cfg [testenv] deps."""
    out = []
    for name in ("tox.ini", "setup.cfg"):
        p = os.path.join(src, name)
        if not os.path.exists(p):
            continue
        try:
            text = open(p, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for block in re.findall(r"^deps\s*=\s*\n((?:[ \t]+\S.*\n)+)", text, re.M):
            for line in block.strip().split("\n"):
                d = line.strip()
                # tox FACTOR syntax: "dj42: Django>=4.2,<5.0" means that dep
                # applies to the dj42 environment only. pip cannot parse the
                # prefix, so strip it and take the dependency.
                d = re.sub(r"^[\w.,!-]+\s*:\s*", "", d)
                # environment markers pip does understand ("; platform_system==")
                # are kept; substitutions and requirement-file refs are not
                if d and not d.startswith(("-", "{", "#")) and "%" not in d:
                    out.append(d)
    return out[:40]


PYTEST_BASE = ["-q", "-p", "no:cacheprovider", "-c", "/dev/null", "--rootdir=."]


def run_tests(py, src, targets=None, timeout=900):
    """Returns (outcome_by_test, raw_tail). Outcome is 'passed'/'failed'/'error'."""
    args = [py, "-m", "pytest", *PYTEST_BASE, "-rA", "--no-header"]
    args += list(targets) if targets else ["."]
    try:
        r = _run(args, cwd=src, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {}, "TIMEOUT"
    out = {}
    for line in (r.stdout or "").split("\n"):
        m = re.match(r"^(PASSED|FAILED|ERROR)\s+(\S+)", line.strip())
        if m:
            out[m.group(2)] = m.group(1).lower()
    # A COLLECTION error is an environment fault, not a reproduced bug: the file
    # never ran. Reported separately so it cannot masquerade as a failing test.
    broken = bool(re.search(r"error during collection|errors during collection",
                            r.stdout or "", re.I))
    return out, ("COLLECTION_ERROR " if broken else "") + (r.stdout or "")[-1200:]


def test_files_in(diff_text):
    """The test files a commit touches -- the only ones worth running first.

    Running a repository's WHOLE suite to discover which tests fail is what
    SWE-bench avoids, and the reason is measurable here: click's suite takes
    ~3.5 min, is run twice per commit, and six commits cost ~42 min for one
    repository. The failing test introduced by a fix commit is, by
    construction, in a file that commit touched.
    """
    out = []
    for line in diff_text.split("\n"):
        m = re.match(r"diff --git a/(\S+) b/(\S+)", line)
        if m and TEST_PATH.search(m.group(2)):
            out.append(m.group(2))
    return out


_TESTDEF = re.compile(r"^\+\s*(?:async\s+)?def\s+(test\w*)\s*\(")
_CLASSDEF = re.compile(r"^[+ ]\s*class\s+(Test\w*)")


def added_tests(test_diff):
    """Test functions this commit ADDED or MODIFIED, by name.

    The fail-to-pass candidate set used to be every failing test in the touched
    files, which silently included tests that were already broken for unrelated
    reasons -- a repository with any pre-existing failure in that file polluted
    the set and the commit was scored `fix_did_not_pass`. 31% of all outcomes
    landed in that bucket. The tests a fix commit writes are, by construction,
    the ones that capture the bug, so those are the candidates.
    """
    names = set()
    for line in test_diff.split("\n"):
        m = _TESTDEF.match(line)
        if m:
            names.add(m.group(1))
    return names


def split_diff(diff_text):
    """Separate a commit's diff into its test and source halves."""
    tests, src, cur, is_test = [], [], None, False
    for line in diff_text.split("\n"):
        if line.startswith("diff --git "):
            if cur is not None:
                (tests if is_test else src).append("\n".join(cur))
            m = re.match(r"diff --git a/(\S+) b/(\S+)", line)
            path = m.group(2) if m else ""
            is_test = bool(TEST_PATH.search(path))
            cur = [line]
        elif cur is not None:
            cur.append(line)
    if cur is not None:
        (tests if is_test else src).append("\n".join(cur))
    return "\n".join(tests), "\n".join(src)


def apply_patch(src, patch_text):
    if not patch_text.strip():
        return True
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as fh:
        fh.write(patch_text.rstrip("\n") + "\n")
        p = fh.name
    try:
        r = _run(["git", "-C", src, "apply", "--whitespace=nowarn", p])
        return r.returncode == 0
    finally:
        os.unlink(p)


def fail_to_pass(repo, sha, diff_text, workdir=None, py=None, recipe=None):
    """SWE-bench's gate, run locally.

    parent + test-diff  -> those tests must FAIL
    parent + whole diff -> they must PASS

    Returns (f2p_tests, detail). An empty f2p set means the commit is not usable
    as a verified task: either the tests did not actually exercise the bug, or
    the environment could not reproduce it.
    """
    test_diff, src_diff = split_diff(diff_text)
    if not test_diff.strip():
        return [], "no_test_diff"
    parent = f"{sha}^"
    workdir = workdir or workdir_for(repo)
    ok, why = checkout(repo, parent, workdir)
    if not ok:
        return [], why
    if py is None:
        py, why = venv_for(repo, workdir, recipe=recipe)
        if py is None:
            return [], why
    if not apply_patch(workdir, test_diff):
        return [], "test_patch_failed"
    # only the test files this commit touched, not the whole suite
    targets = [t for t in test_files_in(test_diff)
               if os.path.exists(os.path.join(workdir, t))]
    before, tail = run_tests(py, workdir, targets=targets or None)
    if tail.startswith("COLLECTION_ERROR"):
        return [], "env_broken_collection"
    failing = [t for t, o in before.items() if o in ("failed", "error")]
    if not failing:
        return [], "tests_did_not_fail"
    # prefer the tests THIS COMMIT wrote; fall back to all failing tests only if
    # none of them are among the failures (e.g. a parametrised or renamed case)
    wrote = added_tests(test_diff)
    if wrote:
        owned = [t for t in failing
                 if any(re.search(rf"(::|\b){re.escape(n)}(\b|\[)", t) for n in wrote)]
        if owned:
            failing = owned
    if not apply_patch(workdir, src_diff):
        return [], "src_patch_failed"
    after, tail = run_tests(py, workdir, targets=failing)
    f2p = [t for t in failing if after.get(t) == "passed"]
    if not f2p:
        return [], "fix_did_not_pass"
    return f2p, "ok"

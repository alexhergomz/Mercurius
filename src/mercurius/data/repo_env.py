"""A read-only tool environment over one repository checkout.

The agent sees a repository only through tools, exactly as a deployed coding
agent would: it lists directories, reads files by line range, greps, and
searches -- search_code is retrieval over the repository's own functions and
classes (BM25 over code chunks), i.e. the RAG a real agent gets. The schemas
in TOOLS are what goes into the chat template's `tools` argument, so training
sequences carry the same tool definitions the deployed model will see.

Every string a tool returns passes through scrub(): secrets and personal data
are replaced before they can reach a training sequence (docs/data_policy.md).
Paths are confined to the checkout; binaries, vendored trees, lockfiles and
minified bundles are invisible, as they would be noise to an agent too.
"""
import math
import os
import re
from collections import Counter

MAX_READ_LINES = 400
MAX_OUT_CHARS = 20_000
SKIP_DIRS = {".git", "node_modules", "vendor", "third_party", "dist", "build",
             "__pycache__", ".venv", "venv", "target", ".tox", ".mypy_cache"}
SKIP_FILES = re.compile(r"(\.lock|lock\.json|\.min\.(js|css)|\.map|\.svg|\.png|\.jpe?g|"
                        r"\.gif|\.ico|\.pdf|\.zip|\.gz|\.tar|\.whl|\.so|\.dylib|\.dll|"
                        r"\.exe|\.bin|\.woff2?|\.ttf|\.eot|\.mp[34]|\.wasm|\.pyc)$", re.I)

# ---------------------------------------------------------------- scrubbing
_SECRET_PATTERNS = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
     "<REDACTED_PRIVATE_KEY>"),
    (re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"), "<REDACTED_AWS_KEY>"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"), "<REDACTED_GITHUB_TOKEN>"),
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"), "<REDACTED_SLACK_TOKEN>"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), "<REDACTED_API_KEY>"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), "<REDACTED_API_KEY>"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
     "<REDACTED_JWT>"),
    (re.compile(r"(?i)\b((?:api|secret|access|auth|private)[_-]?(?:key|token|secret)|password|passwd)"
                r"(\s*[:=]\s*)(['\"])[^'\"\n]{8,}\3"), r"\1\2\3<REDACTED>\3"),
]
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_EMAIL_OK = re.compile(r"@(example\.(com|org|net)|localhost|test\b|users\.noreply\.github\.com)", re.I)


def scrub(text):
    """Replace secrets and e-mail addresses. Example domains are kept (they are
    placeholders, not people). Author NAMES are left: license headers must
    stay intact, and attribution is required by several admitted licenses."""
    for pat, rep in _SECRET_PATTERNS:
        text = pat.sub(rep, text)
    return _EMAIL.sub(lambda m: m.group(0) if _EMAIL_OK.search(m.group(0))
                      else "<REDACTED_EMAIL>", text)


# ------------------------------------------------------------------- tools
TOOLS = [
    {"type": "function", "function": {
        "name": "list_dir",
        "description": "List the entries of a directory in the repository.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Directory path relative to the repository root. Use '.' for the root."}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": f"Read a file, optionally a line range (1-based, inclusive). At most {MAX_READ_LINES} lines are returned per call.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "grep",
        "description": "Search file contents with a regular expression. Returns matching lines as path:line: text.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Python regular expression."},
            "path": {"type": "string", "description": "Directory or file to search, default '.'."},
            "max_results": {"type": "integer"}},
            "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "search_code",
        "description": "Semantic-style search over the repository's functions, classes and modules. Returns the best-matching code chunks with their locations.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What you are looking for, in words or identifiers."},
            "k": {"type": "integer", "description": "Number of chunks to return, default 5."}},
            "required": ["query"]}}},
]

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_DEF = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?(?:def|class|fn|func|function|interface|struct|"
                  r"enum|trait|impl|type|module|public|private|protected|static)\b")


def _terms(text):
    out = []
    for t in _TOKEN.findall(text):
        out.append(t.lower())
        parts = re.findall(r"[a-z]+|[A-Z][a-z]*|\d+", t)       # split camelCase / snake_case
        if len(parts) > 1:
            out.extend(p.lower() for p in parts)
    return out


class RepoEnv:
    def __init__(self, root):
        self.root = os.path.realpath(root)
        self.files = []
        for d, dirs, fs in os.walk(self.root):
            dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS and not x.startswith("."))
            for f in sorted(fs):
                p = os.path.join(d, f)
                if SKIP_FILES.search(f) or os.path.getsize(p) > 1_000_000:
                    continue
                self.files.append(os.path.relpath(p, self.root))
        self._index = None

    # -- helpers
    def _resolve(self, path):
        p = os.path.realpath(os.path.join(self.root, path or "."))
        if not (p == self.root or p.startswith(self.root + os.sep)):
            raise ValueError("path outside the repository")
        return p

    def _text(self, rel):
        try:
            with open(os.path.join(self.root, rel), encoding="utf-8") as fh:
                return fh.read()
        except (UnicodeDecodeError, OSError):
            return None

    @staticmethod
    def _cap(s):
        return s if len(s) <= MAX_OUT_CHARS else s[:MAX_OUT_CHARS] + "\n... [output truncated]"

    # -- tools
    def list_dir(self, path="."):
        p = self._resolve(path)
        if not os.path.isdir(p):
            return f"Error: {path} is not a directory"
        rel = os.path.relpath(p, self.root)
        kids = set()
        for f in self.files:
            if rel == "." or f.startswith(rel + os.sep):
                rest = f if rel == "." else f[len(rel) + 1:]
                head = rest.split(os.sep, 1)
                kids.add(head[0] + ("/" if len(head) > 1 else ""))
        return self._cap("\n".join(sorted(kids, key=lambda k: (not k.endswith("/"), k))) or "(empty)")

    def read_file(self, path, start_line=None, end_line=None):
        p = self._resolve(path)
        rel = os.path.relpath(p, self.root)
        if rel not in self.files:
            return f"Error: {path} not found or not a text file"
        txt = self._text(rel)
        if txt is None:
            return f"Error: {path} is not UTF-8 text"
        lines = txt.splitlines()
        s = max(1, int(start_line or 1))
        e = min(len(lines), int(end_line or s + MAX_READ_LINES - 1), s + MAX_READ_LINES - 1)
        body = "\n".join(f"{i:>5}\t{lines[i - 1]}" for i in range(s, e + 1))
        more = f"\n... ({len(lines) - e} more lines)" if e < len(lines) else ""
        return self._cap(scrub(body) + more)

    def grep(self, pattern, path=".", max_results=50):
        try:
            rx = re.compile(pattern)
        except re.error as err:
            return f"Error: invalid regular expression: {err}"
        base = os.path.relpath(self._resolve(path), self.root)
        hits = []
        for f in self.files:
            if base != "." and not (f == base or f.startswith(base + os.sep)):
                continue
            txt = self._text(f)
            if txt is None:
                continue
            for i, line in enumerate(txt.splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{f}:{i}: {line.strip()[:200]}")
                    if len(hits) >= int(max_results or 50):
                        return self._cap(scrub("\n".join(hits)) + "\n... (more results)")
        return self._cap(scrub("\n".join(hits)) or "No matches.")

    def _build_index(self, chunk_lines=60):
        chunks = []
        for f in self.files:
            txt = self._text(f)
            if txt is None:
                continue
            lines = txt.splitlines()
            starts = [i for i, l in enumerate(lines) if _DEF.match(l)] or [0]
            if starts[0] != 0:
                starts = [0] + starts
            for a, b in zip(starts, starts[1:] + [len(lines)]):
                for s in range(a, b, chunk_lines):           # split long definitions
                    e = min(b, s + chunk_lines)
                    body = "\n".join(lines[s:e])
                    if body.strip():
                        chunks.append((f, s + 1, e, body, Counter(_terms(f + " " + body))))
        df = Counter()
        for c in chunks:
            df.update(c[4].keys())
        n = max(len(chunks), 1)
        avg = sum(sum(c[4].values()) for c in chunks) / n
        self._index = (chunks, {t: math.log(1 + (n - v + 0.5) / (v + 0.5)) for t, v in df.items()}, avg)

    def search_code(self, query, k=5):
        if self._index is None:
            self._build_index()
        chunks, idf, avg = self._index
        q = _terms(query)
        scored = []
        for c in chunks:
            tf, dl = c[4], sum(c[4].values())
            s = sum(idf.get(t, 0) * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * dl / avg))
                    for t in q if t in tf)
            if s > 0:
                scored.append((s, c))
        scored.sort(key=lambda x: -x[0])
        out = [f"--- {c[0]} (lines {c[1]}-{c[2]})\n{c[3]}" for _, c in scored[:int(k or 5)]]
        return self._cap(scrub("\n\n".join(out)) or "No results.")

    def call(self, name, args):
        fn = {"list_dir": self.list_dir, "read_file": self.read_file,
              "grep": self.grep, "search_code": self.search_code}.get(name)
        if fn is None:
            return f"Error: unknown tool {name}"
        try:
            return fn(**(args or {}))
        except (TypeError, ValueError) as err:
            return f"Error: {err}"

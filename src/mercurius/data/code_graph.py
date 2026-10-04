"""A symbol and call graph over one repository, and verifiable tasks built from it.

The point is ground truth that no model produced. Every task here is generated
by PARSING the repository, so the answer is a fact about the code, and a
trajectory can be accepted or rejected mechanically instead of being trusted
(docs/data_policy.md). That is what makes rejection sampling possible: we
generate k rollouts, keep the ones that land on the parsed answer, and the
acceptance rate is a measurement rather than a guess.

Tasks are multi-hop retrieval over a large codebase -- find where a symbol is
defined, find everything that calls it, work out what a signature change would
break. That is deliberately the capability that latent-KV compression and NoPE
put at risk, so the data we can verify is also the data we most need.

tree-sitter rather than a per-language toolchain: our clones are 9 languages
and only 11 of 144 repositories are installable Python with tests, so anything
that needs to BUILD a repository would collapse the corpus back to a
monoculture. Parsing needs nothing installed.
"""
import os
import random
import re
import subprocess
from collections import Counter, defaultdict

from tree_sitter_language_pack import get_parser

LANGS = {".py": "python", ".js": "javascript", ".jsx": "javascript",
         ".ts": "typescript", ".tsx": "tsx", ".go": "go", ".java": "java",
         ".rs": "rust", ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp",
         ".hpp": "cpp", ".rb": "ruby", ".php": "php", ".cs": "csharp",
         ".lua": "lua", ".swift": "swift", ".ex": "elixir", ".exs": "elixir",
         ".dart": "dart", ".clj": "clojure", ".kt": "kotlin", ".scala": "scala"}

# node types that DEFINE a named symbol, per language family
DEF_TYPES = {
    "python": {"function_definition": "function", "class_definition": "class"},
    "javascript": {"function_declaration": "function", "class_declaration": "class",
                   "method_definition": "method", "generator_function_declaration": "function"},
    "typescript": {"function_declaration": "function", "class_declaration": "class",
                   "method_definition": "method", "interface_declaration": "interface",
                   "type_alias_declaration": "type", "enum_declaration": "enum"},
    "tsx": {"function_declaration": "function", "class_declaration": "class",
            "method_definition": "method", "interface_declaration": "interface"},
    "go": {"function_declaration": "function", "method_declaration": "method",
           "type_declaration": "type"},
    "java": {"class_declaration": "class", "method_declaration": "method",
             "interface_declaration": "interface", "enum_declaration": "enum"},
    "rust": {"function_item": "function", "struct_item": "struct", "enum_item": "enum",
             "trait_item": "trait", "type_item": "type"},
    "c": {"function_definition": "function", "struct_specifier": "struct"},
    "cpp": {"function_definition": "function", "class_specifier": "class",
            "struct_specifier": "struct"},
    "ruby": {"method": "method", "class": "class", "module": "module"},
    "php": {"function_definition": "function", "class_declaration": "class",
            "method_declaration": "method"},
    "csharp": {"class_declaration": "class", "method_declaration": "method",
               "interface_declaration": "interface"},
    "lua": {"function_declaration": "function"},
    "swift": {"function_declaration": "function", "class_declaration": "class",
              "protocol_declaration": "protocol"},
    "elixir": {"call": "function"},
    "dart": {"function_signature": "function", "class_definition": "class",
             "method_signature": "method"},
    "clojure": {"list_lit": "form"},
    "kotlin": {"function_declaration": "function", "class_declaration": "class",
               "object_declaration": "object"},
    "scala": {"function_definition": "function", "class_definition": "class",
              "object_definition": "object", "trait_definition": "trait"},
}

CALL_TYPES = {"call", "call_expression", "method_invocation", "function_call_expression",
              "invocation_expression", "macro_invocation"}

# Nodes that DECLARE a symbol without defining it -- a C/C++ header prototype, a
# TypeScript ambient declaration. These are NOT definitions (so they do not make
# a symbol ambiguous) but they ARE part of the answer to "what would a signature
# change force you to edit": the audit of the first pilot found MAVSDK scoring
# 0/3 purely because `impact` omitted the header that declares the function, so
# the teacher's correct answers were being rejected.
DECL_TYPES = {
    "c": {"declaration", "field_declaration"},
    "cpp": {"declaration", "field_declaration"},
}

# identifiers too common to make a task out of: the answer would be ambiguous
STOPWORDS = {"main", "init", "new", "get", "set", "run", "test", "setup", "build",
             "to_string", "toString", "String", "len", "size", "name", "value",
             "add", "remove", "update", "close", "open", "read", "write", "start",
             "stop", "next", "clone", "equals", "hashCode", "print", "log", "error",
             "self", "this", "super", "format", "parse", "load", "save", "call",
             "append", "count", "index", "keys", "items", "values", "copy", "type"}

SKIP_DIRS = {".git", "node_modules", "vendor", "third_party", "dist", "build",
             "__pycache__", ".venv", "venv", "target", ".tox", ".mypy_cache",
             "testdata", "fixtures", "examples", "docs", "generated", "gen"}

# Machine-generated sources: protobuf/gRPC stubs, ORM and binding output. They
# are real code, but a task over them teaches pattern-matching on a code
# generator's template rather than reading a codebase, and they inflate the
# answer sets with near-identical files.
# Minified or bundled output: RepoEnv already hides these from the agent, and
# the graph must hide them too or the ground truth counts call sites in a file
# the agent can never see.
BUNDLED = re.compile(r"(\.min\.(js|css)$|[./-](vendor|bundle|polyfills|runtime)"
                     r"(\.min)?\.js$|\.bundle\.js$)", re.I)

GENERATED = re.compile(r"(\.pb\.(cc|h|go|py)$|_pb2(_grpc)?\.py$|\.grpc\.pb\.|"
                       r"\.g\.dart$|\.freezed\.dart$|_generated\.|\.generated\.|"
                       r"\.designer\.cs$|_pb\.js$|\.pb\.swift$)", re.I)


def _name_of(node, src):
    """The declared name of a definition node, across grammars."""
    n = node.child_by_field_name("name")
    if n is not None:
        return src[n.start_byte:n.end_byte].decode("utf8", "replace")
    d = node.child_by_field_name("declarator")          # c / cpp
    while d is not None:
        if d.type in ("identifier", "field_identifier", "type_identifier"):
            return src[d.start_byte:d.end_byte].decode("utf8", "replace")
        nxt = d.child_by_field_name("declarator")
        if nxt is None:
            for ch in d.children:
                if ch.type in ("identifier", "field_identifier"):
                    return src[ch.start_byte:ch.end_byte].decode("utf8", "replace")
            return None
        d = nxt
    for ch in node.children:                             # go type_declaration etc.
        if ch.type in ("type_spec", "type_identifier", "identifier"):
            if ch.type == "type_spec":
                nn = ch.child_by_field_name("name")
                if nn is not None:
                    return src[nn.start_byte:nn.end_byte].decode("utf8", "replace")
            else:
                return src[ch.start_byte:ch.end_byte].decode("utf8", "replace")
    return None


def _callee(node, src):
    """The bare name being called: a.b.c(...) -> c."""
    f = (node.child_by_field_name("function") or node.child_by_field_name("name")
         or node.child_by_field_name("macro"))
    if f is None:
        return None
    txt = src[f.start_byte:f.end_byte].decode("utf8", "replace")
    txt = txt.split("(")[0].strip()
    if not txt or len(txt) > 120:
        return None
    return re.split(r"[.:\->]+", txt)[-1].strip()


class CodeGraph:
    """Definitions and call sites for one checkout.

    Definitions are keyed by bare name, so a name defined once in the whole
    repository is a task with an unambiguous answer, and one defined many times
    is discarded rather than guessed at.
    """

    def __init__(self, root, max_files=4000, max_bytes=400_000):
        self.root = os.path.realpath(root)
        self.defs = defaultdict(list)      # name -> [(path, line, kind, lang)]
        self.decls = defaultdict(set)      # name -> {path}  (header prototypes)
        self.calls = defaultdict(list)     # name -> [(path, line)]
        self.mentions = Counter()          # name -> every identifier occurrence
        self.sigs = {}                     # (path, line) -> signature text
        self._parsers, n = {}, 0
        for d, dirs, fs in os.walk(self.root):
            dirs[:] = [x for x in dirs if x not in SKIP_DIRS and not x.startswith(".")]
            for f in sorted(fs):
                lang = LANGS.get(os.path.splitext(f)[1])
                if (lang is None or lang not in DEF_TYPES or GENERATED.search(f)
                        or BUNDLED.search(f)):
                    continue
                p = os.path.join(d, f)
                try:
                    if os.path.getsize(p) > max_bytes:
                        continue
                    src = open(p, "rb").read()
                except OSError:
                    continue
                self._scan(os.path.relpath(p, self.root), src, lang)
                n += 1
                if n >= max_files:
                    return

    def _scan(self, rel, src, lang):
        try:
            parser = self._parsers.get(lang) or self._parsers.setdefault(lang, get_parser(lang))
            tree = parser.parse(src)
        except Exception:
            return
        defs, decls = DEF_TYPES[lang], DECL_TYPES.get(lang, ())
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            if node.type in decls:
                # a prototype: a declarator, but no body
                if node.child_by_field_name("body") is None:
                    nm = _name_of(node, src)
                    if nm and nm.isidentifier():
                        self.decls[nm].add(rel)
            if node.type in defs:
                name = _name_of(node, src)
                if name and name.isidentifier():
                    line = node.start_point[0] + 1
                    self.defs[name].append((rel, line, defs[node.type], lang))
                    head = src[node.start_byte:node.end_byte].split(b"\n")[0]
                    self.sigs[(rel, line)] = head.decode("utf8", "replace").strip()[:200]
            elif node.type in CALL_TYPES:
                name = _callee(node, src)
                if name and name.isidentifier():
                    self.calls[name].append((rel, node.start_point[0] + 1))
            elif node.type in ("identifier", "field_identifier", "type_identifier",
                               "property_identifier", "name"):
                txt = src[node.start_byte:node.end_byte].decode("utf8", "replace")
                if txt.isidentifier():
                    self.mentions[txt] += 1
            stack.extend(node.children)

    # ---------------------------------------------------------------- tasks
    def unique_defs(self, min_calls=2, max_calls=14, min_files=2, min_coverage=0.75):
        """Symbols defined EXACTLY once, called from several files: the only
        shape where 'where is it defined' and 'what calls it' both have one
        right answer.

        min_coverage guards AMBIGUITY, which the first ground-truth audit found
        to be the dominant problem -- not missing parses. A name like `getName`,
        `send` or `resolve` may be defined once here and still be a method on a
        dozen unrelated types, so its call-site set is not well defined and the
        task has no single right answer. We detect that WITHOUT a second tool:
        if our structured account (definition + declarations + call sites) does
        not explain most textual occurrences of the identifier, the remainder is
        something we are not modelling, and the symbol is dropped rather than
        turned into a task with an answer we cannot defend.
        """
        out = []
        for name, ds in self.defs.items():
            if len(ds) != 1 or name in STOPWORDS or len(name) < 4:
                continue
            seen = self.mentions.get(name, 0)
            explained = len(self.calls.get(name, [])) + len(ds) + len(self.decls.get(name, ()))
            if seen and explained / seen < min_coverage:
                continue
            sites = [c for c in self.calls.get(name, []) if c != (ds[0][0], ds[0][1])]
            files = {p for p, _ in sites}
            if min_calls <= len(sites) <= max_calls and len(files) >= min_files:
                out.append((name, ds[0], sorted(set(sites))))
        return out


EXT_OF = defaultdict(list)
for _e, _l in LANGS.items():
    EXT_OF[_l].append(_e)


def grep_contradicts(root, symbol, lang, ours):
    """True if grep sees the symbol in files our answer omits.

    A second, completely independent opinion on the ground truth. grep is a
    SUPERSET -- it matches comments, strings and unrelated same-named symbols --
    so agreement is strong evidence and disagreement is a reason to drop the
    task, not to widen the answer. The first pilot lost a repository to an
    answer set we could not defend (missing C headers); this makes that class of
    error cost a discarded task instead of a corrupted trajectory.
    """
    args = ["grep", "-rIl", "--binary-files=without-match"]
    args += [f"--include=*{e}" for e in EXT_OF.get(lang, [])]
    args += [rf"\b{re.escape(symbol)}\b", root]
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    theirs = {os.path.relpath(p, root) for p in r.stdout.split() if p}
    theirs = {p for p in theirs if not (GENERATED.search(p) or BUNDLED.search(p))
              and not any(d in p.split(os.sep) for d in SKIP_DIRS)}
    return bool(theirs - set(ours))


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)


def parse_answer(text):
    m = ANSWER_RE.findall(text or "")
    return m[-1].strip() if m else None


def _norm_path(p):
    return (p or "").strip().strip("`'\"").lstrip("./").replace("\\", "/")


def make_tasks(graph, repo, n=8, rng=None, strict=True):
    """Verifiable tasks, each carrying its own parsed answer.

    strict: cross-check every answer against grep and drop the task if grep
    sees the symbol somewhere the answer does not (see grep_contradicts).
    """
    rng = rng or random.Random(0)
    cands = graph.unique_defs()
    rng.shuffle(cands)
    tasks = []
    for name, (path, line, kind, lang), _sites in cands:
        sites = [c for c in sorted(set(graph.calls.get(name, [])))
                 if c != (path, line)]
        files = sorted({p for p, _ in sites})
        kinds = rng.choice(["locate", "callers", "impact"])
        if kinds == "locate":
            q = (f"In this repository, where is the {kind} `{name}` defined? "
                 f"Answer with the file path and the line its definition starts on, "
                 f"as <answer>{{\"file\": \"path/to/file\", \"line\": 123}}</answer>.")
            ans = {"file": path, "line": line}
        elif kinds == "callers":
            q = (f"Find every place in this repository that CALLS `{name}` "
                 f"(not its definition). Answer with the list of files containing "
                 f"at least one call, as "
                 f"<answer>{{\"files\": [\"a/b.py\", \"c/d.py\"]}}</answer>.")
            ans = {"files": files}
        else:
            q = (f"Suppose the signature of the {kind} `{name}` changes so that all "
                 f"of its call sites must be updated. List every file that would "
                 f"need editing, including the file holding the definition itself, "
                 f"as <answer>{{\"files\": [\"a/b.py\"]}}</answer>.")
            ans = {"files": sorted(set(files) | {path} | graph.decls.get(name, set()))}
        checked = sorted(set(ans.get("files", [])) | ({ans["file"]} if "file" in ans else set()))
        if strict and grep_contradicts(graph.root, name, lang, checked):
            continue
        tasks.append({"repo": repo, "kind": kinds, "symbol": name, "lang": lang,
                      "question": q, "answer": ans,
                      "n_sites": len(sites), "n_files": len(files)})
        if len(tasks) >= n:
            break
    return tasks


def verify(task, final_text):
    """Exact for a location, set-equality for a file list. Returns (ok, detail)."""
    raw = parse_answer(final_text)
    if raw is None:
        return False, "no_answer_tag"
    import ast
    import json
    try:
        got = json.loads(raw)
    except json.JSONDecodeError:
        # Models routinely emit a PYTHON dict -- {'files': ['a.py']} -- which is
        # not JSON. Rejecting it scored correct answers as bad_json (measured:
        # 0/77 gold answers accepted when single-quoted). literal_eval parses
        # literals only, so there is nothing to execute.
        try:
            got = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return False, "bad_json"
    if not isinstance(got, dict):
        return False, "bad_shape"
    want = task["answer"]
    if task["kind"] == "locate":
        if _norm_path(got.get("file")) != _norm_path(want["file"]):
            return False, "wrong_file"
        try:
            # the definition line, or the line of its first decorator/comment
            if abs(int(got.get("line", -1)) - want["line"]) > 3:
                return False, "wrong_line"
        except (TypeError, ValueError):
            return False, "bad_line"
        return True, "ok"
    g = {_norm_path(x) for x in (got.get("files") or []) if isinstance(x, str)}
    w = {_norm_path(x) for x in want["files"]}
    if not g:
        return False, "empty_list"
    if g == w:
        return True, "ok"
    return False, f"set_mismatch:missing={len(w - g)},extra={len(g - w)}"

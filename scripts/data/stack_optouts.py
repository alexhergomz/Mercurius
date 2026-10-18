"""Build an exclusion list from The Stack's opt-out requests.

Source: the issues of github.com/bigcode-project/opt-out-v2, where people ask
for their accounts, organisations or repositories to be removed from The Stack.
We are not The Stack, but the request expresses the author's wish about code
being used to train models, and honouring it is part of the data policy.

Read conservatively:
  * every account / organisation listed under "remove ENTIRELY" is excluded;
  * every repository linked under "Specific repositories" is excluded;
  * the REQUESTER's own account is excluded too, whatever they listed -- some
    write "all" or "me", some list only a few repos. Over-exclusion costs a
    handful of repositories; under-exclusion ignores an explicit request.

    python scripts/stack_optouts.py     # -> data/repos/optouts.json
"""
import json
import os
import re
import subprocess

OUT = os.path.join(os.path.dirname(__file__), "..", "data/repos/optouts.json")
REPO_RE = re.compile(r"github\.com/([A-Za-z0-9-]+)/([A-Za-z0-9_.-]+)")
NAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")
SKIP = {"all", "me", "none", "no", "response", "i", "request", "the", "following",
        "data", "is", "removed", "from", "stack", "and", "my", "account", "accounts"}


def section(body, title):
    m = re.search(r"###\s*" + re.escape(title) + r"[^\n]*\n(.*?)(?=\n###|\Z)", body, re.S | re.I)
    return m.group(1) if m else ""


def main():
    out = subprocess.run(
        ["gh", "api", "--paginate", "repos/bigcode-project/opt-out-v2/issues?state=all&per_page=100",
         "--jq", ".[] | {n: .number, user: .user.login, body: (.body // \"\"), pr: (.pull_request != null)}"],
        capture_output=True, text=True, check=True).stdout
    accounts, repos, n = set(), set(), 0
    for line in out.splitlines():
        it = json.loads(line)
        if it["pr"]:
            continue
        n += 1
        body = it["body"]
        accounts.add(it["user"].lower())
        for tok in re.split(r"[\s,;]+", re.sub(r"https?://\S+", " ", section(body, "Accounts and organizations"))):
            tok = tok.strip("-*_`[]()@").lower()
            if tok and tok not in SKIP and NAME_RE.match(tok):
                accounts.add(tok)
        for m in REPO_RE.finditer(body):
            owner, name = m.group(1).lower(), m.group(2).lower().removesuffix(".git")
            if owner != "bigcode-project":
                repos.add(f"{owner}/{name}")
    json.dump({"issues": n, "accounts": sorted(accounts), "repos": sorted(repos)},
              open(OUT, "w"), indent=0)
    print(f"{n} opt-out issues -> {len(accounts)} accounts/orgs, {len(repos)} repositories")


if __name__ == "__main__":
    main()

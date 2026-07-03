#!/usr/bin/env python3
"""Post-hoc WER from decode_sglang.py results.jsonl (English-normalized)."""
import json, re, string, sys

_tbl = str.maketrans("", "", string.punctuation)
def norm(s):
    return re.sub(r"\s+", " ", s.upper().translate(_tbl)).strip()

def edit(a, b):
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j-1], dp[j])
            prev = cur
    return dp[m]

for path in sys.argv[1:]:
    tot_err = tot_words = n = exact = 0
    for line in open(path):
        r = json.loads(line)
        if str(r["hyp"]).startswith("ERROR"):
            continue
        ref = norm(r["gt"]).split()      # gt stored normalized already; norm() is idempotent
        hyp = norm(r["hyp"]).split()
        tot_err += edit(ref, hyp); tot_words += len(ref); n += 1
        exact += (ref == hyp)
    print(f"{path}: WER={tot_err/max(tot_words,1):.2%}  ({tot_err}/{tot_words} words, "
          f"n={n}, exact={exact} {exact/max(n,1):.0%})")

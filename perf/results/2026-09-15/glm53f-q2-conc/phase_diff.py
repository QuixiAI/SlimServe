"""Diff two cumulative VLLM_QC_PHASE_PROF dumps (A taken after run 1, B after
run 2) and print, per phase, run 1's and run 2's totals and ms/call side by
side. Usage: phase_diff.py <after_run1.txt> <after_run2.txt> [steps1 steps2]"""

import sys


def load(path):
    rows = {}
    for line in open(path):
        parts = line.split()
        if len(parts) == 5 and parts[0][0].isdigit():
            sec, cnt, _, _, name = parts
            rows[name] = (float(sec), int(cnt))
    return rows


a = load(sys.argv[1])
b = load(sys.argv[2])
run1 = a
run2 = {k: (b[k][0] - a.get(k, (0, 0))[0], b[k][1] - a.get(k, (0, 0))[1]) for k in b}
steps1 = run1["execute_model"][1]
steps2 = run2["execute_model"][1]
print(f"steps: run1={steps1} run2={steps2}")
print(f"{'phase':<18}{'r1 ms/step':>11}{'r2 ms/step':>11}{'ratio':>7}   {'r1 calls/step':>13}{'r2 calls/step':>13}")
order = sorted(run2, key=lambda k: -run2[k][0])
for k in order:
    s1, c1 = run1.get(k, (0.0, 0))
    s2, c2 = run2[k]
    per1 = 1000 * s1 / steps1
    per2 = 1000 * s2 / steps2
    ratio = per2 / per1 if per1 else float("inf")
    print(f"{k:<18}{per1:>11.2f}{per2:>11.2f}{ratio:>7.2f}   {c1/steps1:>13.2f}{c2/steps2:>13.2f}")

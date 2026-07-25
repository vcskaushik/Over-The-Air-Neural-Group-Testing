"""Parse HSIC invariance v1 sweep logs into a summary table.

Reads utility logs (main.py --evaluate), leakage.json, and invariance train
stdout ([Diag] lines) for baseline + each lambda, prints a markdown table.
"""
import json, os, re, glob

TM = "Trained_Models"


def util_metrics(logpath):
    """Extract last Acc@1, ROC-AUC, confusion matrix from a main.py --evaluate log."""
    if not os.path.exists(logpath):
        return {}
    txt = open(logpath).read()
    out = {}
    m = re.findall(r"VAL \* Acc@1\s+([\d.]+)", txt)
    if m: out["acc1"] = float(m[-1])
    m = re.findall(r"roc_auc_score\s+([\d.]+)", txt)
    if m: out["auc"] = float(m[-1])
    # last confusion matrix: [[a b] [c d]]
    cms = re.findall(r"confusion_matrix\s*\n\[\[\s*(\d+)\s+(\d+)\]\s*\n\s*\[\s*(\d+)\s+(\d+)\]\]", txt)
    if cms:
        a, b, c, d = map(int, cms[-1])
        out["cm"] = (a, b, c, d)
        out["fp"] = b            # background predicted firearm
        out["recall_fire"] = f"{d}/{c+d}"
    return out


def leakage(path):
    if not os.path.exists(path):
        return None
    return json.load(open(path)).get("top1_imagenet_acc")


def diag(logpath):
    if not os.path.exists(logpath):
        return {}
    txt = open(logpath).read()
    m = re.search(r"\[Diag\] extractor HSIC true=([\d.eE+-]+) permuted=([\d.eE+-]+)", txt)
    warn = "[Diag][WARN]" in txt
    d = {"warn": warn}
    if m:
        d["true"] = float(m.group(1)); d["perm"] = float(m.group(2))
    return d


def row(label, util_log, leak_json, train_log=None):
    u = util_metrics(util_log)
    lk = leakage(leak_json)
    dg = diag(train_log) if train_log else {}
    acc = f"{u.get('acc1','?')}" if u else "?"
    auc = f"{u.get('auc','?')}"
    rec = u.get("recall_fire", "?")
    fp = u.get("fp", "?")
    lks = f"{lk*100:.2f}%" if lk is not None else "?"
    dgs = ""
    if dg:
        if "true" in dg:
            dgs = f"true={dg['true']:.3e} perm={dg['perm']:.3e}" + (" **WARN**" if dg["warn"] else "")
        elif dg.get("warn"):
            dgs = "WARN"
    return f"| {label} | {acc} | {auc} | {rec} | {fp} | {lks} | {dgs} |".replace("{lks}", lks)


print("| config | Acc@1 | ROC-AUC | firearm recall | false-pos | Stage-C leakage | extractor [Diag] |")
print("|---|---|---|---|---|---|---|")
# baseline (Stage A utility from its own log; baseline leakage)
print(row("Stage A baseline",
          f"{TM}/StageA_ITIT_ResNet18_stdout.log",
          f"{TM}/StageC_OnStageA_ResNet18/leakage.json"))
for L in [0, 1, 10, 100, 1000]:
    print(row(f"lam_H={L}",
              f"{TM}/Invariance_ResNet18_lam{L}_utility.log",
              f"{TM}/StageC_Invariance_ResNet18_lam{L}/leakage.json",
              f"{TM}/Invariance_ResNet18_lam{L}_train_stdout.log"))

# refresh diagnostics if present
print("\n### Refresh diagnostics")
for d in sorted(glob.glob(f"{TM}/StageC_Refreshed*/leakage.json")):
    print(f"- {d}: {leakage(d)}")

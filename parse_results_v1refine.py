"""V1-refinement frontier table: whole lambda_H sweep (coarse + fine) with BOTH
the Kaiming Stage-C leakage and the pretrained-adv (honest worst-case) leakage.

Emphasizes the worst-case column, which the HANDOFF-v1-refine names the primary
metric. Reuses the extraction logic from parse_results.py. Missing cells => '?'.
"""
import json, os, re, glob

TM = "Trained_Models"
LAMS = [0, 1, 10, 20, 30, 50, 70, 100, 1000]  # coarse + fine


def util_metrics(logpath):
    if not os.path.exists(logpath):
        return {}
    txt = open(logpath).read()
    out = {}
    m = re.findall(r"VAL \* Acc@1\s+([\d.]+)", txt)
    if m: out["acc1"] = float(m[-1])
    m = re.findall(r"roc_auc_score\s+([\d.]+)", txt)
    if m: out["auc"] = float(m[-1])
    cms = re.findall(r"confusion_matrix\s*\n\[\[\s*(\d+)\s+(\d+)\]\s*\n\s*\[\s*(\d+)\s+(\d+)\]\]", txt)
    if cms:
        a, b, c, d = map(int, cms[-1])
        # cm = [[TN, FP], [FN, TP]]
        out["fp"] = b
        out["recall_fire"] = f"{d}/{c+d}"
        out["fpr"] = b / (a + b) if (a + b) else None   # FP / (FP+TN)
    return out


def leakage(path):
    if not os.path.exists(path):
        return None
    return json.load(open(path)).get("top1_imagenet_acc")


def pct(v):
    return f"{v*100:.2f}%" if v is not None else "?"


def row(label, util_log, kaiming_json, worst_json):
    u = util_metrics(util_log)
    acc = f"{u.get('acc1','?')}"
    auc = f"{u.get('auc','?')}"
    rec = u.get("recall_fire", "?")
    fp = u.get("fp", "?")
    fpr = pct(u.get("fpr")) if u.get("fpr") is not None else "?"
    kai = pct(leakage(kaiming_json))
    wc = pct(leakage(worst_json))
    return f"| {label} | {acc} | {auc} | {rec} | {fp} | {fpr} | {kai} | **{wc}** |"


print("| config | Acc@1 | ROC-AUC | firearm recall | false-pos (count) | FPR | Kaiming leakage | **worst-case (pretrained-adv) leakage** |")
print("|---|---|---|---|---|---|---|---|")
print(row("Stage A baseline",
          f"{TM}/StageA_ITIT_ResNet18_stdout.log",
          f"{TM}/StageC_OnStageA_ResNet18/leakage.json",
          f"{TM}/StageC_OnStageA_ResNet18_advpretrained/leakage.json"))
for L in LAMS:
    print(row(f"lam_H={L}",
              f"{TM}/Invariance_ResNet18_lam{L}_utility.log",
              f"{TM}/StageC_Invariance_ResNet18_lam{L}/leakage.json",
              f"{TM}/StageC_Invariance_ResNet18_lam{L}_advpretrained/leakage.json"))

print("\n### Refresh / worst-case leakage.json values present")
for d in sorted(glob.glob(f"{TM}/StageC_Refreshed*/leakage.json")):
    print(f"- {d}: {pct(leakage(d))}")

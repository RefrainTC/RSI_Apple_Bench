#!/bin/bash
# score_phase2.sh <taskdir> <label> : analysis phase of a score (split from score.sh 2026-11-23, leakage review): needs
# the reference files (calib/calib.json, correct_ref.json, out_ref*.json), which during a trial are sealed in the vault
# and restored only for this phase (remote_score_finish.sh), after every candidate process of the run phase has ended.
# Correctness vs the parent reference, summarize (witness, audit, settings, scored-output checks), verdict, summary.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
TD=$(readlink -f "$1"); L=$2; H=$ISO_ROOT/harness_iso12380; R=$TD/results/$L; LOG=$R/score.log
source "$TD/task.env"
export OUT_MIN_EXACT OUT_TIE_NATS SETTINGS_IGNORE P2_AUDIT_NEED_END P2_AUDIT_IGNORE 2>/dev/null
[ -f "$R/phase1.json" ] || { echo "phase2: no run phase for $L"; exit 2; }
[ -f "$TD/calib/calib.json" ] || { echo "phase2: reference files not in place (sealed?)"; exit 2; }
IPOST=$(python3 -I -c "import json,sys;print(json.load(open(sys.argv[1]))['ipost'])" "$R/phase1.json")
if [ -e "$TD/SEALED" ]; then
  bash $H/integrity.sh verify-ref "$TD" "score_${L}_ref" > "$R/integrity_ref.json" 2>>$LOG || IPOST=1
fi
echo "$(date '+%F %T') PHASE2 START $L" >> $LOG
REF=$(python3 -I -c "import json;print(json.load(open('$TD/calib/calib.json'))['ref_speedup'])")
NOISE=$(python3 -I -c "import json;print(json.load(open('$TD/calib/calib.json'))['noise_band'])")
CORR=$(python3 -I "$TD/compare.py" "$TD/calib/correct_ref.json" "$R/correct/correct.json" $CORRECT_MIN_EXACT $CORRECT_TIE_NATS 2>&1 | tail -1)
python3 -I $H/summarize.py "$TD" "$R/runs" --ref "$REF" > "$R/runs_summary.json"
python3 -I - "$R" "$REF" "$NOISE" "$CORR" "$IPOST" "$TD" "${VERDICT_BAND:-0.005}" <<'PY'
import json, os, sys
sys.path.insert(0, os.path.join(os.environ["ISO_ROOT"], "harness_iso12380")); import audit_check as AC
R, ref, noise, corr, ipost, td, band = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], int(sys.argv[5]), sys.argv[6], float(sys.argv[7])
S = json.load(open(f"{R}/runs_summary.json")); s = S["summary"]
calib = json.load(open(os.path.join(td, "calib", "calib.json")))
try: c = json.loads(corr)
except Exception: c = {"pass": False, "raw": corr}
sc = s.get("sc:cand_vs_parent", {}); sp = sc.get("median_speedup")
scp = [p for p in S["pairs"] if p["label"] == "sc"]
flags = sorted({f for p in scp for f in p["flags"]})
ca = AC.events(os.path.join(R, "correct"), os.environ.get("P2_AUDIT_NEED_END") == "1", os.environ.get("P2_AUDIT_IGNORE", ""))
if ca.get("events") or ca.get("missing_end"): flags.append("audit_correct_run")
ipre = json.load(open(f"{R}/integrity_pre.json")) if os.path.exists(f"{R}/integrity_pre.json") else {}
try: ipo = json.load(open(f"{R}/integrity_post.json"))
except Exception: ipo = {"ok": False}
if os.path.exists(f"{R}/integrity_ref.json"):
    try: ipo = dict(ipo, ref_after_unseal=json.load(open(f"{R}/integrity_ref.json")))
    except Exception: ipo = dict(ipo, ok=False)
    if not (ipo.get("ref_after_unseal") or {}).get("ok"): ipo["ok"] = False
if ipost != 0 or not ipo.get("ok") or "venv_changed" in flags: verdict = "integrity_fail"
elif "audit" in flags or "audit_correct_run" in flags: verdict = "audit_flag"
elif "timing_witness" in flags: verdict = "timing_witness_fail"
elif "settings_mismatch" in flags: verdict = "settings_mismatch"
elif "output_mismatch" in flags: verdict = "output_mismatch"
elif not c.get("pass"): verdict = "correctness_fail"
elif sp is None: verdict = "no_measurement"
elif sp >= ref * (1 + band): verdict = "above_ref"
elif sp >= ref * (1 - band): verdict = "ties_ref"
elif sp > 1 + band: verdict = "above_baseline_below_ref"
else: verdict = "no_gain"
valid = verdict in ("above_ref", "ties_ref", "above_baseline_below_ref", "no_gain")
out = {"verdict": verdict, "speedup": sp, "ref_speedup": ref, "noise_band": noise, "verdict_band": band,
       "reward": (max(0.0, min(1.0, (sp - 1) / (ref - 1))) if (valid and sp and ref > 1) else 0.0),
       "speedup_over_ref": (sp / ref if (valid and sp) else None), "pairs": sc, "flags": flags, "correctness": c,
       "pair_checks": [{k: p.get(k) for k in ("pair", "speedup", "flags", "outcheck", "settings_diff")} for p in scp],
       "transfer": {k: v for k, v in s.items() if k.startswith("tr")}}
# environment sentinel (review F9): parent runs outside +-1.5 % of the calibration's parent median -> suspect pair
pm = (calib.get("parent_primary") or {}).get("median")
if pm:
    sent = [p["pair"] for p in scp for m in ([p["m_first"]] if p["first"] == "parent" else [p["m_second"]]) if abs(m / pm - 1) > 0.015]
    out["environment_suspect_pairs"] = sent
cs = calib.get("storage") or ["nas"]
out.update(storage=S.get("storage"), mixed_storage=S.get("mixed_storage"), calib_storage=cs, storage_matches_calib=(S.get("storage") == cs),
           cpus=S.get("cpus"), cache_states=S.get("cache_states"), hosts=S.get("hosts"), peak_mem_mib=S.get("peak_mem_mib"),
           integrity={"pre": ipre, "post": ipo}, audit_correct_run=ca)
json.dump(out, open(f"{R}/summary.json", "w"), indent=1, default=str)
lines = [f"# score {R}", f"- verdict: {verdict}" + (f" (flags: {', '.join(flags)})" if flags else ""),
         f"- speedup {sp} (REF {ref:.4f}, verdict band ±{band:.4f}, A/A noise ±{noise:.4f}), reward {out['reward']:.3f}",
         f"- storage: {out['storage']} (calibration: {cs}, match: {out['storage_matches_calib']}); cpus {out['cpus']}; caches {out['cache_states']}",
         f"- integrity: pre {ipre.get('ok')} post {ipo.get('ok')}",
         f"- correctness: pass={c.get('pass')}"] + [f"- pair {p['pair']}: speedup {p['speedup']:.4f} flags {p['flags']} outcheck {(p.get('outcheck') or {}).get('pass')}" for p in scp] \
        + [f"- transfer {k}: {v.get('median_speedup')}" for k, v in out["transfer"].items()]
open(f"{R}/summary.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
PY

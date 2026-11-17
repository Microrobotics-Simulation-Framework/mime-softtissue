"""Calibration (samples 4..7) and the held-out validation (leave-one-out, samples 1 and 2) of studies.visco_clot. python -I scripts/visco_calibrate.py OUT [shape_free]"""
import json, sys, time
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from mime_softtissue.studies import visco_clot as vc

out_path = sys.argv[1]
S = vc.samples()
cal = {k: S[k] for k in vc.CALIBRATION}
t0 = time.time()
R = {"plan": "the split and lines committed at 12:18 CET on 2026-11-16, before any force was read", "split": {
    "calibration": list(vc.CALIBRATION), "validation": list(vc.VALIDATION) + ["leave-one-out over 4..7"]}}

SHAPE_FREE = len(sys.argv) > 2 and sys.argv[2] == "shape_free"
R["diagnostic_shape_free"] = SHAPE_FREE
pooled = vc.fit(cal, shape_free=SHAPE_FREE)
R["pooled"] = pooled
print("pooled", {k: pooled[k] for k in ("beta", "tau_s", "c2_over_c1", "rms_over_peak", "max_abs_over_peak", "status")}, flush=True)
print("  amplitude", pooled["amplitude_Pa"], "offset", pooled["offset_mN"])
print("  corr", np.round(np.array(pooled["correlation"]), 3).tolist())
print("  theta sd", np.round(pooled["theta_sd"], 4).tolist())

R["profile_tau1"] = [] if SHAPE_FREE else [vc.profile(cal, t1) for t1 in (0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0)]
for row in R["profile_tau1"]:
    print("profile", {k: (np.round(v, 4).tolist() if not isinstance(v, float) else round(v, 5)) for k, v in row.items()}, flush=True)
per = {}
for k in vc.CALIBRATION:
    per[k] = vc.fit({k: S[k]})
    print("sample", k, "beta", np.round(per[k]["beta"], 4), "tau", np.round(per[k]["tau_s"], 3), "rms",
          round(per[k]["rms_over_peak"], 5), "A", round(per[k]["amplitude_Pa"][k], 1), flush=True)
R["per_sample"] = per
B = np.array([per[k]["beta"] for k in vc.CALIBRATION]); T = np.array([per[k]["tau_s"] for k in vc.CALIBRATION])
R["per_sample_spread"] = {"beta_mean": B.mean(0).tolist(), "beta_sd": B.std(0, ddof=1).tolist(),
                          "tau_mean": T.mean(0).tolist(), "tau_sd": T.std(0, ddof=1).tolist(),
                          "beta_range": [B.min(0).tolist(), B.max(0).tolist()],
                          "tau_range": [T.min(0).tolist(), T.max(0).tolist()]}
X = np.concatenate([B, T], axis=1)
R["per_sample_correlation_(b1,b2,t1,t2)"] = np.corrcoef(X.T).tolist()
print("spread", R["per_sample_spread"])

loo = {}
for k in vc.CALIBRATION:
    rest = {j: S[j] for j in vc.CALIBRATION if j != k}
    f = vc.fit(rest, shape_free=SHAPE_FREE)
    pred = vc.predicted_hold(S[k], vc.prony_of(np.asarray(f["theta"][:4])), f["c2_over_c1"])
    pred.update({"beta": f["beta"], "tau_s": f["tau_s"],
                 "passes": abs(pred["end_ratio_error"]) <= 0.05 and pred["hold_rms_over_peak"] <= 0.05})
    loo[k] = pred
    print("LOO", k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in pred.items()}, flush=True)
R["loo"] = loo

pr = vc.prony_of(np.asarray(pooled["theta"][:4]))
shape = pooled["c2_over_c1"]
# V2: sample 2, amplitude from 0..1 mm, the force to 4 mm predicted
s2 = S["2"]
m2, a2, b2 = vc.first_millimetre(s2, pr, shape)
sel = s2.displacement <= 4.0
f4 = float(np.interp(4.0, s2.displacement, s2.force))
rms = float(np.sqrt(np.mean((m2[sel] - s2.force[sel]) ** 2)))
m2e, a2e, b2e = vc.first_millimetre(s2, None, shape)
R["V2"] = {"amplitude_Pa": a2, "offset_mN": b2, "rms_to_4mm_over_F4": rms / f4, "passes": rms / f4 <= 0.10,
           "F_at": {str(x): [float(np.interp(x, s2.displacement, s2.force)), float(np.interp(x, s2.displacement, m2))]
                    for x in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)},
           "elastic_same_shape_rms_over_F4": float(np.sqrt(np.mean((m2e[sel] - s2.force[sel]) ** 2))) / f4}
print("V2", R["V2"], flush=True)

# V1: sample 1 cycles
s1 = S["1"]
m1, a1, b1 = vc.first_millimetre(s1, pr, shape)
cyc = vc.cycles(s1)
rows = []
for i, c in enumerate(cyc):
    if c["end"] is None:
        continue
    hd = vc.hysteresis(s1.displacement, s1.force, c)
    hm = vc.hysteresis(s1.displacement, m1, c)
    row = {"cycle": i + 1, "peak_mm": hd["peak_mm"], "data_ratio": hd["ratio"], "model_ratio": hm["ratio"],
           "data_peak_mN": float(s1.force[c["peak"]]), "model_peak_mN": float(m1[c["peak"]]),
           "unload_end_mm": float(s1.displacement[c["end"]]), "data_end_mN": float(s1.force[c["end"]]),
           "model_end_mN": float(m1[c["end"]])}
    if i + 1 < len(cyc):
        nxt = cyc[i + 1]
        seg = slice(nxt["start"], nxt["peak"] + 1)
        row["reload_at_prev_peak_data_mN"] = float(np.interp(hd["peak_mm"], s1.displacement[seg], s1.force[seg]))
        row["reload_at_prev_peak_model_mN"] = float(np.interp(hd["peak_mm"], s1.displacement[seg], m1[seg]))
    rows.append(row)
    print("V1 cycle", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
inside = [r for r in rows if r["peak_mm"] <= 4.05]
R["V1"] = {"amplitude_Pa": a1, "offset_mN": b1, "cycles": rows,
           "hysteresis_pass": [abs(r["model_ratio"] / r["data_ratio"] - 1) <= 0.30 for r in inside],
           "reload_pass": [abs(r["reload_at_prev_peak_model_mN"] / r["reload_at_prev_peak_data_mN"] - 1) <= 0.15
                           for r in inside if "reload_at_prev_peak_data_mN" in r]}
print("V1 pass", R["V1"]["hysteresis_pass"], R["V1"]["reload_pass"])
# the published times, held (declared), betas fitted
if not SHAPE_FREE:
    from scipy.optimize import least_squares
    pub = least_squares(lambda b: vc.residuals(np.array([b[0], np.log(3.0), b[1], np.log(61.0)]), cal), np.log([0.3, 0.1]),
                        bounds=(np.log([1e-4, 1e-4]), np.log([10.0, 10.0])), diff_step=1e-4)
    R["published_times_fit"] = {"beta": np.exp(pub.x).tolist(), "tau_s": [3.0, 61.0],
                                "rms_over_peak": float(np.sqrt(np.mean(pub.fun ** 2)))}
    print("published times", R["published_times_fit"], flush=True)
R["published"] = {"tau_s": [3.0, 61.0], "set_percent_cyclic": 16, "source": "Sugerman et al. 2024 (read via LIT_REPLICATE)"}
R["wall_s"] = time.time() - t0
json.dump(R, open(out_path, "w"), indent=1)
print("wall", R["wall_s"])

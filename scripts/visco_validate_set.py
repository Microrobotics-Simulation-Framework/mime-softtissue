"""The demo's set (tau declared = the published 3.0 s and 61 s; beta fitted): the held-out checks of visco_calibrate.py.
python -I scripts/visco_validate_set.py OUT"""
import json, sys
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from scipy.optimize import least_squares
from mime_softtissue.studies import visco_clot as vc

TAU = np.array(vc.PUBLISHED_TAU)
S = vc.samples()


def fit_beta(group):
    r = least_squares(lambda b: vc.residuals(np.array([b[0], np.log(TAU[0]), b[1], np.log(TAU[1])]), group),
                      np.log([0.5, 0.15]), bounds=(np.log([1e-4, 1e-4]), np.log([10.0, 10.0])), diff_step=1e-4)
    J = r.jac
    s2 = float(r.fun @ r.fun) / max(len(r.fun) - 2 - len(group), 1)
    cov = s2 * np.linalg.inv(J.T @ J)
    sd = np.sqrt(np.diag(cov))
    return np.exp(r.x), float(np.sqrt(np.mean(r.fun ** 2))), float(np.max(np.abs(r.fun))), sd, float(cov[0, 1] / sd[0] / sd[1])


R = {"tau_s": TAU.tolist(), "tau_label": "declared: Sugerman et al. 2024 (read via LIT_REPLICATE)"}
cal = {k: S[k] for k in vc.CALIBRATION}
beta, rms, mx, sd_log, corr = fit_beta(cal)
R["pooled"] = {"beta": beta.tolist(), "rms_over_peak": rms, "max_abs_over_peak": mx, "ln_beta_sd": sd_log.tolist(),
               "corr_ln_beta": corr}
pr = vc.prony_of(np.array([np.log(beta[0]), np.log(TAU[0]), np.log(beta[1]), np.log(TAU[1])]))
R["amplitude_Pa"] = {k: vc.amplitude(vc.replay(S[k], pr), S[k].force, vc.offset(S[k]))[0] for k in vc.CALIBRATION}
print("pooled", R["pooled"], R["amplitude_Pa"], flush=True)
per = {}
for k in vc.CALIBRATION:
    b, r, m, _, _ = fit_beta({k: S[k]})
    per[k] = {"beta": b.tolist(), "rms_over_peak": r}
    print("sample", k, per[k], flush=True)
B = np.array([per[k]["beta"] for k in vc.CALIBRATION])
R["per_sample"] = per
R["per_sample_spread"] = {"mean": B.mean(0).tolist(), "sd": B.std(0, ddof=1).tolist(), "min": B.min(0).tolist(),
                          "max": B.max(0).tolist(), "corr_b1_b2": float(np.corrcoef(B.T)[0, 1])}
loo = {}
for k in vc.CALIBRATION:
    b, *_ = fit_beta({j: S[j] for j in vc.CALIBRATION if j != k})
    p = vc.predicted_hold(S[k], vc.prony_of(np.array([np.log(b[0]), np.log(TAU[0]), np.log(b[1]), np.log(TAU[1])])))
    p["beta"] = b.tolist()
    p["passes_ratio"] = abs(p["end_ratio_error"]) <= 0.05
    p["passes_rms"] = p["hold_rms_over_peak"] <= 0.05
    loo[k] = p
    print("LOO", k, {a: (round(v, 4) if isinstance(v, float) else v) for a, v in p.items()}, flush=True)
R["loo"] = loo
s2 = S["2"]
m2, a2, _ = vc.first_millimetre(s2, pr)
sel = s2.displacement <= 4.0
f4 = float(np.interp(4.0, s2.displacement, s2.force))
R["V2"] = {"amplitude_Pa": a2, "rms_to_4mm_over_F4": float(np.sqrt(np.mean((m2[sel] - s2.force[sel]) ** 2))) / f4,
           "F_at_mm": {str(x): [float(np.interp(x, s2.displacement, s2.force)), float(np.interp(x, s2.displacement, m2))]
                       for x in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)}}
print("V2", R["V2"], flush=True)
s1 = S["1"]
m1, a1, _ = vc.first_millimetre(s1, pr)
cyc = vc.cycles(s1)
rows = []
for i, c in enumerate(cyc):
    if c["end"] is None:
        continue
    hd, hm = vc.hysteresis(s1.displacement, s1.force, c), vc.hysteresis(s1.displacement, m1, c)
    row = {"cycle": i + 1, "peak_mm": hd["peak_mm"], "data_ratio": hd["ratio"], "model_ratio": hm["ratio"],
           "data_peak_mN": float(s1.force[c["peak"]]), "model_peak_mN": float(m1[c["peak"]]),
           "unload_end_mm": float(s1.displacement[c["end"]]), "model_end_mN": float(m1[c["end"]])}
    if i + 1 < len(cyc):
        seg = slice(cyc[i + 1]["start"], cyc[i + 1]["peak"] + 1)
        row["reload_data_mN"] = float(np.interp(hd["peak_mm"], s1.displacement[seg], s1.force[seg]))
        row["reload_model_mN"] = float(np.interp(hd["peak_mm"], s1.displacement[seg], m1[seg]))
    rows.append(row)
    print("V1", {a: (round(v, 4) if isinstance(v, float) else v) for a, v in row.items()}, flush=True)
inside = [r for r in rows if r["peak_mm"] <= 4.05]
R["V1"] = {"amplitude_Pa": a1, "cycles": rows,
           "hysteresis_pass": [abs(r["model_ratio"] / r["data_ratio"] - 1) <= 0.30 for r in inside],
           "reload_pass": [abs(r["reload_model_mN"] / r["reload_data_mN"] - 1) <= 0.15 for r in inside if "reload_data_mN" in r]}
print("V1 pass", R["V1"]["hysteresis_pass"], R["V1"]["reload_pass"])
s, srms = vc.rate_consistent_scale(pr)
R["rate_consistent_scale"] = {"s": s, "rms_over_max": srms, "grip_rate_mm_s": 0.2}
s01, _ = vc.rate_consistent_scale(pr, grip_rate=0.1e-3)
R["rate_consistent_scale"]["s_if_0.1_mm_s"] = s01
print("s", R["rate_consistent_scale"])
json.dump(R, open(sys.argv[1], "w"), indent=1)

"""
analyze_phase0.py -- quantitative validation of a completed Phase 0 run.
[FIXED VERSION -- see accompanying writeup for what changed and why]

Computes the single most important number the plots don't show you: the
normalized reconnection rate R = max(dPsi/dt) / (vA_upstream * B0). This is
the headline cross-code result of the original GEM Challenge (Birn et al.
2001) -- resistive AND Hall codes were compared on it. IMPORTANT LITERATURE
CAVEAT (this was missing before): Birn et al. 2001 explicitly found that
plain, spatially-uniform resistive MHD does NOT reach R~0.1 -- it gives a
"dramatically smaller" Sweet-Parker-like rate unless resistivity is
localized or current-dependent. Only the Hall-mediated (or localized-eta)
runs converge to the fast R~0.1 rate. So: judge the HALL run against
~0.1. Do NOT expect the uniform-eta resistive-only run to get there --
a low resistive R is not automatically a bug.

THE BUG THIS VERSION FIXES: the previous version computed R from
`az1_hist`, the flux sampled at a FIXED spatial grid point (the initial
sheet-center location). But the X-line drifts in x once the tearing mode
saturates and islands interact (visible directly in
fixed_point_vs_xline_flux.png: by t=40 the fixed-point flux for the Hall
run has only reached ~0.31, while the X-line-tracked flux has reached
~4.8 -- more than 15x larger). Differencing the fixed-point flux therefore
massively UNDER-estimates dPsi/dt once the X-line has moved away from the
sampling point, which is exactly the failure mode that produced R~0.01
instead of R~0.1 for the Hall run. This version uses the X-line-tracked
flux (az*_xline, already computed by the solver but never plumbed into
this script) instead.

It also adds a second, independent measurement of the same quantity: the
solver's generalized Ohm's law E_z evaluated AT the tracked X-line
(ez*_xline), saved directly during the run. By Faraday's law this Ez *is*
dPsi/dt at that point, with no finite-differencing/noise involved -- it is
the quantity most GEM Challenge papers actually report. Comparing it
against the finite-differenced X-line flux rate is a genuine internal
consistency check: if the two disagree substantially, that's a sign the
X-line tracker, the time resolution (n_diag), or something else in the
diagnostic chain still needs attention -- not a subtle physics effect.

Also fits the early-time exponential growth phase to extract a growth
rate gamma. FIX: the previous fit window (t<5.0, no lower bound) silently
included the initial perturbation-relaxation transient (t<2, the same
window the peak-R logic elsewhere in this script explicitly excludes),
so the reported gamma was contaminated by transient relaxation dynamics,
not pure linear tearing growth. This version starts the fit window at
TRANSIENT_CUTOFF and picks the longest window from there whose log(flux)
vs t is actually linear (R^2 >= R2_MIN), reporting the fit quality rather
than assuming a fixed window is the right one.

Usage:
    python3 analyze_phase0.py phase0_output_20260927_192257_midseed/

Requires a run produced with the patched gem_hall_mhd_phase0.py that
saves az*_xline and ez*_xline in the .npz files. If those keys are
missing (an older run), this script falls back to the old fixed-point
number with a loud warning -- rerun the simulation with the patched
solver before trusting anything quantitative.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print("Usage: python3 analyze_phase0.py <phase0_output_dir>/")
    sys.exit(1)

OUTDIR = sys.argv[1]
TRANSIENT_CUTOFF = 2.0  # exclude the initial perturbation-relaxation burst
R2_MIN = 0.99           # minimum linear-fit quality accepted for gamma
MIN_FIT_POINTS = 5

# Upstream (asymptotic, far from the sheet) reference values used for
# normalization -- matches the convention used in the GEM Challenge papers
# of normalizing by the inflow Alfven speed, not a peak/local value.
B0 = 1.0
N_INF = 0.2  # background density far from the sheet (matches solver's n_inf)
VA_UP = B0 / np.sqrt(N_INF)

print(f"Upstream Alfven speed used for normalization: vA_up = {VA_UP:.4f} "
      f"(B0={B0}, n_inf={N_INF})")
print("Reference (Birn et al. 2001): Hall-mediated / localized-resistivity "
      "runs converge to peak normalized reconnection rate R ~ 0.1. Plain "
      "uniform-eta resistive MHD is NOT expected to reach this -- only "
      "judge the Hall run against 0.1.\n")


GLITCH_FACTOR = 8.0  # a step this many times the local median step size is
# treated as the X-line tracker having jumped to an unrelated point (e.g. a
# plasmoid/O-point) rather than genuine smooth reconnection -- Az cannot
# physically teleport, so a multi-unit single-step change is a diagnostic
# artifact, not fast physics.


def find_tracker_glitches(t, flux, factor=GLITCH_FACTOR):
    """Flag indices adjacent to an unphysically large single-step jump in
    the X-line-tracked flux. Returns a boolean 'untrusted' mask (True =
    exclude from quantitative peak/gamma searches) and the list of glitch
    timestamps for reporting.
    """
    n = len(flux)
    untrusted = np.zeros(n, dtype=bool)
    glitch_times = []
    if n < 3:
        return untrusted, glitch_times
    diffs = np.diff(flux)
    nonzero = np.abs(diffs) > 0
    if not np.any(nonzero):
        return untrusted, glitch_times
    step_med = np.median(np.abs(diffs[nonzero]))
    if step_med <= 0:
        return untrusted, glitch_times
    for i, d in enumerate(diffs):
        if abs(d) > factor * step_med:
            untrusted[i] = True
            untrusted[i + 1] = True
            glitch_times.append((t[i], t[i + 1], d))
    return untrusted, glitch_times


def best_linear_window(t, logflux, t_min, r2_min=R2_MIN, min_points=MIN_FIT_POINTS):
    """Find the longest contiguous window starting at t_min over which
    logflux vs t is well-described by a line (R^2 >= r2_min). Returns
    (gamma, intercept, r2, i_start, i_end) or None if nothing qualifies.
    Grows the window greedily from the shortest valid length so a late
    departure into saturation truncates the fit rather than corrupting it.
    """
    idx0 = int(np.searchsorted(t, t_min))
    n = len(t)
    if n - idx0 < min_points:
        return None

    best = None
    for i_end in range(idx0 + min_points, n + 1):
        tt = t[idx0:i_end]
        yy = logflux[idx0:i_end]
        if not np.all(np.isfinite(yy)):
            break
        coeffs = np.polyfit(tt, yy, 1)
        fit = np.polyval(coeffs, tt)
        ss_res = np.sum((yy - fit) ** 2)
        ss_tot = np.sum((yy - np.mean(yy)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        if r2 >= r2_min:
            best = (coeffs[0], coeffs[1], r2, idx0, i_end)
        elif best is not None:
            # linearity broke down (saturation) -- stop extending
            break
    return best


all_curves = {}          # label -> (t, R_from_flux, R_direct)  [X-line based]
xline_flux_curves = {}   # label -> (t, flux_fixed, flux_xline)

for label in ["resistive", "hall"]:
    path = os.path.join(OUTDIR, f"{label}_timeseries.npz")
    if not os.path.exists(path):
        print(f"[{label}] {path} not found, skipping")
        continue

    data = np.load(path)
    t = data["t"]
    az1 = data["az1"]
    flux_fixed = az1 - az1[0]

    has_xline = "az1_xline" in data.files
    has_ez = "ez1_xline" in data.files

    if not has_xline:
        print(f"[{label}] WARNING: this run has no az1_xline saved (older "
              f"solver). Falling back to the FIXED-POINT diagnostic, which "
              f"is the known-broken one -- rerun with the patched solver "
              f"before trusting this number.")
        flux = flux_fixed
        t_use = t
    else:
        flux = data["az1_xline"] - data["az1_xline"][0]
        t_use = t
        xline_flux_curves[label] = (t, flux_fixed, flux)

    untrusted, glitch_times = (
        find_tracker_glitches(t_use, flux) if has_xline
        else (np.zeros_like(t_use, dtype=bool), [])
    )
    if glitch_times:
        print(f"[{label}] X-LINE TRACKER GLITCH: {len(glitch_times)} "
              f"unphysical single-step jump(s) detected in the tracked "
              f"flux (Az cannot change this fast physically -- the tracker "
              f"jumped to an unrelated point, likely a plasmoid/O-point). "
              f"Excluding these frames from the peak-R and gamma searches:")
        for t0, t1, d in glitch_times:
            print(f"    jump of {d:+.3f} between t={t0:.2f} and t={t1:.2f}")
        print(f"    (rerun with the updated saddle-point tracker in "
              f"gem_hall_mhd_phase0.py to fix this at the source)")

    # --- Rate from finite-differencing the (X-line-tracked, if available) flux
    dpsi_dt = np.gradient(flux, t_use)
    R_flux = dpsi_dt / (VA_UP * B0)

    late_mask = (t_use >= TRANSIENT_CUTOFF) & (~untrusted)
    if np.any(late_mask):
        i_local = int(np.argmax(np.abs(R_flux[late_mask])))
        peak_R_flux = R_flux[late_mask][i_local]
        peak_t_flux = t_use[late_mask][i_local]
    else:
        peak_R_flux, peak_t_flux = float("nan"), float("nan")

    print(f"[{label}] peak R from d/dt of "
          f"{'X-line-tracked' if has_xline else 'FIXED-POINT (fallback)'} "
          f"flux, excluding t<{TRANSIENT_CUTOFF}: "
          f"R = {peak_R_flux:.4f} at t = {peak_t_flux:.2f}")

    # --- Direct rate: Ez sampled at the X-line itself (no differentiation)
    R_direct = None
    if has_ez:
        ez_xline = data["ez1_xline"]
        # SIGN FIX: the solver's flux function is defined by Bx=dAz/dy,
        # By=-dAz/dx (see reconstruct_Az / the Jz=-lap(Az) identity it
        # enforces), and Faraday's law dB/dt=-curl(E) with that convention
        # gives d(Az)/dt = -Ez, not +Ez -- derivable directly from the
        # solver's own induction-equation lines dBxdt=-ddy(Ez),
        # dBydt=ddx(Ez). The previous version of this script used +Ez,
        # which silently flipped the sign of every "direct" rate relative
        # to the (unambiguous, differencing-based) flux rate -- this is why
        # the two independent measures kept disagreeing by >400% even after
        # the tracker's position itself was fixed: they were often
        # comparing a positive number to a negative one.
        R_direct_curve = -ez_xline / (VA_UP * B0)
        if np.any(late_mask):
            i_local = int(np.argmax(np.abs(R_direct_curve[late_mask])))
            peak_R_direct = R_direct_curve[late_mask][i_local]
            peak_t_direct = t_use[late_mask][i_local]
        else:
            peak_R_direct, peak_t_direct = float("nan"), float("nan")
        print(f"[{label}] peak R DIRECT (E_z at tracked X-line, no "
              f"differencing), excluding t<{TRANSIENT_CUTOFF}: "
              f"R = {peak_R_direct:.4f} at t = {peak_t_direct:.2f}  "
              f"<-- compare this one to 0.1 for the Hall run")
        R_direct = R_direct_curve

        if np.isfinite(peak_R_flux) and abs(peak_R_direct) > 1e-6:
            rel_disagreement = abs(peak_R_flux - peak_R_direct) / abs(peak_R_direct)
            if rel_disagreement > 0.3:
                if glitch_times:
                    print(f"[{label}] NOTE: the two rate estimates disagree "
                          f"by {100*rel_disagreement:.0f}% even after "
                          f"excluding the flagged glitch frames -- some "
                          f"residual tracker drift may remain; inspect the "
                          f"plot near the glitch timestamps above.")
                else:
                    print(f"[{label}] NOTE: the two rate estimates disagree "
                          f"by {100*rel_disagreement:.0f}% -- likely n_diag "
                          f"is too coarse to resolve the burst in time (try "
                          f"--n-diag 400+).")
    else:
        print(f"[{label}] no ez1_xline saved (older solver) -- only the "
              f"finite-differenced rate above is available; rerun with the "
              f"patched solver for the direct, less noisy measurement.")

    # --- Is a flagged "glitch" window actually a real fast-reconnection
    # burst rather than a tracker artifact? A genuine burst is EXPECTED to
    # have steps many times the surrounding median (that's what "fast"
    # means), so the >8x-median-step rule alone cannot tell the two apart.
    # A tracker relabeling artifact and real fast reconnection make
    # different, checkable predictions: a relabeling jump changes WHERE the
    # sample is taken but not the true local dPsi/dt, so it should NOT show
    # up correspondingly in the direct (non-differenced) Ez measurement;
    # real fast reconnection should, since Ez is measured independently of
    # the flux history. Comparing the SIGN and rough MAGNITUDE of the two
    # measures over the flagged window is therefore a real physics check,
    # not just another arbitrary threshold.
    if glitch_times and has_ez:
        for t0, t1, d in glitch_times:
            in_window = (t_use >= t0) & (t_use <= t1)
            if not np.any(in_window):
                continue
            mean_flux_rate = float(np.mean(R_flux[in_window]))
            mean_direct_rate = float(np.mean(R_direct[in_window]))
            same_sign = np.sign(mean_flux_rate) == np.sign(mean_direct_rate)
            ratio = (abs(mean_flux_rate) / abs(mean_direct_rate)
                     if abs(mean_direct_rate) > 1e-8 else float("inf"))
            verdict = ("plausibly REAL fast reconnection (both measures "
                       "agree in sign, same order of magnitude) -- do not "
                       "just discard this window without looking at it"
                       if same_sign and ratio < 5.0 else
                       "still looks like a tracker artifact (measures "
                       "disagree in sign or by a large factor)")
            print(f"[{label}] glitch window t=[{t0:.2f},{t1:.2f}]: "
                  f"mean R(flux)={mean_flux_rate:+.4f}, "
                  f"mean R(direct)={mean_direct_rate:+.4f} -> {verdict}")

    all_curves[label] = (t_use, R_flux, R_direct, glitch_times)

    # --- Growth-rate fit, restricted to AFTER the initial transient and to
    # a window that is actually verified to be linear in log-space.
    logflux = np.full_like(flux, np.nan)
    pos = (flux > 1e-4) & (~untrusted)
    logflux[pos] = np.log(flux[pos])
    fit = best_linear_window(t_use, logflux, TRANSIENT_CUTOFF)
    if fit is not None:
        gamma, intercept, r2, i0, i1 = fit
        print(f"[{label}] early-time exponential growth rate (transient "
              f"excluded): gamma ~ {gamma:.4f}  (fit over t in "
              f"[{t_use[i0]:.2f}, {t_use[i1-1]:.2f}], n={i1-i0} points, "
              f"R^2={r2:.4f})")
    else:
        print(f"[{label}] could not find a window after t={TRANSIENT_CUTOFF} "
              f"with R^2>={R2_MIN} for a growth-rate fit -- check n_diag, "
              f"or the flux may never pass through a clean linear-growth "
              f"regime at this resolution")

    headline_R = peak_R_direct if has_ez else peak_R_flux
    if np.isfinite(headline_R):
        if abs(headline_R) < 0.02 and label == "hall":
            print(f"[{label}] WARNING: peak R is well below the ~0.1 "
                  f"Hall-mediated benchmark -- reconnection may be "
                  f"under-resolved, or the run hasn't reached the fast "
                  f"phase yet (check whether R is still rising at t_final "
                  f"in the plot below)")
        elif abs(headline_R) > 0.5:
            print(f"[{label}] WARNING: peak R is well above ~0.1 -- worth "
                  f"double-checking this isn't a transient/noise spike "
                  f"rather than the sustained reconnection rate")
    print()

if all_curves:
    plt.figure(figsize=(8, 5))
    glitch_label_used = False
    for label, (t_use, R_flux, R_direct, glitch_times) in all_curves.items():
        plt.plot(t_use, R_flux, alpha=0.5, linestyle="--",
                  label=f"{label} (d/dt of X-line flux)")
        if R_direct is not None:
            plt.plot(t_use, R_direct, label=f"{label} (direct E_z at X-line)")
        for t0, t1, _ in glitch_times:
            plt.axvspan(t0, t1, color="red", alpha=0.15,
                        label="tracker glitch (excluded)" if not glitch_label_used else None)
            glitch_label_used = True
    plt.axhline(0.1, color="gray", linestyle="--", label="GEM benchmark (~0.1)")
    plt.axvline(TRANSIENT_CUTOFF, color="gray", linestyle=":", alpha=0.6,
                label=f"transient cutoff (t={TRANSIENT_CUTOFF})")
    plt.xlabel("time")
    plt.ylabel("normalized reconnection rate R")
    plt.title("Reconnection rate vs time -- X-line-tracked, two independent measures")
    plt.legend(fontsize=8)
    plt.tight_layout()
    outpath = os.path.join(OUTDIR, "reconnection_rate.png")
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved: {outpath}")
    print("Solid lines are the direct E_z-at-X-line measurement (trust "
          "these); dashed lines are the finite-differenced X-line flux, "
          "shown only as a cross-check. If R is still rising at t_final "
          "rather than having peaked and come back down, the run needs a "
          "longer t_final to capture the real fast-reconnection phase.")

print()

if xline_flux_curves:
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, (t, flux_fixed, flux_xline) in xline_flux_curves.items():
        axs[0].plot(t, flux_fixed, label=f"{label} (fixed point)", linestyle="--")
        axs[1].plot(t, flux_xline, label=f"{label} (X-line tracked)")
    axs[0].set_xlabel("time")
    axs[0].set_ylabel("Az - Az(0)")
    axs[0].set_title("Old diagnostic: fixed spatial point (do not use for R)")
    axs[0].legend()
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("Az - Az(0)")
    axs[1].set_title("X-line-tracked flux (used for R above)")
    axs[1].legend()
    plt.tight_layout()
    xline_path = os.path.join(OUTDIR, "fixed_point_vs_xline_flux.png")
    plt.savefig(xline_path, dpi=150)
    plt.close()
    print(f"Saved: {xline_path}")

    if "resistive" in xline_flux_curves and "hall" in xline_flux_curves:
        t_res, _, res_xline = xline_flux_curves["resistive"]
        t_hall, _, hall_xline = xline_flux_curves["hall"]
        t_res_end, t_hall_end = t_res[-1], t_hall[-1]

        if abs(t_res_end - t_hall_end) > 0.5:
            t_common = min(t_res_end, t_hall_end)
            i_res = int(np.argmin(np.abs(t_res - t_common)))
            i_hall = int(np.argmin(np.abs(t_hall - t_common)))
            print(f"NOTE: resistive ended at t={t_res_end:.2f} and hall ended "
                  f"at t={t_hall_end:.2f} -- comparing at the shared time "
                  f"t~{t_common:.2f} instead of raw 'final' values:")
            print(f"  resistive at t={t_res[i_res]:.2f}: flux={res_xline[i_res]:.4f}")
            print(f"  hall at t={t_hall[i_hall]:.2f}: flux={hall_xline[i_hall]:.4f}")
        else:
            print(f"Final X-line-tracked flux: resistive={res_xline[-1]:.4f}, "
                  f"hall={hall_xline[-1]:.4f}")
        print("Per Birn et al. (2001), Hall-mediated reconnection is expected "
              "to run ahead of plain uniform-eta resistive MHD once both are "
              "measured correctly -- that is the headline GEM result, not an "
              "anomaly to explain away.")
else:
    print("No X-line-tracked data found in the .npz files (older run, or "
          "field not saved) -- rerun with the patched solver.")
"""
gem_hall_mhd_phase0.py -- Phase 0, clean build.

Reduced 2.5D compressible Hall-MHD solver on a doubly-periodic double
Harris-sheet variant of the GEM magnetic reconnection challenge
(Birn et al., JGR 106, 3715-3719, 2001).

PARAMETERS AND WHERE THEY COME FROM (checked against the literature for
this rebuild, not carried over from earlier drafts):

  eta   = 0.005   Resistivity. Confirmed as the standard value used across
                  published Hall-MHD reproductions of this benchmark
                  (e.g. "Systematic 2.5D resistive MHD simulations with
                  ambipolar diffusion and Hall effect for fast magnetic
                  reconnection", which states explicitly: "In all
                  simulations we set the resistivity to eta = 0.005").
                  A distinct, unverified "eta=0.001" figure was used in
                  an earlier draft of this project and could not be
                  traced to any actual source on re-check -- treat 0.005
                  as the validated default, 0.001 as an open stress-test
                  to run deliberately later, not a baseline.
  psi0  = 0.1     Perturbation amplitude. Standard value, confirmed.
  lam   = 0.5     Harris-sheet half-width (d_i units). Standard, confirmed.
  n_inf = 0.2     Background density. Standard, confirmed.
  cs2   = 0.5     Sound speed^2, from p = rho/2 (B0=1, rho0=1). Confirmed
                  directly against the same source as eta.
  d_i   = 1.0     Ion inertial length / Hall coefficient, normalized to 1.
                  Standard convention.
  T_FINAL = 40    In Omega_ci^-1 (ion-cyclotron) time units -- the usual
                  reporting window for this benchmark.

DELIBERATE DEVIATION FROM THE SINGLE-SHEET TEXTBOOK FORM, DOCUMENTED
(not silent): the canonical single-sheet perturbation is a *global*
sinusoid spanning the whole box, e.g. dBx ~ cos(2 pi x/Lx) sin(pi
(y-y0)/Ly). This solver instead uses TWO periodic current sheets (to
avoid an open/conducting boundary condition) with a perturbation that is
Gaussian-localized around each sheet rather than spanning the box, so
each sheet is seeded independently rather than coupled through one
domain-wide mode. This is a common, defensible variant for doubly-
periodic Hall-MHD GEM setups -- flagging it here so it's a documented
choice rather than something to rediscover by surprise later.

NUMERICS, not physics (both introduced purely for numerical stability,
disclosed as such):
  nu (viscosity) and NU4 (4th-order hyperviscosity) are regularization,
  not part of the physical benchmark. Defaults nu=0.001, NU4=1e-3 are
  carried over from earlier runs that were empirically stable at
  eta=0.005 -- they have not been independently re-derived from theory
  and should be treated as "known to work at these settings", not "the
  correct value for all parameter choices".

dt SAFETY LIMITS: four candidates are combined via min() every step --
advection, resistive/viscous diffusion, Hall/whistler CFL, and a 4th
limit (dt_hyper) for the explicit-RK4 stability boundary of the NU4
biharmonic term. IMPORTANT CAVEAT: dt_hyper's derivation is a standard
textbook estimate (RK4's stability boundary on the negative real axis,
~2.785, applied to the biharmonic operator's largest eigenvalue) but has
NOT yet been empirically validated against this exact solver -- the
previous attempt to test it was invalidated by a file mix-up (the script
being run wasn't the one being edited). Trust the printed "initial dt
candidates" breakdown at the start of each run over this docstring until
it's actually been checked.

Usage:
    python3 gem_hall_mhd_phase0.py                  # literature defaults
    python3 gem_hall_mhd_phase0.py --eta 0.001 --tag lowres_stress_test
    python3 gem_hall_mhd_phase0.py --help

Tested target: macOS Apple Silicon, `pip install "jax[cpu]" numpy matplotlib`.
"""

import os
import time
import argparse
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="GEM double-current-sheet Hall-MHD, Phase 0 (clean build)")
    p.add_argument("--nx", type=int, default=128)
    p.add_argument("--ny", type=int, default=128)
    p.add_argument("--lx", type=float, default=25.6)
    p.add_argument("--ly", type=float, default=25.6)
    p.add_argument("--eta", type=float, default=0.005,
                   help="resistivity; 0.005 is the literature-confirmed "
                        "GEM Hall-MHD value (see module docstring)")
    p.add_argument("--nu", type=float, default=0.001, help="kinematic viscosity (numerical)")
    p.add_argument("--nu4", type=float, default=1e-3, help="4th-order hyperviscosity (numerical)")
    p.add_argument("--di-hall", type=float, default=1.0,
                   help="Hall/ion-inertial-length parameter, used only in the "
                        "Hall-MHD run; the resistive-only comparison run "
                        "always uses d_i=0 regardless of this flag")
    p.add_argument("--t-final", type=float, default=40.0)
    p.add_argument("--n-diag", type=int, default=200)
    p.add_argument("--psi0", type=float, default=0.1)
    p.add_argument("--safety-base", type=float, default=0.3)
    p.add_argument("--safety-hall", type=float, default=0.15)
    p.add_argument("--safety-hyper", type=float, default=0.8,
                   help="fraction of the (unvalidated, see docstring) "
                        "explicit-RK4 stability estimate to use for NU4")
    p.add_argument("--blowup-factor", type=float, default=20.0)
    p.add_argument("--energy-factor", type=float, default=5.0)
    p.add_argument("--tag", type=str, default="",
                   help="suffix appended to the output directory name")
    return p.parse_args()


ARGS = parse_args()

RUN_STAMP = time.strftime("%Y%m%d_%H%M%S")
tag_suffix = f"_{ARGS.tag}" if ARGS.tag else ""
OUTDIR = f"phase0_output_{RUN_STAMP}{tag_suffix}"
os.makedirs(OUTDIR, exist_ok=True)

print(f"Script: {os.path.abspath(__file__)}")
print(f"Output directory: {os.path.abspath(OUTDIR)}")
print(f"Params: nx={ARGS.nx} ny={ARGS.ny} eta={ARGS.eta} nu={ARGS.nu} nu4={ARGS.nu4} "
      f"di_hall={ARGS.di_hall} t_final={ARGS.t_final} psi0={ARGS.psi0} "
      f"safety=({ARGS.safety_base},{ARGS.safety_hall},{ARGS.safety_hyper})")

# ----------------------------------------------------------------------
# Grid
# ----------------------------------------------------------------------
Nx, Ny = ARGS.nx, ARGS.ny
Lx, Ly = ARGS.lx, ARGS.ly
dx, dy = Lx / Nx, Ly / Ny

x = jnp.linspace(0.0, Lx, Nx, endpoint=False)
y = jnp.linspace(-Ly / 2, Ly / 2, Ny, endpoint=False)
X, Y = jnp.meshgrid(x, y, indexing="ij")

kx = 2 * jnp.pi * jnp.fft.fftfreq(Nx, d=dx)
ky = 2 * jnp.pi * jnp.fft.fftfreq(Ny, d=dy)
KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
K2 = KX ** 2 + KY ** 2
K2safe = jnp.where(K2 == 0.0, 1.0, K2)
K4 = K2 ** 2
K4_MAX = float(jnp.max(K4))

kx_max = jnp.max(jnp.abs(kx))
ky_max = jnp.max(jnp.abs(ky))
DEALIAS = ((jnp.abs(KX) < (2.0 / 3.0) * kx_max) &
           (jnp.abs(KY) < (2.0 / 3.0) * ky_max)).astype(jnp.float64)

NU4 = ARGS.nu4
RHO_FLOOR = 0.05
RK4_STABILITY_REAL_AXIS = 2.785  # theoretical estimate -- see docstring caveat


def ddx(f):
    return jnp.real(jnp.fft.ifft2(1j * KX * jnp.fft.fft2(f)))


def ddy(f):
    return jnp.real(jnp.fft.ifft2(1j * KY * jnp.fft.fft2(f)))


def laplacian(f):
    return jnp.real(jnp.fft.ifft2(-K2 * jnp.fft.fft2(f)))


def hyper4(f):
    """+grad^4 f. Used downstream as '- NU4 * hyper4(f)', a damping term."""
    return jnp.real(jnp.fft.ifft2(K4 * jnp.fft.fft2(f)))


def dealias(f):
    return jnp.real(jnp.fft.ifft2(DEALIAS * jnp.fft.fft2(f)))


def reconstruct_Az(Bx, By):
    Jz = ddx(By) - ddy(Bx)
    Jz_hat = jnp.fft.fft2(Jz)
    Az_hat = Jz_hat / K2safe
    Az_hat = Az_hat.at[0, 0].set(0.0)
    return jnp.real(jnp.fft.ifft2(Az_hat))


def run_self_tests(tol=1e-6):
    print("Running self-tests on spectral operators...")
    n1x, n1y = 2, 3
    k1x = 2 * jnp.pi * n1x / Lx
    k1y = 2 * jnp.pi * n1y / Ly
    f_test = jnp.sin(k1x * X) * jnp.cos(k1y * Y)

    ddx_analytic = k1x * jnp.cos(k1x * X) * jnp.cos(k1y * Y)
    err = float(jnp.max(jnp.abs(ddx(f_test) - ddx_analytic)))
    assert err < tol, f"ddx self-test FAILED, max error={err}"

    ddy_analytic = -k1y * jnp.sin(k1x * X) * jnp.sin(k1y * Y)
    err = float(jnp.max(jnp.abs(ddy(f_test) - ddy_analytic)))
    assert err < tol, f"ddy self-test FAILED, max error={err}"

    lap_analytic = -(k1x ** 2 + k1y ** 2) * f_test
    err = float(jnp.max(jnp.abs(laplacian(f_test) - lap_analytic)))
    assert err < tol, f"laplacian self-test FAILED, max error={err}"

    hyper_analytic = (k1x ** 2 + k1y ** 2) ** 2 * f_test
    err = float(jnp.max(jnp.abs(hyper4(f_test) - hyper_analytic)))
    assert err < tol, f"hyper4 self-test FAILED (wrong sign?), max error={err}"

    Az0 = jnp.sin(k1x * X) * jnp.sin(k1y * Y)
    Bx_known = ddy(Az0)
    By_known = -ddx(Az0)
    Az_recovered = reconstruct_Az(Bx_known, By_known)
    err = float(jnp.max(jnp.abs(Az_recovered - Az0)))
    assert err < tol, f"reconstruct_Az self-test FAILED, max error={err}"

    print(f"All self-tests passed (max error {err:.2e} < tol {tol}).")


# ----------------------------------------------------------------------
# Initial condition -- see module docstring for literature sourcing and
# the documented Gaussian-localization deviation
# ----------------------------------------------------------------------
rho0 = 1.0
n_inf = 0.2
B0 = 1.0
lam = 0.5
y1, y2 = -Ly / 4.0, Ly / 4.0
cs2 = (B0 ** 2) / (2.0 * rho0)

psi0 = ARGS.psi0
pert_w = 1.0


def initial_state():
    rho = (rho0 / jnp.cosh((Y - y1) / lam) ** 2 +
           rho0 / jnp.cosh((Y - y2) / lam) ** 2 + n_inf)

    Bx_eq = B0 * (jnp.tanh((Y - y1) / lam) - jnp.tanh((Y - y2) / lam) - 1.0)

    dAz = psi0 * jnp.cos(2 * jnp.pi * X / Lx) * (
        jnp.exp(-((Y - y1) / pert_w) ** 2) -
        jnp.exp(-((Y - y2) / pert_w) ** 2)
    )
    dBx = ddy(dAz)
    dBy = -ddx(dAz)

    Bx = Bx_eq + dBx
    By = dBy
    Bz = jnp.zeros_like(X)

    vx = jnp.zeros_like(X)
    vy = jnp.zeros_like(X)
    vz = jnp.zeros_like(X)

    return (rho, vx, vy, vz, Bx, By, Bz)


# ----------------------------------------------------------------------
# RHS: continuity, momentum (J x B - grad p, + viscosity), generalized
# Ohm's law induction equation with resistive + Hall terms
# ----------------------------------------------------------------------
def rhs(state, eta, di, nu):
    rho, vx, vy, vz, Bx, By, Bz = state

    rho_d = dealias(rho)
    vx_d, vy_d, vz_d = dealias(vx), dealias(vy), dealias(vz)
    Bx_d, By_d, Bz_d = dealias(Bx), dealias(By), dealias(Bz)

    Jx = ddy(Bz_d)
    Jy = -ddx(Bz_d)
    Jz = ddx(By_d) - ddy(Bx_d)

    JxB_x = Jy * Bz_d - Jz * By_d
    JxB_y = Jz * Bx_d - Jx * Bz_d
    JxB_z = Jx * By_d - Jy * Bx_d

    p = cs2 * rho_d
    dpdx, dpdy = ddx(p), ddy(p)

    advx = vx_d * ddx(vx_d) + vy_d * ddy(vx_d)
    advy = vx_d * ddx(vy_d) + vy_d * ddy(vy_d)
    advz = vx_d * ddx(vz_d) + vy_d * ddy(vz_d)

    dvxdt = (JxB_x - dpdx) / rho_d - advx + nu * laplacian(vx) - NU4 * hyper4(vx)
    dvydt = (JxB_y - dpdy) / rho_d - advy + nu * laplacian(vy) - NU4 * hyper4(vy)
    dvzdt = (JxB_z) / rho_d - advz + nu * laplacian(vz) - NU4 * hyper4(vz)

    drhodt = -(ddx(rho_d * vx_d) + ddy(rho_d * vy_d)) - NU4 * hyper4(rho)

    vxB_x = vy_d * Bz_d - vz_d * By_d
    vxB_y = vz_d * Bx_d - vx_d * Bz_d
    vxB_z = vx_d * By_d - vy_d * Bx_d

    Ex = -vxB_x + eta * Jx + (di / rho_d) * JxB_x
    Ey = -vxB_y + eta * Jy + (di / rho_d) * JxB_y
    Ez = -vxB_z + eta * Jz + (di / rho_d) * JxB_z

    dBxdt = -ddy(Ez) - NU4 * hyper4(Bx)
    dBydt = ddx(Ez) - NU4 * hyper4(By)
    dBzdt = ddy(Ex) - ddx(Ey) - NU4 * hyper4(Bz)

    return (drhodt, dvxdt, dvydt, dvzdt, dBxdt, dBydt, dBzdt)


def add_state(a, b, scale):
    return tuple(ai + scale * bi for ai, bi in zip(a, b))


def make_stepper(eta, di, nu):
    def rk4_step(state, dt):
        k1 = rhs(state, eta, di, nu)
        s1 = add_state(state, k1, dt / 2)
        k2 = rhs(s1, eta, di, nu)
        s2 = add_state(state, k2, dt / 2)
        k3 = rhs(s2, eta, di, nu)
        s3 = add_state(state, k3, dt)
        k4 = rhs(s3, eta, di, nu)
        return tuple(
            si + (dt / 6.0) * (k1i + 2 * k2i + 2 * k3i + k4i)
            for si, k1i, k2i, k3i, k4i in zip(state, k1, k2, k3, k4)
        )
    return jax.jit(rk4_step)


def estimate_dt(state, di, eta, nu, safety_base, safety_hall, safety_hyper):
    rho, vx, vy, vz, Bx, By, Bz = state
    Jx = ddy(Bz)
    Jy = -ddx(Bz)
    Jz = ddx(By) - ddy(Bx)
    Jmag = jnp.sqrt(Jx ** 2 + Jy ** 2 + Jz ** 2)
    Bmag = jnp.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)

    Bmax = float(jnp.max(Bmag))
    Jmax = float(jnp.max(Jmag))
    rho_min = float(jnp.min(rho))
    vmax = float(jnp.max(jnp.sqrt(vx ** 2 + vy ** 2 + vz ** 2)))
    cs = float(jnp.sqrt(cs2))

    if (not np.isfinite(Bmax) or not np.isfinite(Jmax) or
            not np.isfinite(rho_min) or not np.isfinite(vmax) or rho_min <= 0.0):
        return float("nan")

    vA_max = Bmax / jnp.sqrt(rho_min)
    dt_adv = safety_base * min(dx, dy) / (vA_max + cs + vmax + 1e-8)
    dt_diff = safety_base * min(dx, dy) ** 2 / (4.0 * (eta + nu) + 1e-8)

    if di > 0:
        B_eff = Bmax + dx * Jmax
        dt_hall = safety_hall * min(dx, dy) ** 2 / (di * B_eff * jnp.pi + 1e-8)
    else:
        dt_hall = jnp.inf

    if NU4 > 0:
        dt_hyper = safety_hyper * RK4_STABILITY_REAL_AXIS / (NU4 * K4_MAX + 1e-30)
    else:
        dt_hyper = jnp.inf

    return float(min(dt_adv, dt_diff, dt_hall, dt_hyper))


def check_health(state, B0_ref, blowup_factor, energy_factor, E0_ref=None):
    rho, vx, vy, vz, Bx, By, Bz = state
    names = ["rho", "vx", "vy", "vz", "Bx", "By", "Bz"]
    for name, arr in zip(names, state):
        if not bool(jnp.all(jnp.isfinite(arr))):
            return False, f"non-finite values in {name}"

    rho_min_now = float(jnp.min(rho))
    if rho_min_now <= 0.0:
        return False, f"density went non-positive (min={rho_min_now:.4g})"

    Bmax_now = float(jnp.max(jnp.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)))
    if Bmax_now > blowup_factor * B0_ref:
        return False, f"|B|_max={Bmax_now:.4g} exceeded {blowup_factor}x initial ({B0_ref:.4g})"

    if E0_ref is not None:
        emag_now = 0.5 * float(jnp.mean(Bx ** 2 + By ** 2 + Bz ** 2))
        if emag_now > energy_factor * E0_ref:
            return False, f"magnetic energy {emag_now:.4g} exceeded {energy_factor}x initial ({E0_ref:.4g})"

    return True, "healthy"


# ----------------------------------------------------------------------
# Run one configuration
# ----------------------------------------------------------------------
def run(label, di, eta, nu, t_final, n_diag,
        safety_base, safety_hall, safety_hyper,
        blowup_factor, energy_factor, print_every_frac=0.05):
    state = initial_state()
    step = make_stepper(eta, di, nu)

    rho0_, vx0, vy0, vz0, Bx0, By0, Bz0 = state
    B0_ref = float(jnp.max(jnp.sqrt(Bx0 ** 2 + By0 ** 2 + Bz0 ** 2)))
    E0_ref = 0.5 * float(jnp.mean(Bx0 ** 2 + By0 ** 2 + Bz0 ** 2))

    dt0 = estimate_dt(state, di, eta, nu, safety_base, safety_hall, safety_hyper)
    dt0_adv_only = estimate_dt(state, 0.0, eta, nu, safety_base, safety_hall, safety_hyper)
    print(f"[{label}] initial dt: {dt0:.6g} (without Hall/whistler term: {dt0_adv_only:.6g})")

    t = 0.0
    next_diag_t = 0.0
    diag_dt = t_final / n_diag
    next_print_t = 0.0
    print_dt = t_final * print_every_frac

    t_hist, az1_hist, az2_hist, emag_hist, ekin_hist, jzpeak_hist = [], [], [], [], [], []
    az1_xline_hist, az2_xline_hist = [], []  # X-line-tracking version, see below
    xpos1_hist, xpos2_hist = [], []  # where the tracked X-line actually is, in x
    # Direct reconnection-rate diagnostic: the out-of-plane electric field
    # Ez evaluated AT the tracked X-line, from the same generalized Ohm's
    # law used in the induction equation. By Faraday's law this Ez *is*
    # d(Psi)/dt at that point (curl E = -dB/dt integrated appropriately),
    # so it is a direct, non-differentiated measurement of the reconnection
    # rate -- the standard quantity reported in the GEM Challenge papers --
    # rather than something inferred by finite-differencing a flux history.
    # It serves as an independent cross-check against az*_xline_hist above.
    ez_xline1_hist, ez_xline2_hist = [], []
    # Continuity state for the X-line tracker: None until first snapshot,
    # then the previous ix for each sheet, so subsequent snapshots search a
    # neighborhood around where the tracker WAS rather than the whole band
    # fresh each time. Without this, a transient secondary current
    # concentration (plasmoid, noise) anywhere in the band can make the
    # tracker snap to sampling Az at a completely different, unrelated point
    # -- producing spurious discontinuous jumps in the "flux" time series
    # rather than following one continuously-evolving physical X-line.
    prev_ix = {1: None, 2: None}
    fallback_count = {1: 0, 2: 0}  # how often no saddle point was found in
    # the search window and the tracker had to fall back to the old
    # max-|Jz| heuristic -- see the null-point tracking comment below
    multi_saddle_count = {1: 0, 2: 0}  # CAVEAT, found after the first patch:
    # this counts grid CELLS satisfying D<0 in-window, not distinct
    # topological X-points. D<0 holds on a whole extended neighborhood
    # around any single saddle (it's a continuous field, not a delta
    # function), so this will read ~200/200 even with exactly one X-line
    # present -- it is NOT, by itself, evidence of a plasmoid chain. Kept
    # here (relabeled below) only as a coarse sanity print; a real
    # plasmoid-chain census would need connected-component labeling of the
    # saddle mask, which this prototype does not attempt.
    XLINE_SEARCH_HALFWIDTH = max(4, Nx // 8)  # grid cells; generous enough
    # for smooth X-line drift between snapshots, tight enough to reject a
    # jump to an unrelated feature elsewhere in the domain
    # CONTINUITY_FRAC: second-stage fix, still needed even with the saddle
    # (D<0) filter above. That filter rejects O-points/plasmoid centers,
    # but does nothing to stop the tracker jumping between two DIFFERENT
    # X-points that are BOTH genuine saddles and BOTH inside the search
    # window at once -- exactly what a plasmoid CHAIN produces (an
    # alternating X-O-X-O-X sequence has multiple simultaneous saddles).
    # Picking the global min-|B_perp|^2 saddle each frame with no memory
    # of where the tracker just was lets it relabel "the" X-line onto a
    # neighboring one from one diagnostic snapshot to the next -- a smaller
    # discontinuity than jumping to an O-point, but still unphysical for a
    # continuously-tracked flux, and still enough to trip the >8x-median
    # glitch detector in analyze_phase0.py right at the burst. Fix: add a
    # soft continuity penalty, scaled to the local spread of |B_perp|^2
    # among this frame's saddle candidates so it doesn't need hand-tuning
    # per run, that favors the saddle nearest the tracker's own last
    # position unless a competitor is a much cleaner (much smaller
    # |B_perp|^2) null. Set to 0 to recover the previous (unpenalized)
    # behavior for comparison.
    CONTINUITY_FRAC = 2.0
    healthy = True
    stop_reason = "reached t_final"
    t0_wall = time.time()
    n_steps_taken = 0

    while True:
        healthy_now, reason = check_health(state, B0_ref, blowup_factor, energy_factor, E0_ref=E0_ref)
        if not healthy_now:
            healthy = False
            stop_reason = f"blow-up guard tripped at t={t:.3f}: {reason}"
            break

        if t >= t_final:
            stop_reason = "reached t_final"
            break

        if t >= next_diag_t:
            rho, vx, vy, vz, Bx, By, Bz = state
            Az = reconstruct_Az(Bx, By)
            Jz_now = ddx(By) - ddy(Bx)
            ix_center = Nx // 2
            iy1 = int((y1 + Ly / 2) / dy)
            iy2 = int((y2 + Ly / 2) / dy)
            az1_hist.append(float(Az[ix_center, iy1]))
            az2_hist.append(float(Az[ix_center, iy2]))

            # X-line-tracking version: instead of trusting the ORIGINAL grid
            # point to still be where the X-line is, search near wherever the
            # tracker found the X-line LAST time (a periodic window in x, full
            # y_band around the sheet).
            #
            # CRITERION, FIXED: this used to be "wherever |Jz| is currently
            # largest" in the window. That is only a good proxy for the
            # X-line before secondary islands (plasmoids) form. Once they
            # do -- expected in Hall-mediated reconnection, and visible here
            # as growing noise in the peak-|Jz| trace once t gets past the
            # linear phase -- an O-point (island center) or a competing
            # secondary current layer can transiently carry MORE current
            # than the actual X-line, so max-|Jz| tracking jumps onto it.
            # That produced exactly the pathology seen in the t_final=80 run:
            # a near-instantaneous multi-unit drop in the tracked flux, i.e.
            # Az teleporting -- impossible for a continuous field, so it was
            # a tracking artifact, not reconnection running backwards.
            #
            # Fix: an X-line is a magnetic null with SADDLE topology, not
            # just a strong-current point. In 2D, B is divergence-free, so
            # the Jacobian of (Bx,By) is trace-free and its eigenvalues are
            # either real-and-opposite-sign (saddle = X-point) or purely
            # imaginary (center = O-point), decided by the sign of
            # det(J) = dBx/dx*dBy/dy - dBx/dy*dBy/dx: X-point iff det(J)<0.
            # So within the search window we now look for the point with
            # smallest |B|_perp^2 (closest to a true null) AMONG points that
            # satisfy the saddle condition, which structurally excludes
            # O-points/plasmoids. If no saddle point exists in the window
            # (can happen briefly, e.g. right at reconnection onset before
            # islands are resolved), we fall back to the old max-|Jz|
            # heuristic for that one snapshot and count it, so you can see
            # from the printed summary how often that happened.
            dBxdx_now, dBxdy_now = ddx(Bx), ddy(Bx)
            dBydx_now, dBydy_now = ddx(By), ddy(By)
            D_field = dBxdx_now * dBydy_now - dBxdy_now * dBydx_now
            Bperp2_field = Bx ** 2 + By ** 2

            # Full-field Ez, same generalized Ohm's law as in rhs(), evaluated
            # on the raw (non-dealiased) state to stay consistent with how
            # Jz_now/Az are already computed above for this diagnostic block.
            Jx_now = ddy(Bz)
            Jy_now = -ddx(Bz)
            vxB_z_now = vx * By - vy * Bx
            JxB_z_now = Jx_now * By - Jy_now * Bx
            Ez_field = -vxB_z_now + eta * Jz_now + (di / rho) * JxB_z_now
            # Gradients of Ez, needed below for the sub-grid null refinement:
            # a first-order Taylor step off the nearest GRID point onto the
            # actual sub-grid location where (Bx,By)=(0,0), so Ez is sampled
            # at the true null rather than at whichever pixel happened to be
            # closest to it (see comment at the refinement step below).
            Ez_dx_field, Ez_dy_field = ddx(Ez_field), ddy(Ez_field)

            y_band = 2.0 * lam
            iy_band = max(1, int(y_band / dy))
            for sheet_id, y_sheet, xpos_hist, az_xline_hist, ez_xline_hist in [
                (1, y1, xpos1_hist, az1_xline_hist, ez_xline1_hist),
                (2, y2, xpos2_hist, az2_xline_hist, ez_xline2_hist),
            ]:
                iy_sheet = int((y_sheet + Ly / 2) / dy)
                iy_lo = max(0, iy_sheet - iy_band)
                iy_hi = min(Ny, iy_sheet + iy_band + 1)

                if prev_ix[sheet_id] is None:
                    # first snapshot: no prior location, search the full row
                    ix_candidates = jnp.arange(Nx)
                else:
                    # periodic window centered on the previous location
                    offsets = jnp.arange(-XLINE_SEARCH_HALFWIDTH, XLINE_SEARCH_HALFWIDTH + 1)
                    ix_candidates = (prev_ix[sheet_id] + offsets) % Nx

                D_band = D_field[ix_candidates][:, iy_lo:iy_hi]
                Bperp2_band = Bperp2_field[ix_candidates][:, iy_lo:iy_hi]
                saddle_mask = D_band < 0.0

                if int(jnp.sum(saddle_mask)) > 1:
                    multi_saddle_count[sheet_id] += 1

                if bool(jnp.any(saddle_mask)):
                    scored = jnp.where(saddle_mask, Bperp2_band, jnp.inf)
                    if prev_ix[sheet_id] is not None and CONTINUITY_FRAC > 0.0:
                        # Distance (in x, periodic) from the tracker's own
                        # last confirmed position, in physical units so it
                        # combines sensibly with |B_perp|^2's units once
                        # scaled below.
                        offsets = jnp.arange(-XLINE_SEARCH_HALFWIDTH,
                                              XLINE_SEARCH_HALFWIDTH + 1)
                        dist2 = (offsets.astype(jnp.float64) * dx) ** 2
                        dist2 = dist2[:, None] * jnp.ones_like(scored)
                        # Self-calibrating scale: penalize a full-window
                        # hop by roughly CONTINUITY_FRAC times the spread
                        # of |B_perp|^2 actually seen among this frame's
                        # saddle candidates, rather than a fixed constant
                        # that would need re-tuning for every psi0/eta/t_final.
                        cand_vals = scored[jnp.isfinite(scored)]
                        b_scale = float(jnp.std(cand_vals)) + 1e-12
                        max_d2 = (XLINE_SEARCH_HALFWIDTH * dx) ** 2
                        penalty = CONTINUITY_FRAC * b_scale / max_d2 * dist2
                        scored = scored + penalty
                    flat_idx = int(jnp.argmin(scored))
                else:
                    # no saddle in the window this snapshot -- fall back,
                    # and count it (printed in the run summary)
                    Jz_band = jnp.abs(Jz_now[ix_candidates][:, iy_lo:iy_hi])
                    flat_idx = int(jnp.argmax(Jz_band))
                    fallback_count[sheet_id] += 1

                ix_local, iy_x_local = jnp.unravel_index(flat_idx, Bperp2_band.shape)
                ix_local, iy_x_local = int(ix_local), int(iy_x_local)
                ix_x = int(ix_candidates[ix_local])
                iy_x = iy_lo + iy_x_local

                prev_ix[sheet_id] = ix_x
                xpos_hist.append(float(x[ix_x]))

                # --- Sub-grid null refinement (one Newton/Taylor step) ---
                # The chosen (ix_x, iy_x) is the nearest GRID POINT to the
                # X-line, not the X-line itself: (Bx,By) generically is NOT
                # exactly (0,0) there, it's just the local minimum of
                # |B_perp|^2 on the grid. That residual matters a lot for
                # the DIRECT Ez diagnostic: at the true null the ideal term
                # -(v x B)_z vanishes identically (both v x B and the Hall
                # term J x B are proportional to Bx or By), leaving
                # Ez(null) = eta*Jz(null) alone -- the standard 2D
                # reconnection-rate relation. Off-null by even one grid
                # spacing, that residual ideal-MHD term does NOT vanish and,
                # at small eta, can be larger than the true resistive
                # signal, corrupting the "direct" rate estimate even though
                # the TRACKED POSITION itself (used for the flux/gamma
                # diagnostics) is perfectly fine.
                #
                # Fix: take one Newton step using the local Jacobian of
                # (Bx,By) -- already computed above as dBxdx_now etc. and
                # D_field = det(Jacobian) -- to solve for the sub-grid
                # offset (dx_off, dy_off) that zeroes (Bx,By) to linear
                # order, then Taylor-expand Az and Ez to that same offset
                # using their own local gradients (dAz/dx=-By, dAz/dy=Bx by
                # the flux-function definition used throughout this file).
                # This is the 2D analogue of standard sub-grid null-finding
                # methods used to locate reconnection X-lines to sub-grid
                # accuracy (cf. Parnell et al. 1996 for 3D null
                # classification; Fu et al. 2015, the "first-order Taylor
                # expansion" X-line method applied to spacecraft data).
                Bx0 = float(Bx[ix_x, iy_x])
                By0 = float(By[ix_x, iy_x])
                a11 = float(dBxdx_now[ix_x, iy_x]); a12 = float(dBxdy_now[ix_x, iy_x])
                a21 = float(dBydx_now[ix_x, iy_x]); a22 = float(dBydy_now[ix_x, iy_x])
                D0 = a11 * a22 - a12 * a21  # == D_field[ix_x, iy_x], guaranteed
                # nonzero and negative since ix_x,iy_x was chosen to satisfy
                # the saddle condition (or is the max-|Jz| fallback point;
                # guard D0 either way before dividing).
                if abs(D0) > 1e-12:
                    dx_off = (-Bx0 * a22 + By0 * a12) / D0
                    dy_off = (-By0 * a11 + a21 * Bx0) / D0
                    # Clip to a fraction of one cell: a Newton step assumes
                    # local linearity, which can break down if the chosen
                    # point is a poor initial guess (e.g. the max-|Jz|
                    # fallback branch); clipping keeps the correction a
                    # genuine sub-grid refinement rather than a wild
                    # extrapolation.
                    dx_off = float(np.clip(dx_off, -0.75 * dx, 0.75 * dx))
                    dy_off = float(np.clip(dy_off, -0.75 * dy, 0.75 * dy))
                else:
                    dx_off = dy_off = 0.0

                Az0 = float(Az[ix_x, iy_x])
                Ez0 = float(Ez_field[ix_x, iy_x])
                Ez_dx0 = float(Ez_dx_field[ix_x, iy_x])
                Ez_dy0 = float(Ez_dy_field[ix_x, iy_x])

                az_refined = Az0 + Bx0 * dy_off - By0 * dx_off
                ez_refined = Ez0 + Ez_dx0 * dx_off + Ez_dy0 * dy_off

                az_xline_hist.append(az_refined)
                ez_xline_hist.append(ez_refined)

            emag_hist.append(0.5 * float(jnp.mean(Bx ** 2 + By ** 2 + Bz ** 2)))
            ekin_hist.append(0.5 * float(jnp.mean(rho * (vx ** 2 + vy ** 2 + vz ** 2))))
            jzpeak_hist.append(float(jnp.max(jnp.abs(Jz_now))))
            t_hist.append(t)
            next_diag_t += diag_dt

        dt = estimate_dt(state, di, eta, nu, safety_base, safety_hall, safety_hyper)
        if not np.isfinite(dt):
            healthy = False
            stop_reason = f"blow-up guard tripped at t={t:.3f}: dt became non-finite before stepping"
            break

        state = step(state, dt)
        rho_n, vx_n, vy_n, vz_n, Bx_n, By_n, Bz_n = state
        rho_n = jnp.clip(rho_n, RHO_FLOOR, None)
        state = (rho_n, vx_n, vy_n, vz_n, Bx_n, By_n, Bz_n)

        t += dt
        n_steps_taken += 1

        if t >= next_print_t:
            elapsed = time.time() - t0_wall
            frac = max(t / t_final, 1e-6)
            est_total = elapsed / frac
            print(f"[{label}] t={t:.2f}/{t_final} (dt={dt:.5f}, "
                  f"{n_steps_taken} steps so far), elapsed={elapsed:.1f}s, "
                  f"est. total={est_total/60:.1f} min")
            next_print_t += print_dt

    wall = time.time() - t0_wall
    print(f"[{label}] stopped: {stop_reason}. wall={wall:.1f}s, "
          f"steps={n_steps_taken}")
    n_diag_taken = len(t_hist)
    for sheet_id in (1, 2):
        if n_diag_taken > 0 and fallback_count[sheet_id] > 0:
            print(f"[{label}] sheet {sheet_id}: X-line tracker used the "
                  f"max-|Jz| fallback (no saddle point in window) on "
                  f"{fallback_count[sheet_id]}/{n_diag_taken} diagnostic "
                  f"snapshots -- inspect those timestamps if the flux curve "
                  f"still looks glitchy")
        if n_diag_taken > 0:
            print(f"[{label}] sheet {sheet_id}: {multi_saddle_count[sheet_id]}"
                  f"/{n_diag_taken} snapshots had >1 grid cell satisfying "
                  f"D<0 in-window (expected near any single X-line -- see "
                  f"code comment; NOT a plasmoid-chain count on its own). "
                  f"The continuity penalty (CONTINUITY_FRAC={CONTINUITY_FRAC}) "
                  f"still guards against relabeling onto a genuinely "
                  f"different saddle when one does exist.")

    rho, vx, vy, vz, Bx, By, Bz = state
    return {
        "t": np.array(t_hist),
        "az1": np.array(az1_hist),
        "az1_xline": np.array(az1_xline_hist),
        "az2_xline": np.array(az2_xline_hist),
        "ez1_xline": np.array(ez_xline1_hist),
        "ez2_xline": np.array(ez_xline2_hist),
        "xpos1": np.array(xpos1_hist),
        "xpos2": np.array(xpos2_hist),
        "az2": np.array(az2_hist),
        "emag": np.array(emag_hist),
        "ekin": np.array(ekin_hist),
        "jzpeak": np.array(jzpeak_hist),
        "Bz_final": np.asarray(Bz),
        "healthy": healthy,
        "stop_reason": stop_reason,
        "t_stopped": t,
        "n_steps": n_steps_taken,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    run_self_tests()

    common = dict(eta=ARGS.eta, nu=ARGS.nu, t_final=ARGS.t_final, n_diag=ARGS.n_diag,
                  safety_base=ARGS.safety_base, safety_hall=ARGS.safety_hall,
                  safety_hyper=ARGS.safety_hyper, blowup_factor=ARGS.blowup_factor,
                  energy_factor=ARGS.energy_factor)

    resistive = run("resistive-only (d_i=0)", di=0.0, **common)
    hall = run(f"Hall-MHD (d_i={ARGS.di_hall})", di=ARGS.di_hall, **common)

    for label, data in [("resistive", resistive), ("hall", hall)]:
        np.savez(
            os.path.join(OUTDIR, f"{label}_timeseries.npz"),
            t=data["t"], az1=data["az1"], az2=data["az2"],
            emag=data["emag"], ekin=data["ekin"], jzpeak=data["jzpeak"],
            az1_xline=data["az1_xline"], az2_xline=data["az2_xline"],
            ez1_xline=data["ez1_xline"], ez2_xline=data["ez2_xline"],
            xpos1=data["xpos1"], xpos2=data["xpos2"],
        )

    plt.figure(figsize=(7, 5))
    plt.plot(resistive["t"], resistive["az1"] - resistive["az1"][0], label="resistive, sheet 1")
    plt.plot(hall["t"], hall["az1"] - hall["az1"][0], label="Hall, sheet 1")
    plt.xlabel("time")
    plt.ylabel("Az(sheet center) - Az(t=0)")
    plt.title("Reconnected flux proxy: Hall vs resistive-only")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "reconnected_flux.png"), dpi=150)
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].plot(resistive["t"], resistive["emag"], label="resistive")
    axs[0].plot(hall["t"], hall["emag"], label="Hall")
    axs[0].set_title("Magnetic energy density")
    axs[0].legend()
    axs[1].plot(resistive["t"], resistive["ekin"], label="resistive")
    axs[1].plot(hall["t"], hall["ekin"], label="Hall")
    axs[1].set_title("Kinetic energy density")
    axs[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "energy.png"), dpi=150)
    plt.close()

    for label, data in [("hall", hall), ("resistive", resistive)]:
        plt.figure(figsize=(6, 5))
        vmax = np.max(np.abs(data["Bz_final"])) + 1e-12
        plt.pcolormesh(np.array(X), np.array(Y), data["Bz_final"],
                        cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        plt.colorbar(label="Bz")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Bz at t_final ({label} run)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"bz_quadrupole_{label}.png"), dpi=150)
        plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, data, color in [("resistive", resistive, "tab:blue"),
                                ("hall", hall, "tab:orange")]:
        flux = data["az1"] - data["az1"][0]
        mask = (flux > 1e-6) & (data["t"] < 15.0)
        axs[0].semilogy(data["t"][mask], flux[mask], label=label, color=color)
    axs[0].set_xlabel("time")
    axs[0].set_ylabel("Az(sheet) - Az(0)  [log scale]")
    axs[0].set_title("Early-time flux growth (log scale)")
    axs[0].legend()

    for label, data, color in [("resistive", resistive, "tab:blue"),
                                ("hall", hall, "tab:orange")]:
        axs[1].plot(data["t"], data["jzpeak"], label=label, color=color)
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("peak |Jz|")
    axs[1].set_title("Peak current density (reconnection-intensity proxy)")
    axs[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "growth_rate_diagnostics.png"), dpi=150)
    plt.close()

    with open(os.path.join(OUTDIR, "run_log.txt"), "w") as f:
        f.write(f"script: {os.path.abspath(__file__)}\n")
        f.write(f"run_stamp: {RUN_STAMP}  tag: '{ARGS.tag}'\n")
        f.write(f"Nx,Ny = {Nx},{Ny}; Lx,Ly = {Lx},{Ly}\n")
        f.write(f"eta={ARGS.eta}, nu={ARGS.nu}, NU4={NU4}, di_hall={ARGS.di_hall}, "
                f"cs2={cs2}, psi0={psi0}, T_FINAL={ARGS.t_final}\n")
        f.write(f"safety_base={ARGS.safety_base}, safety_hall={ARGS.safety_hall}, "
                f"safety_hyper={ARGS.safety_hyper}, K4_max={K4_MAX:.6g}\n")
        f.write(f"resistive: healthy={resistive['healthy']}, "
                f"stop_reason='{resistive['stop_reason']}', "
                f"t_stopped={resistive['t_stopped']:.3f}, steps={resistive['n_steps']}\n")
        f.write(f"hall: healthy={hall['healthy']}, "
                f"stop_reason='{hall['stop_reason']}', "
                f"t_stopped={hall['t_stopped']:.3f}, steps={hall['n_steps']}\n")
        f.write(f"resistive max|Bz| at stop: {np.max(np.abs(resistive['Bz_final']))}\n")
        f.write(f"hall max|Bz| at stop: {np.max(np.abs(hall['Bz_final']))}\n")

    print(f"Done. See {os.path.abspath(OUTDIR)}/")
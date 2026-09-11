import numpy as np
from scipy import constants
from scipy.ndimage import center_of_mass, shift as ndi_shift
from scipy.optimize import least_squares


def energy2wavelength(kV):
    """
    Relativistic electron wavelength.

    Parameters
    ----------
    kV : float
        Beam energy in kilovolts (electron acceleration voltage).

    Returns
    -------
    wavelength : float
        Electron wavelength in Angstrom.
    """
    V = kV * 1e3  # volts
    E = constants.e * V  # kinetic energy, Joules
    m0c2 = constants.m_e * constants.c**2
    wavelength_m = constants.h / np.sqrt(2 * constants.m_e * E * (1 + E / (2 * m0c2)))
    return wavelength_m * 1e10  # meters -> Angstrom


def SimProbe(dr, shape, wavelength, aberration_coefficients, semiangle_cutoff):
    """
    Simulate a STEM probe wave function from the contrast transfer function.

    Parameters
    ----------
    dr : float or (float, float)
        Real-space pixel size (sampling) in Angstrom. A single float is used
        for both x and y; a tuple gives (dx, dy).
    shape : (int, int)
        Number of grid points (Nx, Ny) of the simulation.
    wavelength : float
        Electron wavelength in Angstrom.
    aberration_coefficients : dict
        Aberration coefficients up to 3rd order (C30). Recognized keys
        (all default to 0 if omitted):
            'C10'                 defocus [Angstrom]
            'C12', 'phi12'        2-fold astigmatism [Angstrom], angle [rad]
            'C21', 'phi21'        coma [Angstrom], angle [rad]
            'C23', 'phi23'        trefoil [Angstrom], angle [rad]
            'C30'                 spherical aberration Cs [Angstrom]
    semiangle_cutoff : float
        Objective aperture semiangle cutoff, in milliradians (mrad).

    Returns
    -------
    probe : ndarray, complex128, shape = `shape`
        Real-space probe wave function, fftshifted so the probe is
        centered in the array. Normalized to unit total intensity
        (sum(|probe|**2) == 1).
    """
    Nx, Ny = shape

    if np.isscalar(dr):
        dx = dy = float(dr)
    else:
        dx, dy = dr

    # reciprocal-space grid (1/Angstrom)
    kx = np.fft.fftfreq(Nx, d=dx)
    ky = np.fft.fftfreq(Ny, d=dy)
    kx, ky = np.meshgrid(kx, ky, indexing="ij")

    k = np.sqrt(kx**2 + ky**2)
    phi = np.arctan2(ky, kx)

    # scattering semiangle (radians), small-angle approx: alpha ~ k*lambda
    alpha = k * wavelength

    # --- hard aperture ---
    cutoff = semiangle_cutoff * 1e-3  # mrad -> rad
    aperture = (alpha <= cutoff).astype(np.float64)

    # --- aberration coefficients (defaults to 0) ---
    c = aberration_coefficients
    C10 = c.get("C10", 0.0)
    C12, phi12 = c.get("C12", 0.0), c.get("phi12", 0.0)
    C21, phi21 = c.get("C21", 0.0), c.get("phi21", 0.0)
    C23, phi23 = c.get("C23", 0.0), c.get("phi23", 0.0)
    C30 = c.get("C30", 0.0)

    # --- aberration function chi(alpha, phi), up to C30 ---
    chi = (
        0.5 * C10 * alpha**2
        + 0.5 * C12 * alpha**2 * np.cos(2 * (phi - phi12))
        + (1.0 / 3.0) * C21 * alpha**3 * np.cos(1 * (phi - phi21))
        + (1.0 / 3.0) * C23 * alpha**3 * np.cos(3 * (phi - phi23))
        + 0.25 * C30 * alpha**4
    )
    chi *= 2 * np.pi / wavelength

    # --- complex CTF and real-space probe ---
    ctf = aperture * np.exp(-1j * chi)

    probe = np.fft.ifft2(ctf)
    probe = np.fft.fftshift(probe)

    norm = np.sqrt(np.sum(np.abs(probe) ** 2))
    if norm > 0:
        probe = probe / norm

    return probe


# --- internal Cartesian parametrization for fitting -------------------------
#
# The (C_m, phi_m) polar form of each non-isotropic aberration term is
# awkward to fit directly with a gradient-based optimizer because phi_m is
# periodic and the term is degenerate at C_m = 0 (phi_m undefined). Instead
# the optimizer works on the equivalent Cartesian components
#
#     C_m * cos(m*(phi - phi_m)) = a_m * cos(m*phi) + b_m * sin(m*phi)
#
# with a_m = C_m*cos(m*phi_m), b_m = C_m*sin(m*phi_m), which are smooth,
# unconstrained, and free of the phi_m periodicity/degeneracy. C10 and C30
# (m=0, isotropic) need no such treatment.
#
# Fit vector: x = [C10, C30, a12, b12, a21, b21, a23, b23]

_PARAM_NAMES = ["C10", "C30", "a12", "b12", "a21", "b21", "a23", "b23"]


def _coeffs_to_x(coeffs):
    """Physical (C, phi) dict -> Cartesian fit vector."""
    C10 = coeffs.get("C10", 0.0)
    C30 = coeffs.get("C30", 0.0)
    C12, phi12 = coeffs.get("C12", 0.0), coeffs.get("phi12", 0.0)
    C21, phi21 = coeffs.get("C21", 0.0), coeffs.get("phi21", 0.0)
    C23, phi23 = coeffs.get("C23", 0.0), coeffs.get("phi23", 0.0)
    return np.array(
        [
            C10,
            C30,
            C12 * np.cos(2 * phi12),
            C12 * np.sin(2 * phi12),
            C21 * np.cos(phi21),
            C21 * np.sin(phi21),
            C23 * np.cos(3 * phi23),
            C23 * np.sin(3 * phi23),
        ]
    )


def _x_to_coeffs(x):
    """Cartesian fit vector -> physical (C, phi) dict, for SimProbe / reporting."""
    C10, C30, a12, b12, a21, b21, a23, b23 = x
    C12 = np.hypot(a12, b12)
    C21 = np.hypot(a21, b21)
    C23 = np.hypot(a23, b23)
    phi12 = (np.arctan2(b12, a12) / 2) % np.pi
    phi21 = np.arctan2(b21, a21) % (2 * np.pi)
    phi23 = (np.arctan2(b23, a23) / 3) % (2 * np.pi / 3)
    return {
        "C10": C10,
        "C12": C12,
        "phi12": phi12,
        "C21": C21,
        "phi21": phi21,
        "C23": C23,
        "phi23": phi23,
        "C30": C30,
    }


def _recenter_intensity(intensity):
    """Shift an intensity image so its centroid sits on the array center."""
    shape = intensity.shape
    center = np.array([(n - 1) / 2.0 for n in shape])
    com = np.array(center_of_mass(intensity))
    return ndi_shift(intensity, center - com, mode="constant", cval=0.0)


def _natural_scales(wavelength, semiangle_cutoff):
    """
    Characteristic magnitude of each Cartesian fit parameter: the
    coefficient value that gives ~1 radian of aberration phase at the
    aperture edge, i.e. C_scale(n) = (n+1) * lambda / alpha_max**(n+1).

    Used both as `x_scale` (trust-region scaling) and as the finite-
    difference step base for `least_squares`. This matters a lot here:
    at the perfectly-round (unaberrated) probe, the probe INTENSITY is
    stationary to first order in every aberration coefficient (adding a
    small aberration only perturbs the phase of the wave function), so
    the naive default step (relative to the current parameter value,
    which is degenerate at x=0) sees a numerically zero gradient and the
    optimizer never moves. Stepping by an amount comparable to one
    radian of phase avoids that degenerate point.
    """
    alpha_max = semiangle_cutoff * 1e-3  # mrad -> rad
    scale1 = 2 * wavelength / alpha_max**2       # n=1: C10, C12
    scale2 = 3 * wavelength / alpha_max**3       # n=2: C21, C23
    scale3 = 4 * wavelength / alpha_max**4       # n=3: C30
    # order matches _PARAM_NAMES: C10, C30, a12, b12, a21, b21, a23, b23
    return np.array([scale1, scale3, scale1, scale1, scale2, scale2, scale2, scale2])


def FitAberrations(
    ProbeIntensity,
    dr,
    kV,
    semiangle_cutoff,
    initial_coefficients=None,
    recenter=True,
    n_restarts=8,
    seed=0,
    max_nfev=300,
):
    """
    Fit aberration coefficients (up to C30) to a measured STEM probe
    intensity, using SimProbe as the forward model.

    This is a nonlinear least-squares fit of the 8 aberration parameters
    (C10, C12, phi12, C21, phi21, C23, phi23, C30) so that
    |SimProbe(...)|**2 matches the (normalized, optionally recentered)
    measured intensity as closely as possible.

    Caveat: recovering aberrations from a *single* probe intensity image
    is a non-convex, and for some terms non-unique, inverse problem (this
    is the classic reason real CTEM/STEM aberration measurement uses a
    through-focal series or a Zemlin tableau of tilted images rather than
    one image). In particular, an unaberrated probe is exactly stationary
    to first order in every coefficient, so the cost landscape has a flat
    saddle at zero aberration; `n_restarts` random restarts (with a
    physically-scaled jitter) are used to help escape this, but a good
    `initial_coefficients` guess (e.g. from the microscope's own corrector
    readout) will converge faster and more reliably than a cold start.
    Even then, expect coma (C21) to come back the least well-constrained
    of the eight parameters: its effect on a single, on-axis probe
    intensity is weak and easily traded off against the other terms at
    low residual cost, which is exactly why real microscopes measure coma
    from a beam-tilt series (Zemlin tableau) rather than a single image.

    Parameters
    ----------
    ProbeIntensity : ndarray, shape (Nx, Ny)
        Measured probe intensity (e.g. from a Ronchigram/vacuum probe image
        or a reconstructed probe). Any positive overall scale is fine — it
        is renormalized internally.
    dr : float or (float, float)
        Real-space pixel size of `ProbeIntensity`, in Angstrom.
    kV : float
        Beam energy in kilovolts; converted to wavelength via
        `energy2wavelength`.
    semiangle_cutoff : float
        Objective aperture semiangle cutoff, in mrad (assumed known, not
        fitted).
    initial_coefficients : dict, optional
        Starting guess, same keys as `SimProbe`'s `aberration_coefficients`.
        Defaults to a cold start (all zeros) with random restarts.
    recenter : bool, optional
        If True (default), shift `ProbeIntensity` so its centroid matches
        the array center before fitting (SimProbe always returns a
        centered probe, so an off-center measured probe would otherwise
        bias the fit).
    n_restarts : int, optional
        Number of least-squares runs from jittered starting points; the
        lowest-cost result is kept. The first restart always starts
        exactly at `initial_coefficients` (nudged off zero if that is the
        cold-start default).
    seed : int, optional
        Seed for the restart jitter, for reproducibility.
    max_nfev : int, optional
        Maximum number of function evaluations per least-squares run.

    Returns
    -------
    dict with keys:
        'coefficients'      fitted aberration dict (same keys as SimProbe's
                            `aberration_coefficients`)
        'wavelength'        electron wavelength used, in Angstrom
        'fitted_intensity'  simulated probe intensity at the fitted
                            coefficients, normalized to unit sum
        'target_intensity'  the (recentered, normalized) measured intensity
                            actually fit against
        'result'            the best scipy.optimize.OptimizeResult
    """
    shape = ProbeIntensity.shape
    wavelength = energy2wavelength(kV)

    target = np.asarray(ProbeIntensity, dtype=np.float64)
    if recenter:
        target = _recenter_intensity(target)
    target = target / target.sum()

    def residuals(x):
        coeffs = _x_to_coeffs(x)
        probe = SimProbe(dr, shape, wavelength, coeffs, semiangle_cutoff)
        sim_intensity = np.abs(probe) ** 2
        sim_intensity = sim_intensity / sim_intensity.sum()
        return (sim_intensity - target).ravel()

    x_scale = _natural_scales(wavelength, semiangle_cutoff)

    if initial_coefficients is None:
        center = np.zeros(len(_PARAM_NAMES))
        jitter = x_scale
    else:
        center = _coeffs_to_x(initial_coefficients)
        jitter = np.maximum(0.15 * np.abs(center), 0.15 * x_scale)

    rng = np.random.default_rng(seed)
    best = None
    for i in range(max(n_restarts, 1)):
        if i == 0:
            x0 = center if not np.allclose(center, 0) else 0.05 * x_scale
        else:
            x0 = center + rng.uniform(-1.0, 1.0, size=center.shape) * jitter
        result = least_squares(
            residuals, x0, x_scale=x_scale, diff_step=1e-2, max_nfev=max_nfev
        )
        if best is None or result.cost < best.cost:
            best = result

    fitted_coefficients = _x_to_coeffs(best.x)
    fitted_probe = SimProbe(dr, shape, wavelength, fitted_coefficients, semiangle_cutoff)
    fitted_intensity = np.abs(fitted_probe) ** 2
    fitted_intensity = fitted_intensity / fitted_intensity.sum()

    return {
        "coefficients": fitted_coefficients,
        "wavelength": wavelength,
        "fitted_intensity": fitted_intensity,
        "target_intensity": target,
        "result": best,
    }
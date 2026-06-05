import galsim
import numpy as np

# ===========================================================================
# Sellmeier coefficients for the filter substrate (Suprasil 3001)
# Sellmeier coefficients can be found here:
# https://www.heraeus-covantics.com/media/Media/Documents/Products_and_Solutions/OPT/EN/Data_and_Properties_Optics_fused_silica_EN.pdf
# ===========================================================================
# Wavelengths in the Sellmeier formula above are in micrometers nanometres
# since we use nanometers internally, we need to convert the C constants to nm²

_B1, _B2, _B3 = 6.73693289e-01, 4.31173589e-01, 9.05320925e-01
_NM_TO_UM = 1000.0
_C1 = 4.50899296e-03 * _NM_TO_UM**2
_C2 = 1.33349842e-02 * _NM_TO_UM**2
_C3 = 9.92216527e01 * _NM_TO_UM**2


# ===========================================================================
# Roman WFI instrument constants
# ===========================================================================

# Most instrument constants can be found here:
# https://roman-docs.stsci.edu/roman-instruments/the-wide-field-instrument/observing-with-the-wfi/wfi-quick-reference
# The pupil demagnification is calculated from the ratio of the entrance and exit pupil.
# Technically, the pupil is not perfectly circular, and the demagnification can vary accross
# the field of view. For simplicity, we use an average demagnification from all 18 SCAS, 
# one factor for each axis (x and y). This can be easily calculated using STPSF, where you
# can access the demagnification from the WFI pupil header, example:
# wfi = stpsf.roman.WFI()
# wfi.filter = 'F158'
# wfi.detector = 'SCA08'
# with fits.open(wfi.pupil) as hdul:
#    header = hdul[0].header
#    pupil_demagnification_x = header['PUPIL SCALE FACTOR X']
#    pupil_demagnification_y = header['PUPIL SCALE FACTOR Y']
PIXEL_SIZE_MM = 0.010  # detector pixel size, 10 µm
PIX_SCALE_ARCSEC = 0.11  # nominal plate scale, arcsec/pixel
FRATIO = 8  # telescope focal ratio
TEL_DIAM = 2.36  # telescope primary diameter, m
PUPIL_DEMAG_X = 26.8  # pupil demagnification factor along x-axis
PUPIL_DEMAG_Y = 27.5  # pupil demagnification factor along y-axis
PUPIL_DEMAG = 27.15  # pupil demagnification factor average x,y axis

# Derived angular/spatial scale constants
ARCSEC_TO_MM = PIXEL_SIZE_MM / PIX_SCALE_ARCSEC  # mm/arcsec
MM_TO_DEG = 1.0 / (ARCSEC_TO_MM * 3600.0)  # deg/mm

# Filter geometry (all in metres)
# Note on radius of curvature: The code used below assumes the +z direction is
# towards the detector. This is opposite to the convention used in PSFSim and
# in the Roman optical design, where the detector direction is in the -z direction.
# To account for this change, we need to flip the sign of the radius of
# curvature (-1.5m and -1.493m) in the Roman optical design specifications.
FILTER_THICKNESS = 10e-3  # filter thickness
R1 = 1.5000  # radius of curvature of the entrance surface (S1)
R2 = 1.49931453814  # radius of curvature of the exit surface (S2)
PUPIL_TO_S1 = 10e-3  # distance from the exit pupil to the S1 vertex
S2_TO_FPA = 0.673  # distance from the S2 vertex to the focal plane

# Center of each SCA in the FPA coordinate system, millimetres.
# Center at position (2044, 2044) in science frame
SCA_centers = {
    1: (-22.257157, 11.646353),
    2: (-22.231008, -36.256453),
    3: (-22.185227, -79.102885),
    4: (-66.654970, 20.515410),
    5: (-66.677900, -27.488866),
    6: (-66.625565, -70.230582),
    7: (-110.948084, 41.997556),
    8: (-111.036456, -6.248923),
    9: (-111.376836, -48.637204),
    10: (22.183178, 11.649654),
    11: (22.183137, -36.259689),
    12: (22.156910, -79.106113),
    13: (66.587496, 20.518763),
    14: (66.633266, -27.495304),
    15: (66.607034, -70.240271),
    16: (110.893622, 42.007488),
    17: (111.001570, -6.252038),
    18: (111.368061, -48.650099),
}


# ===========================================================================
# Refractive-index model
# ===========================================================================


def sellmeier_dispersion(lam_nm, B1, B2, B3, C1, C2, C3):
    """
    Sellmeier dispersion relation for the refractive index of a glass substrate.
    (https://en.wikipedia.org/wiki/Sellmeier_equation)

    Parameters
    ----------
    lam_nm : float or array_like
        Wavelength(s) in nanometres.
    B1, B2, B3 : float
        Sellmeier B coefficients (dimensionless).
    C1, C2, C3 : float
        Sellmeier C coefficients (nm squared).

    Returns
    -------
    n : ndarray
        Refractive index (dimensionless).
    """
    lam_nm = np.asarray(lam_nm, dtype=float)
    lam2 = lam_nm**2
    return np.sqrt(1.0 + B1 * lam2 / (lam2 - C1) + B2 * lam2 / (lam2 - C2) + B3 * lam2 / (lam2 - C3))


def n_Suprasil3001(lam_nm):
    """
    Refractive index of the Roman filter substrate (Suprasil 3001).
    Coefficients B and C come from Hareus fits of the Sellmeier
    dispersion relation

    Parameters
    ----------
    lam_nm : float or array_like
        Wavelength(s) in nanometres.

    Returns
    -------
    n : ndarray
        Refractive index (dimensionless).
    """
    return sellmeier_dispersion(lam_nm, _B1, _B2, _B3, _C1, _C2, _C3)


# ===========================================================================
# Ray-tracing geometry helpers
# ===========================================================================


def _snell_refract_unit(p, n_hat, n_old, n_new):
    """
    Exact vector Snell refraction in 2-D.

    Parameters
    ----------
    p : array_like, shape (2,) or (..., 2)
        Incident unit direction [pq, pz].
    n_hat : array_like, shape (2,) or (..., 2)
        Unit outward surface normal (pointing away from the center of curvature).
        The formula assumes ``dot(n_hat, p) > 0`` when the ray travels in the
        same general direction as the normal.
    n_old, n_new : float or ndarray
        Refractive indices of the medium before and after the surface.

    Returns
    -------
    p_out : ndarray, shape like p
        Refracted unit direction.
    """
    p = np.asarray(p, dtype=float)
    n_hat = np.asarray(n_hat, dtype=float)

    # Component of the incident direction along the surface normal.
    mu = np.sum(n_hat * p, axis=-1)
    mu_abs = np.abs(mu)

    # Squared sine of the angle of incidence
    sin2_inc = np.maximum(0.0, 1.0 - mu_abs**2)

    eta = np.asarray(n_new, dtype=float) / np.asarray(n_old, dtype=float)
    root = np.sqrt(eta**2 - sin2_inc)

    # Update only the normal component; the tangential component is preserved
    # This is the equivalent of the vector form of Snell's law
    corr = ((root - mu_abs) * np.sign(mu))[..., None]
    p_out = p + corr * n_hat

    # Renormalise
    p_out /= np.linalg.norm(p_out, axis=-1, keepdims=True)
    return p_out


def _sphere_sag(q, R):
    """
    Sag (vertex-relative surface height) of a spherical surface.
    The surface equation is  (q^2 + z^2) / (2R) − z = 0,  with the vertex at z = 0.

    Parameters
    ----------
    q : float or array_like
        Transverse coordinate, metres.
    R : float
        Radius of curvature, metres.  Positive means the center of curvature
        lies on the +z side of the vertex.

    Returns
    -------
    z : ndarray
        Surface height in metres at each q.
    """
    q = np.asarray(q, dtype=float)
    disc = R**2 - q**2
    if np.any(disc < 0):
        raise ValueError("Ray lies outside spherical surface domain.")
    return R - np.sign(R) * np.sqrt(disc)


def _sphere_normal(q, z, R):
    """
    Unit outward surface normal of a spherical surface.

    Parameters
    ----------
    q, z : float or array_like
        Transverse and axial coordinates of the surface point, metres.
    R : float
        Radius of curvature, metres.

    Returns
    -------
    n_hat : ndarray, shape (..., 2)
        Unit outward normal [nq, nz].
    """
    q = np.asarray(q, dtype=float)
    z = np.asarray(z, dtype=float)

    rinv = 1.0 / R
    n = np.stack([rinv * q, rinv * z - 1.0], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n


def _intersect_line_with_sphere(q0, z0, p, R, z_vertex):
    """
    Intersect a ray with a spherical surface whose vertex lies at z = z_vertex.

    Parameters
    ----------
    q0, z0 : float
        Ray origin, metres.
    p : array_like, shape (..., 2)
        Unit ray direction(s) [pq, pz].  Leading dimensions are broadcast.
    R : float
        Radius of curvature of the surface, metres.
    z_vertex : float
        Axial position of the surface vertex, metres.

    Returns
    -------
    q_hit, z_hit : ndarray
        Intersection coordinates, metres.
    L : ndarray
        Path length from the ray origin to the intersection, metres.
    """
    p = np.asarray(p, dtype=float)
    pq = p[..., 0]
    pz = p[..., 1]

    # Work in the surface's local frame (vertex at origin).
    zp0 = z0 - z_vertex

    # Substitute the ray parametrisation q = q0 + L*pq, z' = zp0 + L*pz into
    # the sphere equation to obtain a quadratic in path length L
    a = 0.5 / R * (pq**2 + pz**2)
    b = (1.0 / R) * (q0 * pq + zp0 * pz) - pz
    c = 0.5 / R * (q0**2 + zp0**2) - zp0

    disc = b * b - 4.0 * a * c
    if np.any(disc < 0):
        raise ValueError("No real intersection with spherical surface.")
    sqrt_disc = np.sqrt(disc)

    L1 = (-b + sqrt_disc) / (2.0 * a)
    L2 = (-b - sqrt_disc) / (2.0 * a)

    # Keep the smallest positive root
    pos1 = L1 > 0
    pos2 = L2 > 0
    L = np.where(
        pos1 & pos2,
        np.minimum(L1, L2),
        np.where(pos1, L1, np.where(pos2, L2, np.nan)),
    )

    if np.any(~np.isfinite(L)):
        raise ValueError("No forward intersection found.")

    q_hit = q0 + L * pq
    z_hit = z0 + L * pz
    return q_hit, z_hit, L


def _propagate_to_plane(q0, z0, p, z_plane):
    """
    Propagate a ray to the plane z = z_plane.

    Parameters
    ----------
    q0, z0 : float
        Ray origin, metres.
    p : array_like, shape (..., 2)
        Unit ray direction(s) [pq, pz].
    z_plane : float
        Axial position of the target plane, metres.

    Returns
    -------
    q : ndarray
        Transverse coordinate at the plane, metres.
    z : ndarray
        z_plane (returned for interface consistency).
    L : ndarray
        Path length from the origin to the plane, metres.
    """
    p = np.asarray(p, dtype=float)
    pq = p[..., 0]
    pz = p[..., 1]

    if np.any(np.abs(pz) < 1e-30):
        raise ValueError("Ray is parallel to target plane.")

    L = (z_plane - z0) / pz
    return q0 + L * pq, np.asarray(z_plane, dtype=float), L


# ===========================================================================
# Filter ray-trace
# ===========================================================================


def _trace_chief_ray(theta0, n_glass, t, R1, R2, s_pupil_to_S1, L_S2_to_fpa):
    """
    Trace the chief ray through a meniscus filter and return its FPA landing position.
    The ray starts in vaccum, refracts through two spherical surfaces separated by
    glass of index ``n_glass``, and propagates to the focal-plane array.

    Parameters
    ----------
    theta0 : float
        Chief-ray field angle in the orthogonal plane, radians.
    n_glass : float or ndarray
        Refractive index of the filter glass.
    t : float
        Filter thickness, metres.
    R1, R2 : float
        Radii of curvature of the entrance (S1) and exit (S2) surfaces, metres.
    s_pupil_to_S1 : float
        Axial distance from the exit pupil to the S1 vertex, metres.
    L_S2_to_fpa : float
        Axial distance from the S2 vertex to the focal plane, metres.

    Returns
    -------
    q_fpa : float or ndarray
        Transverse landing position on the FPA, metres.
    """
    # Chief-ray height at S1 (paraxial approximation from field angle and pupil distance).
    q1 = s_pupil_to_S1 * np.tan(theta0)
    z1 = _sphere_sag(q1, R1)

    # Incident chief ray in vaccum.
    p0 = np.array([np.sin(theta0), np.cos(theta0)], dtype=float)

    # Refract at S1 (vaccum -> glass).
    n1_hat = _sphere_normal(q1, z1, R1)
    p1 = _snell_refract_unit(p0, n1_hat, 1.0, n_glass)

    # Propagate through the glass and intersect S2.
    q2, z2, _ = _intersect_line_with_sphere(q1, z1, p1, R2, z_vertex=t)

    # Refract at S2 (glass -> vaccum).
    z2_local = z2 - t
    n2_hat = _sphere_normal(q2, z2_local, R2)
    p2 = _snell_refract_unit(p1, n2_hat, n_glass, 1.0)

    # Propagate to the detector plane.
    z_fpa = t + L_S2_to_fpa
    q_fpa, _, _ = _propagate_to_plane(q2, z2, p2, z_fpa)
    return q_fpa


def _chromatic_lateral_shift(theta0, nfunc, lam, lam_ref, t, R1, R2, s_pupil_to_S1, L_S2_to_fpa):
    """
    Lateral chromatic image shift between two wavelengths.

    Parameters
    ----------
    theta0 : float
        Chief-ray field angle, radians.
    nfunc : callable
        Refractive-index function ``n(lam_nm)``.  Should accept ndarray input.
    lam : float or ndarray
        Wavelength(s) at which to evaluate the shift, nanometres.
    lam_ref : float
        Reference wavelength, nanometres.
    t : float
        Filter thickness, metres.
    R1, R2 : float
        Radii of curvature of S1 and S2, metres.
    s_pupil_to_S1 : float
        Distance from the exit pupil to the S1 vertex, metres.
    L_S2_to_fpa : float
        Distance from the S2 vertex to the focal plane, metres.

    Returns
    -------
    dq : float or ndarray
        Lateral shift  q(lam) − q(lam_ref)  in metres.
    """
    lam = np.asarray(lam, dtype=float)

    n = nfunc(lam)
    n_ref = nfunc(lam_ref)

    kwargs = dict(t=t, R1=R1, R2=R2, s_pupil_to_S1=s_pupil_to_S1, L_S2_to_fpa=L_S2_to_fpa)

    q = _trace_chief_ray(theta0=theta0, n_glass=n, **kwargs)
    q_ref = _trace_chief_ray(theta0=theta0, n_glass=n_ref, **kwargs)

    return q - q_ref


# ===========================================================================
# SCA geometry
# ===========================================================================


def _sca_to_fpa_coords(sca, x_sci, y_sci):
    """
    Convert SCA science-frame pixel coordinates to FPA-frame Cartesian coordinates.
    Center positions are in position (2044, 2044) in science frame.

    Parameters
    ----------
    sca : int
        SCA index.
    x_sci, y_sci : float
        Pixel coordinates in the science frame.

    Returns
    -------
    x_fpa, y_fpa : float
        Position on the FPA in millimetres.
    """
    x_center, y_center = SCA_centers[sca]
    x_fpa = x_center + PIXEL_SIZE_MM * (2044 - x_sci)
    y_fpa = y_center - PIXEL_SIZE_MM * (2044 - y_sci)
    return x_fpa, y_fpa


def getAOI(sca, x_sci, y_sci):
    """
    Angle of incidence (and its x/y components) for a given SCA pixel position.

    Parameters
    ----------
    sca : int
        SCA index.
    x_sci, y_sci : float
        Pixel coordinates in the science frame.

    Returns
    -------
    aoi : float
        Total angle of incidence, degrees.
    aoi_x, aoi_y : float
        x- and y-plane components of the angle of incidence, degrees.
    """
    x_fpa, y_fpa = _sca_to_fpa_coords(sca, x_sci, y_sci)
    aoi = np.hypot(x_fpa, y_fpa) * MM_TO_DEG * PUPIL_DEMAG
    aoi_x = x_fpa * MM_TO_DEG * PUPIL_DEMAG_X
    aoi_y = y_fpa * MM_TO_DEG * PUPIL_DEMAG_Y
    return aoi, aoi_x, aoi_y


# ===========================================================================
# Photon Operator class
# ===========================================================================


class RomanFilterRefraction(galsim.PhotonOp):
    """
    GalSim ``PhotonOp`` that applies the chromatic lateral shift from the Roman WFI filters.

    For each photon in the array, the shift relative to the effective wavelength
    is computed and applied directly to the photon's (x, y) position in pixels.

    Parameters
    ----------
    bandpass : galsim.Bandpass
        The filter throughput.
    n : callable, optional
        Refractive-index function ``n(lam_nm)``.  Defaults to ``n_Suprasil3001``.
    pixel_scale_arcsec : float, optional
        Detector plate scale in arcsec/pixel.  Default: ``PIX_SCALE_ARCSEC``.
    SCA : int, optional
        SCA index.  Default: 1.
    SCA_pos : galsim.PositionD, optional
        Pixel position on the SCA.  Defaults to detector center (2044, 2044).
    """

    # -----------------------------------------------------------------------
    # Constructor and public interface
    # -----------------------------------------------------------------------

    def __init__(self, bandpass, n=n_Suprasil3001, pixel_scale_arcsec=PIX_SCALE_ARCSEC, SCA=1,
                 SCA_pos=galsim.PositionD(2044, 2044)):

        focal_length_m = FRATIO * TEL_DIAM
        pixscale_rad = pixel_scale_arcsec * np.radians(1 / 3600)  # arcsec -> rad
        self.pixel_pitch_um = focal_length_m * pixscale_rad * 1e6  # pixel pitch in micrometer

        self.sca = SCA
        self.sca_pos = SCA_pos
        self.eff_wave_nm = bandpass.effective_wavelength
        self.n = n

    def applyTo(self, photon_array, local_wcs=None, rng=None):
        """
        Apply chromatic lateral shifts to all photons in ``photon_array``.

        Parameters
        ----------
        photon_array : galsim.PhotonArray
            Must have wavelengths allocated (add a ``WavelengthSampler`` first).
        local_wcs, rng
            Unused; present for GalSim interface compatibility.
        """
        if not photon_array.hasAllocatedWavelengths():
            raise RuntimeError("No wavelengths on PhotonArray. Add a WavelengthSampler first.")

        dx_um, dy_um = self._get_lateral_shifts(photon_array.wavelength)

        # convert shifts from um to pixels
        dx_pix = dx_um / self.pixel_pitch_um
        dy_pix = dy_um / self.pixel_pitch_um

        photon_array.x += dx_pix
        photon_array.y += dy_pix

    def _get_lateral_shifts(self, lam1, lam2=None):
        """
        Chromatic lateral image shifts (dx, dy) for this instance's SCA position.

        The shift is defined as the FPA position at ``lam1`` minus the FPA position
        at the reference wavelength ``lam2``.

        Parameters
        ----------
        lam1 : float or ndarray
            Wavelength(s) to compare, nanometres.  May be a scalar or 1-D array.
        lam2 : float, optional
            Reference wavelength, nanometres.  Defaults to ``self.eff_wave_nm``.

        Returns
        -------
        dx, dy : float or ndarray
            Chromatic lateral shifts along x and y in microns.  Scalar if ``lam1``
            was scalar, otherwise an array with the same shape as ``lam1``.
        """
        if lam2 is None:
            lam2 = self.eff_wave_nm

        lam1 = np.asarray(lam1, dtype=float)

        # get angles of incidence based on FPA position
        aoi, aoi_x, aoi_y = getAOI(self.sca, self.sca_pos.x, self.sca_pos.y)

        common = dict(
            nfunc=self.n,
            lam=lam1,
            lam_ref=lam2,
            t=FILTER_THICKNESS,
            R1=R1,
            R2=R2,
            s_pupil_to_S1=PUPIL_TO_S1,
            L_S2_to_fpa=S2_TO_FPA,
        )
        # get x and y shifts
        dq_x = _chromatic_lateral_shift(theta0=np.radians(aoi_x), **common)
        dq_y = _chromatic_lateral_shift(theta0=np.radians(aoi_y), **common)

        # Convert metres → microns.
        dx = np.asarray(dq_x) * 1e6
        dy = np.asarray(dq_y) * 1e6

        if lam1.ndim == 0:
            return float(dx), float(dy)
        return dx, dy


class RomanFilterRefractionBuilder(galsim.config.PhotonOpBuilder):
    """Build a RomanFilterRefraction PhotonOp from a config dict.
    Both SCA and SCA_pos are taken automatically from the base config.
    """

    def buildPhotonOp(self, config, base, logger):
        if "bandpass" not in base:
            raise galsim.GalSimConfigError("bandpass is required in base config for RomanFilterRefraction")
        if "image" not in base or "SCA" not in base["image"]:
            raise galsim.GalSimConfigError("SCA must be set in image config for RomanFilterRefraction")
        if "image_pos" not in base:
            raise galsim.GalSimConfigError("image_pos must be set in base config for RomanFilterRefraction")

        bandpass = base["bandpass"]
        sca = galsim.config.ParseValue(base["image"], "SCA", base, int)[0]
        pos = base["image_pos"]  # galsim.PositionD in pixel coordinates

        return RomanFilterRefraction(
            bandpass=bandpass,
            SCA=sca,
            SCA_pos=pos,
        )


galsim.config.RegisterPhotonOpType("RomanFilterRefraction", RomanFilterRefractionBuilder())

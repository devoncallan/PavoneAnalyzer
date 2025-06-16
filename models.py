import numpy as np

probe_radius_um = 100
probe_radius_mm = probe_radius_um / 1000
probe_radius_m = probe_radius_mm / 1000

sample_thickness_mm = 0.1
sample_thickness_um = sample_thickness_mm * 1000

# Assumed Poisson's ratio of the substrate
poisson_ratio = 0.5


def correction(d_max_um: np.ndarray):

    ratio = probe_radius_um / sample_thickness_um
    inv_ratio = 1 / ratio

    delta = d_max_um  # Max indentation depth
    R = probe_radius_um
    h = sample_thickness_um

    omega = ((R * delta) / (h**2)) ** (1.5)

    if ratio > 2:  # For 2 < ratio < 12.7 regime
        alpha = 9.5
        beta = 4.212
    else:
        alpha = 10.05 - 0.63 * np.sqrt(inv_ratio) * (3.1 + inv_ratio**2)
        beta = 4.8 - 4.23 * inv_ratio**2

    num = 1 + 2.3 * omega
    denom = 1 + 1.15 * omega ** (1.0 / 3) + alpha * omega + beta * omega**2

    correction_factor = num / denom

    return correction_factor


def hertz_model(d_positive_um: np.ndarray, E_Hertz: float):

    # E = 3 * (1 - v**2) * F / (4 * R**0.5 * d**1.5)
    # F = (4 * R**0.5 * d**1.5 * E) / (3 * (1 - v**2))

    # Missing poisson's ratio factor?
    return (4 * probe_radius_m**0.5 * d_positive_um**1.5 * E_Hertz) / 3


def hu_model(d_positive_um: np.ndarray, E_Hu: float):

    contact_area_radius_mm = np.sqrt(probe_radius_mm * d_positive_um / 1000)

    return ((4 * E_Hu * d_positive_um * contact_area_radius_mm) / 3) * ((2.36 * ()))

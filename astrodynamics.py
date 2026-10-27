"""Core astrodynamics routines.

Two-body and circular restricted three-body (CR3BP) dynamics, orbital element
conversions, analytical planetary ephemerides, Julian date conversions, J2 secular
rates and impulsive transfers.

Units are km, s and degrees unless stated otherwise.
"""
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.integrate import odeint

############################################################ CONSTANTS ############################################################

AU = 149597870.7           # astronomical unit (km)
R_EARTH = 6378.137         # Earth equatorial radius (km)
J2_EARTH = 1.08262668e-3   # Earth second zonal harmonic (-)
SOLAR_DAY = 86400.0        # (s)
SIDEREAL_DAY = 86164.0905  # (s)

PLANETS = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto']

# gravitational parameters (km^3/s^2)
MU = {
    'Sun': 1.32712440018e11,
    'Mercury': 2.20319e4,
    'Venus': 3.24859e5,
    'Earth': 3.986004418e5,
    'Mars': 4.28284e4,
    'Jupiter': 1.26713e8,
    'Saturn': 3.79406e7,
    'Uranus': 5.79456e6,
    'Neptune': 6.83653e6,
    'Pluto': 9.755e2,
}

# equatorial radii (km)
RADIUS = {
    'Mercury': 2.44e3,
    'Venus': 6.051e3,
    'Earth': R_EARTH,
    'Mars': 3.397e3,
    'Jupiter': 7.1492e4,
    'Saturn': 6.0268e4,
    'Uranus': 2.5559e4,
    'Neptune': 2.4764e4,
    'Pluto': 1.16e3,
}

# mean heliocentric orbit radii (km)
ORBIT_RADIUS = {
    'Mercury': 57.9e6,
    'Venus': 108.2e6,
    'Earth': 149.6e6,
    'Mars': 227.9e6,
    'Jupiter': 779.3e6,
    'Saturn': 1427e6,
    'Uranus': 2871e6,
    'Neptune': 4497e6,
    'Pluto': 5913e6,
}


def _lookup(table, name):
    key = name.strip().capitalize()
    if key not in table:
        raise ValueError(f"Unknown body '{name}'. Allowed values: {', '.join(table)}")
    return table[key]


def planet_mu(name):
    """Gravitational parameter (km^3/s^2) of the Sun or of a planet."""
    return _lookup(MU, name)


def planet_radius(name):
    """Equatorial radius (km) of a planet."""
    return _lookup(RADIUS, name)


def planet_orbit_radius(name):
    """Mean heliocentric orbit radius (km) of a planet."""
    return _lookup(ORBIT_RADIUS, name)


def planet_id(name):
    """Index (0 = Mercury ... 8 = Pluto) of a planet in PLANETS."""
    key = name.strip().capitalize()
    if key not in PLANETS:
        raise ValueError(f"Unknown planet '{name}'. Allowed values: {', '.join(PLANETS)}")
    return PLANETS.index(key)

########################################################### TWO BODY MODEL ########################################################

def two_body_model(state, t, mu):
    """Time derivative of the two-body state [x, y, z, vx, vy, vz] (km, km/s)."""
    state = np.asarray(state, dtype=float)
    r = state[:3]
    acc = -mu * r / np.linalg.norm(r)**3
    return np.concatenate((state[3:], acc))


def propagate_two_body(init_state, duration, n_steps, mu):
    """Integrate the two-body problem over `duration` seconds.

    Returns the X, Y, Z position histories (km).
    """
    t = np.linspace(0, duration, n_steps)
    sol = odeint(two_body_model, init_state, t, args=(mu,), rtol=1e-10, atol=1e-8)
    return sol[:, 0], sol[:, 1], sol[:, 2]

########################################################### THREE BODY MODEL ######################################################

def cr3bp_model(state, t, mu):
    """Time derivative of the non-dimensional CR3BP state in the rotating frame.

    mu is the mass ratio m2 / (m1 + m2).
    """
    x, y, z, vx, vy, vz = state
    r1_3 = ((x + mu)**2 + y**2 + z**2)**1.5
    r2_3 = ((x - 1 + mu)**2 + y**2 + z**2)**1.5
    ax = x + 2 * vy - (1 - mu) * (x + mu) / r1_3 - mu * (x - 1 + mu) / r2_3
    ay = y - 2 * vx - (1 - mu) * y / r1_3 - mu * y / r2_3
    az = -(1 - mu) * z / r1_3 - mu * z / r2_3
    return [vx, vy, vz, ax, ay, az]


def propagate_cr3bp(nd_init_state, tau, n_steps, mu):
    """Integrate the CR3BP for a non-dimensional time `tau` (2*pi = one revolution of the primaries).

    Returns the X, Y, Z position histories in the rotating frame.
    """
    t = np.linspace(0, tau, n_steps)
    sol = odeint(cr3bp_model, nd_init_state, t, args=(mu,), rtol=1e-11, atol=1e-12)
    return sol[:, 0], sol[:, 1], sol[:, 2]


def cr3bp_potential(x, y, mu, z=0.0):
    """CR3BP pseudo-potential Omega (works element-wise on arrays)."""
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)
    return 0.5 * (x**2 + y**2) + (1 - mu) / r1 + mu / r2


def jacobi_constant(state, mu):
    """Jacobi constant C = 2*Omega - v^2 of a non-dimensional CR3BP state."""
    x, y, z, vx, vy, vz = state
    return 2 * cr3bp_potential(x, y, mu, z) - (vx**2 + vy**2 + vz**2)


def libration_points(mu):
    """Rotating-frame coordinates of the five libration points, as a DataFrame."""
    def collinear(x):
        return x - (1 - mu) * (x + mu) / abs(x + mu)**3 - mu * (x - 1 + mu) / abs(x - 1 + mu)**3

    eps = 1e-9
    l1x = optimize.brentq(collinear, -mu + eps, 1 - mu - eps)
    l2x = optimize.brentq(collinear, 1 - mu + eps, 2.0)
    l3x = optimize.brentq(collinear, -2.0, -mu - eps)
    ly = np.sqrt(3) / 2

    data = [["L1", l1x, 0.0, 0.0],
            ["L2", l2x, 0.0, 0.0],
            ["L3", l3x, 0.0, 0.0],
            ["L4", 0.5 - mu, ly, 0.0],
            ["L5", 0.5 - mu, -ly, 0.0]]
    return pd.DataFrame(data, columns=["Point", "x", "y", "z"])

################################################## ORBIT REPRESENTATION CONVERSION #################################################

def _rot_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def keplerian_to_state(keplerian, mu):
    """Convert Keplerian elements to a Cartesian state vector.

    keplerian: [a (km), e, i (deg), RAAN (deg), arg of periapsis (deg), true anomaly (deg)]
    Returns [x, y, z, vx, vy, vz] (km, km/s). Elliptical orbits only (0 <= e < 1).
    """
    a, e, i, raan, aop, nu = keplerian

    if e < 0 or e >= 1:
        raise ValueError("Eccentricity must be 0 <= e < 1 for elliptical orbit.")
    if a <= 0:
        raise ValueError("Semi-major axis must be positive.")

    i, raan, aop, nu = np.radians([i, raan, aop, nu])

    p = a * (1 - e**2)  # semi-latus rectum
    r_mag = p / (1 + e * np.cos(nu))

    # position and velocity in the perifocal frame (PQW)
    r_pqw = r_mag * np.array([np.cos(nu), np.sin(nu), 0])
    v_pqw = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0])

    # PQW -> inertial: Rz(RAAN) * Rx(i) * Rz(aop)
    rot = _rot_z(raan) @ _rot_x(i) @ _rot_z(aop)
    return np.concatenate((rot @ r_pqw, rot @ v_pqw))


def state_to_keplerian(state, mu):
    """Convert a Cartesian state vector to Keplerian elements.

    state: [x, y, z, vx, vy, vz] (km, km/s)
    Returns (a, e, i, RAAN, arg of periapsis, true anomaly) with angles in deg.
    Elliptical orbits only. For equatorial orbits RAAN is set to 0 and the argument
    of periapsis is measured from the x axis; for circular orbits the argument of
    periapsis is set to 0 and the true anomaly becomes the argument of latitude.
    """
    tol = 1e-11
    r = np.asarray(state[:3], dtype=float)
    v = np.asarray(state[3:], dtype=float)

    r_mag = np.linalg.norm(r)
    if r_mag == 0:
        raise ValueError("Position vector magnitude is zero.")
    v_mag = np.linalg.norm(v)

    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)
    if h_mag == 0:
        raise ValueError("Angular momentum is zero (collinear position and velocity).")

    energy = v_mag**2 / 2 - mu / r_mag
    if energy >= 0:
        raise ValueError("Orbit is not bound (energy non-negative).")
    a = -mu / (2 * energy)

    e_vec = np.cross(v, h) / mu - r / r_mag
    e = np.linalg.norm(e_vec)
    if e >= 1:
        raise ValueError("Orbit is not elliptical (e >= 1).")

    i = np.arccos(np.clip(h[2] / h_mag, -1, 1))

    n = np.cross([0, 0, 1], h)  # node vector
    n_mag = np.linalg.norm(n)
    equatorial = n_mag < tol * h_mag
    circular = e < tol

    raan = 0.0 if equatorial else np.arctan2(n[1], n[0])

    def angle_from(ref, vec):
        # angle from `ref` to `vec`, measured in the orbit plane in the direction of motion
        return np.arctan2(np.dot(np.cross(ref, vec), h) / h_mag, np.dot(ref, vec))

    x_axis = np.array([1.0, 0.0, 0.0])
    if circular:
        aop = 0.0
        nu = angle_from(x_axis if equatorial else n, r)
    else:
        aop = angle_from(x_axis if equatorial else n, e_vec)
        nu = angle_from(e_vec, r)

    raan, aop, nu = (np.degrees([raan, aop, nu]) % 360)
    return a, e, np.degrees(i), raan, aop, nu


def keplerian_to_equinoctial(keplerian, fr=1):
    """Convert Keplerian elements to equinoctial elements.

    keplerian: [a (km), e, i (deg), RAAN (deg), arg of periapsis (deg), mean anomaly (deg)]
    fr: +1 for the direct set (standard), -1 for the retrograde set (i close to 180 deg).
    """
    a, e, i, raan, aop, mean_anom = keplerian
    i, raan, aop, mean_anom = np.radians([i, raan, aop, mean_anom])

    zeta = aop + fr * raan
    tan_half_i = np.tan(i / 2)**fr

    return {
        "a (km)": a,
        "h": e * np.sin(zeta),
        "k": e * np.cos(zeta),
        "p": tan_half_i * np.sin(raan),
        "q": tan_half_i * np.cos(raan),
        "lambda (deg)": np.degrees(mean_anom + zeta) % 360,
    }


def equinoctial_to_keplerian(equinoctial, fr=1):
    """Convert equinoctial elements to Keplerian elements.

    equinoctial: [a (km), h, k, p, q, mean longitude lambda (deg)]
    fr: +1 for the direct set (standard), -1 for the retrograde set.
    """
    a, h, k, p, q, lam = equinoctial

    e = np.hypot(h, k)
    s = np.hypot(p, q)  # tan(i/2)^fr
    i = 2 * np.arctan(s) if fr == 1 else np.pi - 2 * np.arctan(s)
    raan = np.arctan2(p, q) if s > 1e-12 else 0.0
    zeta = np.arctan2(h, k)  # aop + fr * RAAN

    return {
        "Semimajor axis (km)": a,
        "Eccentricity": e,
        "Inclination (deg)": np.degrees(i),
        "RAAN (deg)": np.degrees(raan) % 360,
        "Arg of Perigee (deg)": np.degrees(zeta - fr * raan) % 360,
        "Mean anomaly (deg)": (lam - np.degrees(zeta)) % 360,
    }

######################################################### INTERPLANETARY ##########################################################

# J2000 elements [a (AU), e, i, RAAN, longitude of perihelion, mean longitude] (deg)
_J2000_ELEMENTS = np.array([
    [0.38709927,  0.20563593,  7.00497902,  48.33076593,  77.45779628, 252.25032350],
    [0.72333566,  0.00677672,  3.39467605,  76.67984255, 131.60246718, 181.97909950],
    [1.00000261,  0.01671123, -0.00001531,   0.0,        102.93768193, 100.46457166],
    [1.52371034,  0.09339410,  1.84969142,  49.55953891, -23.94362959,  -4.55343205],
    [5.20288700,  0.04838624,  1.30439695, 100.47390909,  14.72847983,  34.39644501],
    [9.53667594,  0.05386179,  2.48599187, 113.66242448,  92.59887831,  49.95424423],
    [19.18916464, 0.04725744,  0.77263783,  74.01692503, 170.95427630, 313.23810451],
    [30.06992276, 0.00859048,  1.77004347, 131.78422574,  44.96476227, -55.12002969],
    [39.48211675, 0.24882730, 17.14001206, 110.30393684, 224.06891629, 238.92903833],
])

# rates of the J2000 elements per Julian century
_CENTENNIAL_RATES = np.array([
    [ 0.00000037,  0.00001906, -0.00594749, -0.12534081,  0.16047689, 149472.67411175],
    [ 0.00000390, -0.00004107, -0.00078890, -0.27769418,  0.00268329,  58517.81538729],
    [ 0.00000562, -0.00004392, -0.01294668,  0.0,         0.32327364,  35999.37244981],
    [ 0.0001847,   0.00007882, -0.00813131, -0.29257343,  0.44441088,  19140.30268499],
    [-0.00011607, -0.00013253, -0.00183714,  0.20469106,  0.21252668,   3034.74612775],
    [-0.00125060, -0.00050991,  0.00193609, -0.28867794, -0.41897216,   1222.49362201],
    [-0.00196176, -0.00004397, -0.00242939,  0.04240589,  0.40805281,    428.48202785],
    [ 0.00026291,  0.00005105,  0.00035372, -0.00508664, -0.32241464,    218.45945325],
    [-0.00031596,  0.00005170,  0.00004818, -0.01183482, -0.04062942,    145.20780515],
])


def planet_elements(planet, jd):
    """Heliocentric elements of a planet at a Julian date.

    planet: planet index (0 - 8) or name.
    Returns [a (km), e, i, RAAN, longitude of perihelion, mean longitude] (deg).
    """
    if isinstance(planet, str):
        planet = planet_id(planet)
    centuries = (jd - 2451545.0) / 36525  # Julian centuries since J2000
    elements = _J2000_ELEMENTS[planet] + _CENTENNIAL_RATES[planet] * centuries
    elements[0] *= AU
    return elements


def planet_state_vector(planet, jd):
    """Heliocentric position (km) and velocity (km/s) of a planet at a Julian date."""
    a, e, inc, raan, lon_peri, mean_lon = planet_elements(planet, jd)
    aop = (lon_peri - raan) % 360
    mean_anom = (mean_lon - lon_peri) % 360
    ecc_anom = solve_kepler(e, np.radians(mean_anom))
    true_anom = np.degrees(2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(ecc_anom / 2))) % 360
    state = keplerian_to_state([a, e, inc, raan, aop, true_anom], MU['Sun'])
    return state[:3], state[3:]

##################################################### GENERAL ASTRODYNAMICS #######################################################

def solve_kepler(e, mean_anom, tol=1e-10, max_iter=50):
    """Solve Kepler's equation E - e*sin(E) = M for the eccentric anomaly (rad) with Newton's method."""
    ecc_anom = mean_anom + e / 2 if mean_anom < np.pi else mean_anom - e / 2
    for _ in range(max_iter):
        ratio = (ecc_anom - e * np.sin(ecc_anom) - mean_anom) / (1 - e * np.cos(ecc_anom))
        ecc_anom -= ratio
        if abs(ratio) < tol:
            return ecc_anom
    raise RuntimeError("Kepler's equation did not converge.")


def mean_motion(sma, mu):
    """Mean motion (rad/s) from the semimajor axis (km)."""
    return np.sqrt(mu / sma**3)


def mean_motion_from_period(period):
    """Mean motion (rad/s) from the orbit period (s)."""
    return 2 * np.pi / period


def orbital_period(sma, mu):
    """Orbit period (s) from the semimajor axis (km)."""
    return 2 * np.pi * np.sqrt(sma**3 / mu)


def flight_path_angle(theta, ecc):
    """Flight path angle (deg) from the true anomaly (deg) and the eccentricity."""
    theta = np.radians(theta)
    return np.degrees(np.arctan2(ecc * np.sin(theta), 1 + ecc * np.cos(theta)))


def nodal_regression_rate(alt, inc, ecc=0.0):
    """Secular RAAN drift due to J2 (deg/day) for altitude (km) and inclination (deg)."""
    sma = R_EARTH + np.asarray(alt)
    p = sma * (1 - ecc**2)
    rate = -1.5 * mean_motion(sma, MU['Earth']) * J2_EARTH * (R_EARTH / p)**2 * np.cos(np.radians(inc))
    return np.degrees(rate) * SOLAR_DAY


def apsidal_rotation_rate(alt, inc, ecc=0.0):
    """Secular argument of perigee drift due to J2 (deg/day) for altitude (km) and inclination (deg)."""
    sma = R_EARTH + np.asarray(alt)
    p = sma * (1 - ecc**2)
    rate = 0.75 * mean_motion(sma, MU['Earth']) * J2_EARTH * (R_EARTH / p)**2 * (4 - 5 * np.sin(np.radians(inc))**2)
    return np.degrees(rate) * SOLAR_DAY

############################################################ TIME ##################################################################

def jd_from_gregorian(year, month, day, hour=0, minute=0, second=0):
    """Julian date of a Gregorian calendar date and UT time (valid 1900 - 2100)."""
    j0 = (367 * year - np.fix(7 * (year + np.fix((month + 9) / 12)) / 4)
          + np.fix(275 * month / 9) + day + 1721013.5)
    return j0 + (hour + minute / 60 + second / 3600) / 24


def jd_from_date_string(date, fmt='%d-%m-%Y'):
    """Julian date of a date string, by default in dd-mm-yyyy format."""
    dt = datetime.strptime(date, fmt)
    return jd_from_gregorian(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)


def datetime_from_jd(jd):
    """UTC datetime of a Julian date."""
    jdn = int(jd)
    L = jdn + 68569
    N = int(4 * L / 146097)
    L = L - int((146097 * N + 3) / 4)
    I = int(4000 * (L + 1) / 1461001)
    L = L - int(1461 * I / 4) + 31
    J = int(80 * L / 2447)
    day = L - int(2447 * J / 80)
    L = int(J / 11)
    month = J + 2 - 12 * L
    year = 100 * (N - 49) + I + L
    # the Julian day starts at noon
    return datetime(year, month, day, tzinfo=timezone.utc) + timedelta(days=jd % 1, hours=12)


def date_string_from_jd(jd, fmt='%d-%m-%Y'):
    """Date string of a Julian date, by default in dd-mm-yyyy format."""
    return datetime_from_jd(float(jd)).strftime(fmt)

######################################################### ORBIT TRANSFERS ##########################################################

def hohmann_delta_vs(mu, r1, r2):
    """Departure and arrival Delta V (km/s) of a Hohmann transfer between circular orbits r1 -> r2 (km).

    The values are positive when raising the orbit and negative when lowering it.
    """
    dv1 = np.sqrt(mu / r1) * (np.sqrt(2 * r2 / (r1 + r2)) - 1)
    dv2 = np.sqrt(mu / r2) * (1 - np.sqrt(2 * r1 / (r1 + r2)))
    return dv1, dv2


def hohmann_transfer_time(mu, r1, r2):
    """Duration (s) of a Hohmann transfer between circular orbits r1 and r2 (km)."""
    return orbital_period((r1 + r2) / 2, mu) / 2


def bielliptic_delta_v(mu, r1, r2, rb):
    """Total Delta V (km/s) of a bi-elliptic transfer r1 -> rb -> r2 between circular orbits (r1 < r2 < rb)."""
    if not r1 < r2 < rb:
        raise ValueError("Must have r1 < r2 < rb")
    a1 = (r1 + rb) / 2
    a2 = (r2 + rb) / 2
    dv1 = np.sqrt(mu * (2 / r1 - 1 / a1)) - np.sqrt(mu / r1)    # leave r1
    dv2 = np.sqrt(mu * (2 / rb - 1 / a2)) - np.sqrt(mu * (2 / rb - 1 / a1))  # raise periapsis at rb
    dv3 = np.sqrt(mu * (2 / r2 - 1 / a2)) - np.sqrt(mu / r2)    # circularize at r2
    return dv1 + dv2 + dv3

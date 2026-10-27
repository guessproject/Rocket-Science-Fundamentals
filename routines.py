"""Plotting and reporting routines used by the Astrodynamics notebooks."""
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from scipy import optimize
from scipy.integrate import odeint

import astrodynamics as ad
import lambert


def _set_equal_3d_limits(ax, *coords):
    lim = max(np.max(np.abs(c)) for c in coords)
    ax.set_xlim3d(-lim, lim)
    ax.set_ylim3d(-lim, lim)
    ax.set_zlim3d(-lim, lim)
    ax.set_box_aspect((1, 1, 1))

########################################################### TWO BODY ##############################################################

def plot_two_body_orbit(init_state, duration, n_steps, central_body):
    """3D plot of a Keplerian orbit propagated for `duration` seconds around `central_body`."""
    X, Y, Z = ad.propagate_two_body(init_state, duration, n_steps, ad.planet_mu(central_body))

    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    ax.set_title('Orbit in inertial frame')
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_zlabel('z (km)')
    ax.plot3D(X, Y, Z, c='green', label='Keplerian orbit')

    # central body
    phi, theta = np.meshgrid(np.linspace(0, 2 * np.pi, 50), np.linspace(0, np.pi, 50))
    r_cb = ad.planet_radius(central_body)
    ax.plot_surface(r_cb * np.cos(phi) * np.sin(theta), r_cb * np.sin(phi) * np.sin(theta),
                    r_cb * np.cos(theta), color='gray', alpha=0.7)

    _set_equal_3d_limits(ax, X, Y, Z)
    ax.legend()
    plt.show()


def plot_circular_orbit_parameters():
    """Period, velocity, revolutions per day and energy of circular Earth orbits vs altitude."""
    mu = ad.MU['Earth']
    altitude = np.linspace(250, 36000, 100)  # km
    sma = ad.R_EARTH + altitude

    period = ad.orbital_period(sma, mu)
    velocity = np.sqrt(mu / sma)
    energy = -mu / (2 * sma)
    revs_day = 86400 / period

    plt.figure(figsize=(10, 6))
    panels = [(period / 60, 'Period (min)'), (velocity, 'Velocity (km/s)'),
              (revs_day, 'Revs per day'), (energy, 'Energy (km^2/s^2)')]
    for k, (values, label) in enumerate(panels, 1):
        plt.subplot(2, 2, k)
        plt.plot(altitude / 1000, values, 'black')
        plt.xlabel('Altitude (km*10^3)')
        plt.ylabel(label)
        plt.grid(True)
    plt.tight_layout()
    plt.show()

######################################################### PERTURBATIONS ###########################################################

def _plot_j2_rate(rate_function, ylabel, title):
    inc = np.linspace(0, 180, 1000)
    plt.figure(figsize=(8, 4))
    for alt in [400, 600, 800, 1000]:
        plt.plot(inc, rate_function(alt, inc), linewidth=2, label=f'{alt} km')
    plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
    plt.axvline(90, color='gray', linestyle=':', linewidth=0.8)
    plt.xlabel('Inclination [deg]')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Altitude')
    plt.tight_layout()
    plt.show()


def plot_nodal_regression():
    """J2 nodal regression rate vs inclination for a set of circular orbit altitudes."""
    _plot_j2_rate(ad.nodal_regression_rate, 'Nodal Regression Rate [deg/day]', 'J2 Nodal Regression Rate vs Inclination')


def plot_apsidal_rotation():
    """J2 apsidal rotation rate vs inclination for a set of circular orbit altitudes."""
    _plot_j2_rate(ad.apsidal_rotation_rate, 'Apsidal Rotation Rate [deg/day]', 'J2 Apsidal Rotation Rate vs Inclination')


def plot_sso_inclination(ecc):
    """Inclination of Sun-synchronous orbits vs altitude for a given eccentricity."""
    mu = ad.MU['Earth']
    raan_dot = 1.991063853e-7  # Sun-synchronous nodal drift (rad/s)
    altitude = np.linspace(250, 1000, 400)
    sma = ad.R_EARTH + altitude
    inc = np.degrees(np.arccos(-2 * sma**3.5 * raan_dot * (1 - ecc**2)**2
                               / (3 * ad.R_EARTH**2 * ad.J2_EARTH * np.sqrt(mu))))

    plt.figure(figsize=(12, 4))
    plt.plot(altitude, inc, color='black', label='inc')
    plt.grid(True)
    plt.xlabel('alt (km)')
    plt.ylabel('inc (deg)')
    plt.legend()
    plt.title(f'SSO Orbit Inclination for ecc = {ecc}')
    plt.show()

########################################################### MANEUVERS #############################################################

def plot_intrack_maneuver_effects(init_sma, init_ecc, delta_v):
    """Changes of sma, eccentricity and argument of periapsis caused by an in-track Delta V (m/s)
    applied at each true anomaly of an equatorial Earth orbit."""
    mu = ad.MU['Earth']
    true_anomaly = np.arange(360)
    d_sma = np.empty(360)
    d_ecc = np.empty(360)
    aop = np.empty(360)
    for k, ta in enumerate(true_anomaly):
        state = ad.keplerian_to_state([init_sma, init_ecc, 0, 0, 0, ta], mu)
        v = state[3:]
        v_mag = np.linalg.norm(v)
        state[3:] = v * (v_mag + delta_v * 0.001) / v_mag
        kep = ad.state_to_keplerian(state, mu)
        d_sma[k] = kep[0] - init_sma
        d_ecc[k] = kep[1] - init_ecc
        aop[k] = kep[4] - 360 if kep[4] > 180 else kep[4]

    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    panels = [(d_sma, 'Delta semimajor axis', 'km', 'blue'),
              (d_ecc, 'Delta eccentricity', None, 'red'),
              (aop, 'Delta argument of periapsis', 'deg', 'green')]
    for ax, (values, title, unit, color) in zip(axs, panels):
        ax.plot(true_anomaly, values, color=color)
        ax.set_title(title)
        if unit:
            ax.set_ylabel(unit)
        ax.grid(True, which='both', linestyle='--', linewidth=0.8)
    axs[0].set_xticks(np.arange(0, 361, 15))
    axs[2].set_xlabel('True anomaly at Δv (deg)')
    plt.tight_layout()
    plt.show()


def plot_hohmann_vs_bielliptic(r1, r2, rb):
    """Compare Hohmann and bi-elliptic transfers between circular Earth orbits r1 -> r2 (km)."""
    mu = ad.MU['Earth']
    dv_hohmann = sum(abs(dv) for dv in ad.hohmann_delta_vs(mu, r1, r2))
    dv_bielliptic = ad.bielliptic_delta_v(mu, r1, r2, rb)
    print(f"Hohmann Delta-V: {dv_hohmann} km/s")
    print(f"Bielliptic Delta-V: {dv_bielliptic} km/s")

    # parametric analysis on the intermediate radius
    rb_ratios = np.linspace(1.01, 10, 100)
    dv_bi = [ad.bielliptic_delta_v(mu, r1, r2, ratio * r2) for ratio in rb_ratios]

    plt.figure(figsize=(12, 4))
    plt.plot(rb_ratios, dv_bi, label='Bielliptic Delta-V')
    plt.plot(rb_ratios, np.full_like(rb_ratios, dv_hohmann), label='Hohmann Delta-V', linestyle='--')
    plt.xlabel('Intermediate Radius Ratio (rb / r2)')
    plt.ylabel('Total Delta-V (km/s)')
    plt.title('Hohmann vs Bielliptic Transfer Delta-V')
    plt.legend()
    plt.grid(True)
    plt.show()


def print_optimal_plane_change(delta_i, r_1, r_2):
    """Optimal split of an inclination change delta_i (deg) between the two burns of a Hohmann
    transfer from circular orbit r_1 to circular orbit r_2 (km), around the Earth."""
    mu = ad.MU['Earth']
    di_total = np.radians(delta_i)

    v_1 = np.sqrt(mu / r_1)
    v_2 = np.sqrt(mu / r_2)
    a_t = (r_1 + r_2) / 2
    v_t1 = np.sqrt(mu * (2 / r_1 - 1 / a_t))
    v_t2 = np.sqrt(mu * (2 / r_2 - 1 / a_t))

    def burn(v_a, v_b, di):
        return np.sqrt(v_a**2 + v_b**2 - 2 * v_a * v_b * np.cos(di))

    def d_total_dv(di_1):
        di_2 = di_total - di_1
        return (v_1 * v_t1 * np.sin(di_1) / burn(v_1, v_t1, di_1)
                - v_2 * v_t2 * np.sin(di_2) / burn(v_2, v_t2, di_2))

    dv_full_at_r2 = abs(v_t1 - v_1) + burn(v_t2, v_2, di_total)
    di_1_opt = optimize.newton(d_total_dv, x0=0.0)
    dv_opt = burn(v_1, v_t1, di_1_opt) + burn(v_2, v_t2, di_total - di_1_opt)

    print(f"Full inclination change at r2: ΔV = {dv_full_at_r2} km/s")
    print(f"Optimal Δi_1 = {np.degrees(di_1_opt)} deg")
    print(f"Optimal Δi_2 = {delta_i - np.degrees(di_1_opt)} deg")
    print(f"Optimal ΔV = {dv_opt} km/s")

######################################################### INTERPLANETARY ##########################################################

def plot_transfer(central_body, dep_planet, arr_planet, jd_dep, jd_arr, n_steps):
    """Plot the Lambert transfer between two planets and print the hyperbolic excess velocities."""
    mu = ad.planet_mu(central_body)
    tof = (jd_arr - jd_dep) * 86400

    r1, v1 = ad.planet_state_vector(dep_planet, jd_dep)
    r2, v2 = ad.planet_state_vector(arr_planet, jd_arr)
    X1, Y1, Z1 = ad.propagate_two_body(np.concatenate((r1, v1)),
                                       ad.orbital_period(ad.planet_orbit_radius(dep_planet), mu), n_steps, mu)
    X2, Y2, Z2 = ad.propagate_two_body(np.concatenate((r2, v2)),
                                       ad.orbital_period(ad.planet_orbit_radius(arr_planet), mu), n_steps, mu)

    v1_list, v2_list = lambert.solve(mu, r1, r2, tof, 'prograde')
    v1_trans, v2_trans = v1_list[0], v2_list[0]
    X_t, Y_t, Z_t = ad.propagate_two_body(np.concatenate((r1, v1_trans)), tof, n_steps, mu)

    v_inf_1 = np.linalg.norm(v1_trans - v1)
    v_inf_2 = np.linalg.norm(v2_trans - v2)
    print(f'Flight Time = {round(tof / 86400, 2)} days')
    print(f'V1∞    = {v_inf_1} km/sec')
    print(f'V2∞    = {v_inf_2} km/sec')
    print(f'V∞ tot = {v_inf_1 + v_inf_2} km/sec')

    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    ax.set_title('Transfer orbit')
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_zlabel('z (km)')
    ax.plot3D(X1, Y1, Z1, c='gray')
    ax.plot3D(X2, Y2, Z2, c='gray')
    ax.plot3D(0, 0, 0, c='black', marker='o', label=central_body)
    ax.plot3D(X1[0], Y1[0], Z1[0], c='blue', marker='o', label=dep_planet)
    ax.plot3D(X2[0], Y2[0], Z2[0], c='green', marker='o', label=arr_planet)
    ax.plot3D(X_t, Y_t, Z_t, c='red', label='Transfer orbit')
    _set_equal_3d_limits(ax, X1, Y1, X2, Y2)
    ax.legend()
    plt.show()


def plot_angular_extents(peri_alt, departure_planet):
    """Angular extent beta of the departure hyperbola vs C3, for a periapsis altitude (km)."""
    mu = ad.planet_mu(departure_planet)
    c3 = np.arange(1, 81)
    ecc = 1 + (peri_alt + ad.planet_radius(departure_planet)) * c3 / mu
    beta = np.degrees(np.arccos(1 / ecc))

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.set_xlabel('C3 (km^2/sec^2)', labelpad=2)
    ax.set_ylabel('β (deg)', labelpad=2)
    ax.plot(c3, beta, color='black', linewidth=2)
    plt.title('Angular extent')
    plt.grid(True)
    plt.show()

###################################################### PROXIMITY OPERATIONS #######################################################

def plot_hcw(initial_state, alt, num_revs):
    """Relative motion from the Hill-Clohessy-Wiltshire equations.

    initial_state: [x, y, z, vx, vy, vz] (m, m/s) with x radial, y in-track, z cross-track
    alt:           altitude of the circular target orbit (km)
    """
    n = ad.mean_motion(ad.R_EARTH + alt, ad.MU['Earth'])  # rad/s

    def hcw(state, t):
        x, y, z, vx, vy, vz = state
        return [vx, vy, vz, 2 * n * vy + 3 * n**2 * x, -2 * n * vx, -n**2 * z]

    t = np.linspace(0, num_revs * 2 * np.pi / n, 1000)
    sol = odeint(hcw, initial_state, t)
    x, y, z = sol[:, 0], sol[:, 1], sol[:, 2]

    x0, vx0, vy0 = initial_state[0], initial_state[3], initial_state[4]
    amplitude = math.hypot(3 * x0 + 2 * vy0 / n, vx0 / n)
    print(f"Mean motion = {n} rad/sec")
    print(f'Radial oscillation amplitude = {math.trunc(amplitude)} m')
    print(f'Intrack oscillation amplitude = {math.trunc(2 * amplitude)} m')
    print(f'Critical intrack rate for drift stopping = {-2 * n * x0} m/s')
    print(f'Critical intrack rate for straight line = {-1.5 * n * x0} m/s')

    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    fig.suptitle('Relative path', fontsize=10)
    projections = [(y, x, 'XY', 'Y (m)', 'X (m)', 'blue'),
                   (x, z, 'XZ', 'X (m)', 'Z (m)', 'red'),
                   (y, z, 'YZ', 'Y (m)', 'Z (m)', 'green')]
    for ax, (h, v, title, xlabel, ylabel, color) in zip(axs, projections):
        ax.plot(h, v, color=color)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    ax = plt.axes(projection='3d')
    ax.set_title('Relative motion')
    ax.set_xlabel('Y (Intrack - m)')
    ax.set_ylabel('Z (Crosstrack - m)')
    ax.set_zlabel('X (Radial - m)')
    ax.plot3D(y, z, x, c='green')
    ax.plot3D(0, 0, 0, c='black', marker='o')
    plt.show()

########################################################## THREE BODY #############################################################

def print_cr3bp_earth_moon_parameters():
    """Characteristic quantities and mass parameter of the Earth-Moon CR3BP."""
    G = 6.67408e-20                    # universal gravitational constant (km^3/kg/s^2)
    m_earth = 5.972364886393727e24     # (kg)
    m_moon = 7.34603167111482e22       # (kg)
    l_star = 390877.4158212686         # semimajor axis of the (circular) Moon orbit, from STK (km)

    m_star = m_earth + m_moon
    t_star = math.sqrt(l_star**3 / (G * m_star))  # the primaries' period is 2*pi*t_star
    v_star = l_star / t_star
    mu = m_moon / m_star

    print("Dimensional parameters for the Earth-Moon system:")
    print(f'm* = {m_star} kg')
    print(f'l* = {l_star} km')
    print(f't* = {t_star} sec ({t_star / 86400} days)')
    print(f'v* = {v_star} km/sec')
    print(f'Orbit period of the primaries = 2πt* = {2 * math.pi * t_star / 86400} days')
    print('')
    print("Mass parameter for the Earth Moon system:")
    print(f'μ = {mu}')


def plot_cr3bp_orbit(nd_init_state, tau, n_steps, mu):
    """3D plot of a CR3BP trajectory in the rotating frame."""
    X, Y, Z = ad.propagate_cr3bp(nd_init_state, tau, n_steps, mu)

    plt.figure(figsize=(10, 7))
    ax = plt.axes(projection='3d')
    ax.set_title(f'Rotating Frame CR3BP Orbit (μ = {round(mu, 6)})')
    ax.set_xlabel('x [ND]')
    ax.set_ylabel('y [ND]')
    ax.set_zlabel('z [ND]')
    ax.plot3D(X, Y, Z, c='green')
    ax.plot3D(-mu, 0, 0, c='black', marker='o', label='Primary')
    ax.plot3D(1 - mu, 0, 0, c='purple', marker='o', label='Secondary')
    ax.legend()
    _set_equal_3d_limits(ax, X, Y, Z, [1.0])
    plt.show()


def plot_cr3bp_potential(mu):
    """3D surface of the CR3BP pseudo-potential."""
    X, Y = np.meshgrid(np.linspace(-1.1, 1.1, 1000), np.linspace(-1.1, 1.1, 1000))
    # limit the peaks close to the primaries for visualization
    omega = np.clip(ad.cr3bp_potential(X, Y, mu), None, 1.8)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, omega, cmap='managua', linewidth=0, antialiased=True, alpha=0.7)
    ax.scatter(-mu, 0, omega.min(), color='blue', s=100, label='Primary')
    ax.scatter(1 - mu, 0, omega.min(), color='red', s=50, label='Secondary')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel(r'$\Omega$')
    ax.set_title('CR3BP Effective Potential')
    ax.view_init(elev=200, azim=210)
    fig.colorbar(surf, shrink=0.6, label='Potential')
    ax.legend()
    plt.tight_layout()
    plt.show()


def print_libration_points_jacobi(mu):
    """Jacobi constant at the five libration points."""
    print(f"Jacobi Constant at libration points for μ = {mu}")
    for _, p in ad.libration_points(mu).iterrows():
        print(f"{p['Point']} = {ad.jacobi_constant([p['x'], p['y'], p['z'], 0, 0, 0], mu)}")


def plot_libration_points(df, mu):
    """Plot the libration points (DataFrame from astrodynamics.libration_points) and the primaries."""
    print(f"Libration points coordinates for μ = {mu}")
    print("")
    print(df)

    # center of mass marker
    circle = mpath.Path.unit_circle()
    wedge_1 = mpath.Path.wedge(90, 180)
    wedge_2 = mpath.Path.wedge(270, 0)
    verts = np.concatenate([circle.vertices, wedge_1.vertices[::-1, ...], wedge_2.vertices[::-1, ...]])
    codes = np.concatenate([circle.codes, wedge_1.codes, wedge_2.codes])
    center_of_mass = mpath.Path(verts, codes)

    theta = np.linspace(0, 2 * np.pi, 200)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=96)
    ax.set_xlabel("$x^*$")
    ax.set_ylabel("$y^*$")
    ax.axhline(0, color='k')
    # orbits of m2 and m1 around the barycenter
    ax.plot((1 - mu) * np.cos(theta), (1 - mu) * np.sin(theta))
    ax.plot(mu * np.cos(theta), mu * np.sin(theta))
    ax.plot([-mu, 0.5 - mu, 1 - mu, 0.5 - mu, -mu], [0, np.sqrt(3) / 2, 0, -np.sqrt(3) / 2, 0], 'k', ls="--", lw=1)
    for (_, p), marker in zip(df.iterrows(), ['rv', 'r^', 'rp', 'rX', 'rs']):
        ax.plot(p['x'], p['y'], marker, label=f"$L_{p['Point'][1]}$")
    ax.plot(0, 0, 'k', marker=center_of_mass, markersize=10)
    ax.plot(-mu, 0, 'bo', label="$m1$")
    ax.plot(1 - mu, 0, 'go', label="$m2$")
    ax.legend(loc='upper left')
    ax.set_aspect("equal")


def _jacobi_grid(mu, size=3.0, n=512):
    # cell-centered grid of the zero-velocity Jacobi constant 2*Omega
    x = (np.arange(n) + 0.5) * size / n - size / 2
    X, Y = np.meshgrid(x, x)
    return x, X, Y, 2 * ad.cr3bp_potential(X, Y, mu)


def plot_zero_velocity_curves(mu):
    """Zero-velocity curves for a range of Jacobi constants."""
    size = 3.0
    x, X, Y, J = _jacobi_grid(mu, size)
    levels = np.linspace(2.5, 4, 10)

    plt.figure(figsize=(9.5, 7))
    plt.title(f'Zero-Velocity Curve for μ = {mu}')
    plt.xlabel('X Position (scaled units)')
    plt.ylabel('Y Position (scaled units)')
    plt.xlim(-size / 2, size / 2)
    plt.ylim(-size / 2, size / 2)
    cf = plt.contourf(X, Y, J, levels=levels, extend='max')
    C = plt.contour(X, Y, J, levels=levels, colors='black')
    plt.clabel(C, inline=1, fontsize=10)
    dx = size / len(x)
    plt.scatter([-mu, 1 - mu], [0, 0])
    for xp, label in zip([-mu, 1 - mu], ["M1", "M2"]):
        plt.text(xp + 10 * dx, -10 * dx, label)
    plt.colorbar(cf)
    plt.show()


def plot_zero_velocity_curve(jacobi, mu):
    """Zero-velocity curve for a single Jacobi constant; the forbidden region is shaded."""
    size = 3.0
    x, X, Y, J = _jacobi_grid(mu, size)
    df = ad.libration_points(mu)

    plt.figure(figsize=(9.5, 7))
    plt.title(f'Zero-Velocity Surfaces (Rotating Frame), C = {jacobi}')
    plt.xlabel('X Position (scaled units)')
    plt.ylabel('Y Position (scaled units)')
    plt.xlim(-size / 2, size / 2)
    plt.ylim(-size / 2, size / 2)
    # motion is only possible where 2*Omega >= C
    plt.contourf(X, Y, J, levels=[J.min(), jacobi], colors=['0.75'])
    C = plt.contour(X, Y, J, levels=[jacobi], colors='black')
    plt.clabel(C, inline=5, fontsize=10)
    plt.gca().set_aspect('equal')

    dx = size / len(x)
    plt.scatter([-mu, 1 - mu], [0, 0])
    collinear = df.iloc[:3]
    plt.scatter(collinear['x'], [0, 0, 0])
    for xp, label in zip(collinear['x'], collinear['Point']):
        plt.text(xp + 5 * dx, -5 * dx, label)
    plt.show()

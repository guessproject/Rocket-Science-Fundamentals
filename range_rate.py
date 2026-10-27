import numpy as np


# ============================================================
# Physical constants
# ============================================================

MU_EARTH = 3.986004418e14       # Earth gravitational parameter [m^3/s^2]
R_EARTH = 6_378_137.0           # Earth equatorial radius [m]
OMEGA_EARTH = 7.2921150e-5      # Earth rotation rate [rad/s]
C_LIGHT = 299_792_458.0         # Speed of light [m/s]


# ============================================================
# Example spacecraft propagator
# ============================================================

ALTITUDE = 700_000.0            # Circular-orbit altitude [m]
ORBIT_RADIUS = R_EARTH + ALTITUDE

# Circular-orbit angular rate
MEAN_MOTION = np.sqrt(MU_EARTH / ORBIT_RADIUS**3)


def spacecraft_position(t):
    """
    Spacecraft position in an Earth-centered inertial frame.

    A circular, equatorial orbit is used for this example.

    Parameters
    ----------
    t : float
        Time from the reference epoch [s].

    Returns
    -------
    numpy.ndarray
        Spacecraft inertial position [m].
    """
    theta = MEAN_MOTION * t

    return ORBIT_RADIUS * np.array([
        np.cos(theta),
        np.sin(theta),
        0.0
    ])


# ============================================================
# Ground-station position
# ============================================================

STATION_LATITUDE = np.deg2rad(35.0)
STATION_LONGITUDE_0 = np.deg2rad(-120.0)


def station_position(t):
    """
    Ground-station position in the same inertial frame used
    for the spacecraft.

    The Earth is modeled as a rotating sphere.

    Parameters
    ----------
    t : float
        Time from the reference epoch [s].

    Returns
    -------
    numpy.ndarray
        Ground-station inertial position [m].
    """
    longitude = STATION_LONGITUDE_0 + OMEGA_EARTH * t

    return R_EARTH * np.array([
        np.cos(STATION_LATITUDE) * np.cos(longitude),
        np.cos(STATION_LATITUDE) * np.sin(longitude),
        np.sin(STATION_LATITUDE)
    ])


# ============================================================
# Light-time-corrected range
# ============================================================

def evaluate_one_way_range(
    reception_time,
    spacecraft_position_function,
    station_position_function,
    tolerance=1.0e-13,
    max_iterations=20
):
    """
    Evaluate one-way range including signal light-time delay.

    The measurement is assumed to be time-tagged at signal
    reception by the ground station.

    Parameters
    ----------
    reception_time : float
        Signal reception time [s].

    spacecraft_position_function : callable
        Function returning spacecraft inertial position [m].

    station_position_function : callable
        Function returning station inertial position [m].

    tolerance : float
        Convergence tolerance on light time [s].

    max_iterations : int
        Maximum number of fixed-point iterations.

    Returns
    -------
    result : dict
        Dictionary containing range, light time, transmission
        time, positions and iteration history.
    """

    # Station position at signal reception
    r_station_rx = station_position_function(reception_time)

    # Step 1: instantaneous range as the initial estimate
    r_spacecraft_rx = spacecraft_position_function(reception_time)

    instantaneous_range = np.linalg.norm(
        r_spacecraft_rx - r_station_rx
    )

    light_time = instantaneous_range / C_LIGHT

    iteration_history = []

    # Step 2: iterate the retarded spacecraft time
    for iteration in range(1, max_iterations + 1):

        # Estimated signal-transmission time
        transmission_time = reception_time - light_time

        # Spacecraft position when the signal was transmitted
        r_spacecraft_tx = spacecraft_position_function(
            transmission_time
        )

        # New geometric path length
        corrected_range = np.linalg.norm(
            r_spacecraft_tx - r_station_rx
        )

        # New light-time estimate
        new_light_time = corrected_range / C_LIGHT

        change_in_light_time = abs(
            new_light_time - light_time
        )

        iteration_history.append({
            "iteration": iteration,
            "transmission_time_s": transmission_time,
            "range_m": corrected_range,
            "light_time_s": new_light_time,
            "light_time_change_s": change_in_light_time
        })

        # Step 3: convergence test
        if change_in_light_time < tolerance:
            return {
                "converged": True,
                "iterations": iteration,
                "reception_time_s": reception_time,
                "transmission_time_s":
                    reception_time - new_light_time,
                "instantaneous_range_m": instantaneous_range,
                "corrected_range_m": corrected_range,
                "light_time_s": new_light_time,
                "light_time_correction_m":
                    corrected_range - instantaneous_range,
                "spacecraft_position_tx_m": r_spacecraft_tx,
                "station_position_rx_m": r_station_rx,
                "history": iteration_history
            }

        light_time = new_light_time

    raise RuntimeError(
        f"Light-time iteration did not converge after "
        f"{max_iterations} iterations."
    )


# ============================================================
# Numerical example
# ============================================================

if __name__ == "__main__":

    # Measurement reception time
    t_receive = 1800.0

    result = evaluate_one_way_range(
        reception_time=t_receive,
        spacecraft_position_function=spacecraft_position,
        station_position_function=station_position
    )

    print("Light-time-corrected range solution")
    print("-----------------------------------")
    print(f"Converged              : {result['converged']}")
    print(f"Iterations             : {result['iterations']}")
    print(f"Reception time         : {result['reception_time_s']:.9f} s")
    print(f"Transmission time      : {result['transmission_time_s']:.9f} s")
    print(f"Light time             : {result['light_time_s']:.12f} s")
    print(f"Instantaneous range    : "
          f"{result['instantaneous_range_m'] / 1000.0:.6f} km")
    print(f"Corrected range        : "
          f"{result['corrected_range_m'] / 1000.0:.6f} km")
    print(f"Light-time correction  : "
          f"{result['light_time_correction_m']:.6f} m")

    print("\nIteration history")
    print("-----------------")

    for item in result["history"]:
        print(
            f"k = {item['iteration']:2d}, "
            f"tau = {item['light_time_s']:.15f} s, "
            f"rho = {item['range_m']:.6f} m, "
            f"|delta tau| = "
            f"{item['light_time_change_s']:.3e} s"
        )
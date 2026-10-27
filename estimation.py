"""Least squares and Kalman filter examples used by the Observation and Estimation notebooks."""
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def train_tracking(x0, v0, interval, n_meas_a, sigma_a, n_meas_b, sigma_b):
    """Weighted least squares estimate of a train's x0, v0 from two trackers, and their fusion."""
    times = {'A': np.arange(0, interval, interval / n_meas_a),
             'B': np.arange(0, interval, interval / n_meas_b)}
    sigmas = {'A': sigma_a, 'B': sigma_b}

    estimates, covariances, residuals = {}, {}, {}
    for name, t in times.items():
        # simulate the measurements
        y = x0 + v0 * t + sigmas[name] * np.random.randn(len(t))
        H = np.column_stack((np.ones(len(t)), t))
        W = np.eye(len(t)) / sigmas[name]**2
        # normal equation
        P = np.linalg.inv(H.T @ W @ H)
        x_hat = P @ H.T @ W @ y
        estimates[name], covariances[name] = x_hat, P
        residuals[name] = y - H @ x_hat

    # sensor fusion
    var_a = np.diag(covariances['A'])
    var_b = np.diag(covariances['B'])
    fused = (var_b * estimates['A'] + var_a * estimates['B']) / (var_a + var_b)
    fused_sigma = np.sqrt(1 / (1 / var_a + 1 / var_b))

    plt.figure(figsize=(11, 4))
    plt.grid(True)
    plt.xlabel("Time (sec)")
    plt.title('Residuals (m)')
    plt.scatter(times['A'], residuals['A'], color='blue', label='Tracker A')
    plt.scatter(times['B'], residuals['B'], color='red', label='Tracker B')
    plt.legend()
    plt.show()

    rows = [(f'Tracker {name}', estimates[name], np.sqrt(np.diag(covariances[name]))) for name in ('A', 'B')]
    rows.append(('sensor fusion', fused, fused_sigma))
    for title, x_hat, sigma in rows:
        print(f'--------------------- State estimation from {title} ---------------------------')
        print(f'Estimated x0 = {x_hat[0]} - {sigma[0]} m 1-sigma')
        print(f'Estimated v0 = {x_hat[1]} - {sigma[1]} m/sec 1-sigma')
        print(' ')


def static_estimator(true_temp, one_sigma, n_meas, initial_guess):
    """Recursive average of noisy measurements of a constant temperature."""
    measured = true_temp + one_sigma * np.random.randn(n_meas)

    estimates = np.empty(n_meas + 1)
    estimates[0] = initial_guess
    for m in range(1, n_meas + 1):
        # the initial guess counts as the first sample, hence the 1/(m+1) gain
        estimates[m] = estimates[m - 1] + (measured[m - 1] - estimates[m - 1]) / (m + 1)

    n = np.arange(1, n_meas + 1)
    plt.figure(figsize=(15, 5))
    plt.plot(n, measured, 'ro')
    plt.plot(estimates, color='blue', linewidth=3, label='Estimated value', linestyle='--')
    plt.plot(n, np.full(n_meas, true_temp), color='green', label='True value')
    plt.xlabel('N measurements')
    plt.ylabel('Temperature (deg)')
    plt.xticks(np.arange(1, n_meas, 1.0))
    plt.grid(True)
    plt.legend()
    plt.show()


def kalman_1d(true_temp, one_sigma, n_meas, initial_guess, initial_uncertainty):
    """One-dimensional Kalman filter estimating a constant temperature."""
    estimates = np.empty(n_meas + 1)
    uncertainties = np.empty(n_meas + 1)
    gains = np.empty(n_meas)
    estimates[0] = initial_guess
    uncertainties[0] = initial_uncertainty

    measurements = true_temp + one_sigma * np.random.randn(n_meas)
    p = initial_uncertainty**2  # state variance
    r = one_sigma**2            # measurement variance
    state = initial_guess

    for k, y in enumerate(measurements, 1):
        # update (the prediction is trivial: the temperature is constant)
        gain = p / (p + r)
        state = state + gain * (y - state)
        p = (1 - gain) * p

        estimates[k] = state
        gains[k - 1] = gain
        uncertainties[k] = math.sqrt(p)

    fig = plt.figure(figsize=(12, 10))
    ax1, ax2, ax3 = (fig.add_subplot(3, 1, i) for i in (1, 2, 3))
    n0 = np.arange(0, n_meas + 1)
    ax1.plot(n0[1:], measurements, 'ro', label='Measured value')
    ax1.plot(n0[1:], np.full(n_meas, true_temp), color='green', label='True value')
    ax1.plot(n0, estimates, color='blue', linewidth=3, label='Estimated value', linestyle='--', marker='o', markersize=7)
    ax1.set_xticks(n0)
    ax2.plot(n0, uncertainties, color='black', linestyle='--', label='Estimated value 1-sigma uncertainty', linewidth=2.0)
    ax2.set_xticks(n0)
    ax3.plot(n0[:-1], gains, color='blueviolet', linestyle='--', label='Kalman gain', linewidth=2.0)
    ax3.set_xticks(n0[:-1])
    for ax in (ax1, ax2, ax3):
        ax.legend()
        ax.grid(True)
    plt.xlabel('N measurements')
    plt.show()


def kalman_process_noise(init_temp, gradient_temp, one_sigma, n_meas, time_step, initial_guess,
                         initial_uncertainty, w, q):
    """1D Kalman filter with a constant-temperature model tracking a temperature that grows linearly.

    w: standard deviation of the random-walk process noise added to the true temperature at each step
    q: process noise variance assumed by the filter
    """
    estimates = np.empty(n_meas)
    uncertainties = np.empty(n_meas)
    gains = np.empty(n_meas)

    # truth (linear trend + random walk) and measurements
    times = np.arange(0, n_meas * time_step, time_step)
    true_values = init_temp + times * gradient_temp + np.cumsum(w * np.random.randn(n_meas))
    measurements = true_values + one_sigma * np.random.randn(n_meas)

    p = initial_uncertainty**2  # state variance
    r = one_sigma**2            # measurement variance
    state = initial_guess

    for k, y in enumerate(measurements):
        # update
        gain = p / (p + r)
        estimates[k] = state
        uncertainties[k] = math.sqrt(p)
        gains[k] = gain
        state = state + gain * (y - state)
        p = (1 - gain) * p

        # predict: constant model, the uncertainty grows by the process noise
        p = p + q

    n = np.arange(1, n_meas + 1)
    high = estimates + 3 * uncertainties
    low = estimates - 3 * uncertainties

    fig = plt.figure(figsize=(12, 9))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax1.plot(n, measurements, color='red', marker='x', label='Measured value')
    ax1.plot(n, low, color='orange', linewidth=1, label='3-sigma confidence')
    ax1.plot(n, high, color='orange', linewidth=1)
    ax1.plot(n, true_values, color='green', label='True value')
    ax1.plot(n, estimates, color='blue', linewidth=2, label='Estimated value', linestyle='--', marker='o', markersize=4)
    ax1.fill_between(n, low, high, color='khaki')
    ax1.legend()
    ax1.set_xticks(n)
    ax1.grid(True)

    ax2.plot(n, gains, color='blueviolet', linestyle='--', label='Kalman gain', linewidth=2.0)
    ax2.plot(n, uncertainties, color='black', linestyle='--', label='1-sigma uncertainty', linewidth=2.0)
    ax2.legend()
    ax2.set_xticks(n)
    ax2.grid(True)
    ax2.set_xlabel('N measurements')
    for ax in (ax1, ax2):
        ax.tick_params(axis='both', which='major', labelsize=10)
    plt.show()


def ekf_step(z, update_number, dT, x_unc, y_unc, vx_unc, vy_unc):
    """One step of an extended Kalman filter tracking a ship from range and bearing measurements.

    z: [ranges (m), bearings (deg), measurement covariances, ...]; the bearing is measured
       from the y axis, atan2(x, y). The filter state is stored between calls as attributes
       of this function; update_number = 0 restarts the filter.
    Returns [x, y, P, vx, vy, K, residual].
    """
    j = update_number
    rng, bearing, R = z[0][j], z[1][j], z[2][j]
    meas_x = rng * np.sin(np.radians(bearing))
    meas_y = rng * np.cos(np.radians(bearing))
    P0 = np.diag([x_unc**2, y_unc**2, vx_unc**2, vy_unc**2])
    residual = np.zeros((2, 1))
    K = np.zeros((4, 2))

    if j == 0:
        # initialize the position from the first measurement
        ekf_step.x = np.array([[meas_x], [meas_y], [0.0], [0.0]])
        ekf_step.P = P0
    elif j == 1:
        # re-initialize with the velocity from the first two measurements
        vx = (meas_x - ekf_step.x[0, 0]) / dT
        vy = (meas_y - ekf_step.x[1, 0]) / dT
        ekf_step.x = np.array([[meas_x], [meas_y], [vx], [vy]])
        ekf_step.P = P0
    else:
        # constant velocity model with white-noise acceleration
        A = np.array([[1, 0, dT, 0],
                      [0, 1, 0, dT],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        sd = 1  # spectral density
        Q = sd * np.array([[dT**3 / 3, 0, dT**2 / 2, 0],
                           [0, dT**3 / 3, 0, dT**2 / 2],
                           [dT**2 / 2, 0, dT, 0],
                           [0, dT**2 / 2, 0, dT]])

        # predict
        x_p = A @ ekf_step.x
        P_p = A @ ekf_step.P @ A.T + Q

        # linearized measurement model; the bearing row is in deg/m like the measurement
        px, py = x_p[0, 0], x_p[1, 0]
        r2 = px**2 + py**2
        r = np.sqrt(r2)
        H = np.array([[px / r, py / r, 0, 0],
                      [np.degrees(py / r2), np.degrees(-px / r2), 0, 0]])

        # update
        S = H @ P_p @ H.T + R
        K = P_p @ H.T @ np.linalg.inv(S)
        predicted = np.array([[r], [np.degrees(np.arctan2(px, py))]])
        residual = np.array([[rng], [bearing]]) - predicted
        residual[1, 0] = (residual[1, 0] + 180) % 360 - 180  # wrap the bearing residual
        ekf_step.x = x_p + K @ residual
        ekf_step.P = P_p - K @ H @ P_p

    x = ekf_step.x
    return [x[0], x[1], ekf_step.P, x[2], x[3], K, residual]


def kalman_free_fall(initial_pos, initial_vel, sigma_meas, n_meas, time_step, estimated_pos, estimated_vel,
                     sigma_pos, sigma_vel, sp):
    """Kalman filter estimating height and velocity of a falling object from height measurements.

    sp is the spectral density of the white-noise acceleration used for the process noise.
    """
    g = 9.81
    meas_times = np.linspace(time_step, n_meas * time_step, n_meas)
    true_pos = initial_pos + initial_vel * meas_times - 0.5 * g * meas_times**2
    true_vel = initial_vel - g * meas_times
    measurements = true_pos + sigma_meas * np.random.randn(n_meas)

    A = np.array([[1, time_step],
                  [0, 1]])
    G_u = -g * np.array([[time_step**2 / 2],
                         [time_step]])
    H = np.array([[1, 0]])
    R = np.array([[sigma_meas**2]])
    Q = sp * np.array([[time_step**3 / 3, time_step**2 / 2],
                       [time_step**2 / 2, time_step]])
    x = np.array([[estimated_pos], [estimated_vel]])
    P = np.diag([sigma_pos**2, sigma_vel**2])

    pos_est, vel_est, pos_sig, vel_sig = (np.empty(n_meas) for _ in range(4))
    for k, y in enumerate(measurements):
        # predict
        x_p = A @ x + G_u
        P_p = A @ P @ A.T + Q
        # update
        S = H @ P_p @ H.T + R
        K = P_p @ H.T @ np.linalg.inv(S)
        x = x_p + K @ (y - H @ x_p)
        P = P_p - K @ H @ P_p

        pos_est[k], vel_est[k] = x[0, 0], x[1, 0]
        pos_sig[k], vel_sig[k] = math.sqrt(P[0, 0]), math.sqrt(P[1, 1])

    def plot_estimate(est, sig, truth, est_label, true_label, est_color, true_color, ylabel, meas=None):
        plt.figure(figsize=(11.5, 3.5))
        plt.plot(meas_times, est, color=est_color, linewidth=2, label=est_label, linestyle='--', marker='o', markersize=7)
        plt.plot(meas_times, truth, color=true_color, marker='x', label=true_label)
        plt.plot(meas_times, est - 3 * sig, color='orange', label='3-sigma confidence')
        plt.plot(meas_times, est + 3 * sig, color='orange')
        plt.fill_between(meas_times, est - 3 * sig, est + 3 * sig, color='khaki')
        if meas is not None:
            plt.scatter(meas_times, meas, color='cyan', label='Measurements')
        plt.legend()
        plt.ylabel(ylabel)
        plt.xlabel('Time (sec)')
        plt.grid(True)
        plt.show()

    plot_estimate(pos_est, pos_sig, true_pos, 'Estimated pos', 'True position', 'blue', 'fuchsia', 'Height (m)', measurements)
    plot_estimate(vel_est, vel_sig, true_vel, 'Estimated vel', 'True velocity', 'black', 'deeppink', 'Velocity (m/sec)')

    fig = plt.figure(figsize=(11.5, 4))
    for i, (sig, label) in enumerate([(pos_sig, 'position uncertainty'), (vel_sig, 'velocity uncertainty')], 1):
        ax = fig.add_subplot(1, 2, i)
        ax.plot(meas_times, sig, color='black', marker='x', label=label)
        ax.legend()
        ax.grid(True)
    plt.show()

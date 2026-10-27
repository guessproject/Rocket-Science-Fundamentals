"""Statistics, stochastic processes and least squares examples used by the Observation and Estimation notebooks."""
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

########################################################## STATISTICS #############################################################

def plot_pdf(a, b):
    """PDF f(x) = 3x^2 on [0, 1], shading the probability of a <= x <= b."""
    print(f'Probability = {round(b**3 - a**3, 3)}')
    plt.figure(figsize=(10, 3))
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    x = np.linspace(0, 1, 100)
    y = 3 * x * x
    plt.plot(x, y, '-r', label='PDF')
    plt.fill_between(x, y, where=(x >= a) & (x <= b), color='gray')
    plt.legend()


def probability_from_pdf(true_distance, systematic_error, one_sigma, x_min, x_max):
    """Probability that a normally distributed measurement falls between x_min and x_max."""
    mean = true_distance + systematic_error
    x = np.linspace(mean - 4 * one_sigma, mean + 4 * one_sigma, 100)

    plt.figure(figsize=(11, 5))
    plt.plot(x, norm.pdf(x, mean, one_sigma), label='Normal Distribution', linewidth=3, color='green', linestyle='--')
    plt.fill_between(x, norm.pdf(x, mean, one_sigma), where=(x >= x_min) & (x <= x_max), color='khaki')
    plt.xlabel('Measure')
    plt.ylabel('f(x)')
    plt.title('Probability density function')
    plt.grid(True)
    plt.legend()

    prob = norm.cdf(x_max, mean, one_sigma) - norm.cdf(x_min, mean, one_sigma)
    print(f'The probability to get a measure between {x_min} and {x_max} is {prob * 100} %')


def plot_normal_distribution(mean, sigma, n_sigma):
    """Normal PDF with the +/- n_sigma interval shaded, and its probability."""
    x = np.linspace(mean - 4 * sigma, mean + 4 * sigma, 100)
    low, high = mean - n_sigma * sigma, mean + n_sigma * sigma

    prob = norm.cdf(n_sigma) - norm.cdf(-n_sigma)
    print(f'The probability to get a measure between {low} and {high} is {prob * 100} %')

    plt.figure(figsize=(11, 5))
    plt.plot(x, norm.pdf(x, mean, sigma), label='PDF', linewidth=3, color='dimgray')
    plt.axvline(x=low, color='r', linestyle='--', label=f'{n_sigma} sigma')
    plt.axvline(x=high, color='r', linestyle='--')
    plt.grid(True)
    plt.xlabel("Measure")
    plt.ylabel("f(x)")
    plt.legend()
    plt.fill_between(x, norm.pdf(x, mean, sigma), where=(x >= low) & (x <= high), color='khaki')
    plt.title('Normal Distribution')
    plt.show()


def data_fusion(mean1, sigma1, mean2, sigma2):
    """Fuse two normal distributions into the minimum-variance estimate."""
    var1 = sigma1**2
    var2 = sigma2**2
    new_mean = (var2 * mean1 + var1 * mean2) / (var2 + var1)
    new_sigma = math.sqrt(1 / (1 / var2 + 1 / var1))

    plt.figure(figsize=(11, 5))
    for mean, sigma, label, color, width in [(mean1, sigma1, 'Distribution 1', 'dimgray', 1.5),
                                             (mean2, sigma2, 'Distribution 2', 'darkgoldenrod', 1.5),
                                             (new_mean, new_sigma, 'Resulting distribution', 'darkorange', 4)]:
        x = np.linspace(mean - 4 * sigma, mean + 4 * sigma, 100)
        plt.plot(x, norm.pdf(x, mean, sigma), label=label, color=color, linewidth=width)
    plt.grid(True)
    plt.xlabel("Mean")
    plt.legend()
    plt.title('Data Fusion')
    plt.show()

    data = [["Distribution 1", mean1, sigma1], ["Distribution 2", mean2, sigma2], ["Fusion", new_mean, new_sigma]]
    print(pd.DataFrame(data, columns=["Source", "Mean", "Sigma"]))


def plot_xy_scatter(x_mean, x_sigma, y_mean, y_sigma, n_population):
    """Scatter of two independent normal variables with the sample 3-sigma covariance ellipse."""
    x = np.random.normal(x_mean, x_sigma, n_population)
    y = np.random.normal(y_mean, y_sigma, n_population)
    cov = np.cov(x, y)
    rho = np.corrcoef(x, y)

    # 3-sigma ellipse from the eigen-decomposition of the sample covariance
    eigval, eigvec = np.linalg.eigh(cov)
    angles = np.linspace(0, 2 * np.pi, 100)
    ellipse = eigvec @ (3 * np.sqrt(eigval)[:, None] * np.array([np.cos(angles), np.sin(angles)]))

    plt.figure(figsize=(10, 5))
    plt.subplot(111, aspect='equal')
    plt.scatter(x, y, color='darkorange')
    plt.plot(x.mean() + ellipse[0], y.mean() + ellipse[1], color='green', linewidth=3, linestyle='--',
             label='3-σ confidence level')
    plt.grid(True)
    plt.xlabel('m')
    plt.ylabel('m')
    plt.legend(loc='upper left')

    print(f'X standard deviation (m) = {math.sqrt(cov[0][0])}')
    print(f'Y standard deviation (m) = {math.sqrt(cov[1][1])}')
    print(f'Correlation coefficient  = {rho[0][1]}')


def plot_autocorrelation(dataset1, dataset2, n_population, lag):
    """Normalized autocorrelation, for lags 0 .. lag-1, of two datasets of n_population samples."""
    def acf(data):
        centered = data - np.mean(data)
        full = np.correlate(centered, centered, 'full')
        return full[n_population - 1:n_population - 1 + lag] / (n_population * np.var(data))

    plt.figure(figsize=(11, 4))
    plt.grid(True)
    plt.xlabel("Data")
    plt.plot(dataset1, 'lightgray', label='Random variables')
    plt.plot(acf(dataset1), 'darkgray', label='Autocorrelation of random variables')
    plt.plot(dataset2, 'lightsalmon', label='Sin(x)')
    plt.plot(acf(dataset2), 'orangered', label='Autocorrelation of Sin(x)')
    plt.legend()

###################################################### STOCHASTIC PROCESSES #######################################################

def plot_measurement_model(n_meas, systematic_error, modeled_temp, one_sigma, white_noise):
    """Simulated temperature measurements, their residuals and histogram."""
    x = np.arange(1, n_meas + 1)
    measured = (modeled_temp + one_sigma * np.random.randn(n_meas)
                + white_noise * np.random.randn(n_meas) + systematic_error)
    residuals = measured - modeled_temp

    plt.figure(figsize=(10, 3))
    plt.scatter(x, measured, color='red', s=6)
    plt.title('Measured Temperatures')
    plt.ylabel('deg')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 3))
    plt.scatter(x, residuals, color='purple', s=6)
    plt.title('Residuals')
    plt.ylabel('deg')
    plt.grid(True)
    plt.show()

    x_min = modeled_temp - 4 * one_sigma
    x_max = modeled_temp + 4 * one_sigma
    x = np.linspace(x_min, x_max, 100)
    normal = norm.pdf(x, modeled_temp, one_sigma)
    plt.figure(figsize=(10, 5))
    plt.hist(measured, 50, density=1, facecolor='g', alpha=0.75, edgecolor='black')
    plt.plot(x, normal, label='Normal Distribution', linewidth=3, color='red', linestyle='--')
    plt.xlabel('Range')
    plt.ylabel('f(x)')
    plt.title('Histogram')
    plt.axis([x_min, x_max, 0, normal.max()])
    plt.grid(True)
    plt.legend()


def plot_random_walk(n_steps, random_seed):
    """Two-dimensional random walk on a grid."""
    np.random.seed(random_seed)
    steps = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    moves = steps[np.random.choice(4, n_steps - 1)]
    path = np.vstack(([0, 0], np.cumsum(moves, axis=0)))
    x, y = path[:, 0], path[:, 1]

    fig, axs = plt.subplots(3, 1, figsize=(9, 9))
    axs[0].plot(x, y, 'orangered')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_title('x vs y')
    axs[0].set_aspect('equal')
    for ax, values, name in [(axs[1], x, 'x'), (axs[2], y, 'y')]:
        ax.plot(range(n_steps), values)
        ax.set_xlabel('Time')
        ax.set_ylabel(name)
        ax.set_title(f'{name} evolution')
        ax.set_xticks(np.arange(0, n_steps + 1, math.ceil(n_steps / 20)))
    for ax in axs:
        ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_brownian_motion(init_value, T, N):
    """Brownian motion (Wiener process) over a duration T with N steps."""
    dt = T / N
    B = init_value + np.concatenate(([0.0], np.cumsum(np.random.normal(0, np.sqrt(dt), N))))

    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, T, N + 1), B, 'darkgreen')
    plt.xlabel('Time')
    plt.ylabel('B(t)')
    plt.title('Brownian motion process')
    plt.grid(True)
    plt.show()


def plot_exponential_decay(init_value, T, N, tau, sigma):
    """First-order Gauss-Markov process with time constant tau and steady-state standard deviation sigma."""
    dt = T / N
    phi = math.exp(-dt / tau)
    X = np.zeros(N + 1)
    X[0] = init_value
    for i in range(1, N + 1):
        X[i] = X[i - 1] * phi + np.random.normal(0, sigma) * math.sqrt(1 - phi**2)

    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, T, N + 1), X, 'brown')
    plt.xlabel('Time')
    plt.ylabel('X(t)')
    plt.title('Exponential decay process')
    plt.grid(True)
    plt.show()

######################################################### LEAST SQUARES ###########################################################

def plot_height_weight(df):
    """Scatter plot of the height/weight dataset."""
    data = df.to_numpy()
    plt.figure(figsize=(10, 4))
    plt.grid(True)
    plt.xlabel("Height (cm)")
    plt.ylabel("Weight (kg)")
    plt.scatter(data[:, 0], data[:, 1])


def _fit_and_plot(df, degree, label):
    data = df.to_numpy()
    x, y = data[:, 0], data[:, 1]
    m = len(x)

    # normal equation with H = [1, x, x^2, ...]
    H = np.vander(x, degree + 1, increasing=True)
    coeffs = np.linalg.solve(H.T @ H, H.T @ y)
    for i, c in enumerate(coeffs, 1):
        print(f'Best x{i} = {c}')
    cost = np.sum((H @ coeffs - y)**2) / (2 * m)
    print(f'Cost function value is:{round(cost, 2)}')

    _plot_model(x, y, coeffs, label)


def _plot_model(x, y, coeffs, label):
    plt.figure(figsize=(10, 4))
    plt.grid(True)
    plt.xlabel("Height (cm)")
    plt.ylabel("Weight (kg)")
    plt.scatter(x, y)
    xs = np.linspace(x.min(), x.max(), 115)
    plt.plot(xs, np.polyval(coeffs[::-1], xs), '-r', label=label)
    plt.legend()


def linear_least_squares(df):
    """Fit weight = x1 + x2 * height with the normal equation."""
    _fit_and_plot(df, 1, 'Linear Model (straight line)')


def quadratic_least_squares(df):
    """Fit weight = x1 + x2 * height + x3 * height^2 with the normal equation."""
    _fit_and_plot(df, 2, 'Quadratic Model (parabolic line)')


def plot_linear_cost_function(df, a, b):
    """Cost function of the line weight = a + b * height on the height/weight dataset."""
    data = df.to_numpy()
    x, y = data[:, 0], data[:, 1]
    cost = np.sum((a + b * x - y)**2) / (2 * len(x))
    print(f'Cost function value is:{round(cost, 2)}')
    _plot_model(x, y, [a, b], 'Linear Model (straight line)')

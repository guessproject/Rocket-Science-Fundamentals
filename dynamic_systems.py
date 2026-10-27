"""Examples used by the Dynamic Systems notebooks."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from algorithms import euler, rk4, rk45

##################################################### NUMERICAL INTEGRATION #######################################################

def _free_fall(g):
    def f(t, y):
        h, v = y
        return np.array([v, -g])
    return f


def _free_fall_exact(init_state, g, t):
    h0, v0 = init_state
    return h0 + v0 * t - 0.5 * g * t**2


def euler_vs_rk4_free_fall(init_state, g, step_size, time_span):
    """Compare Euler and RK4 against the exact free-fall height. init_state = [h0 (m), v0 (m/s)]."""
    t, euler_state = euler(_free_fall(g), time_span, init_state, dt=step_size)
    t, rk4_state = rk4(_free_fall(g), time_span, init_state, dt=step_size)
    t_exact = np.linspace(time_span[0], t[-1], 100)

    plt.figure(figsize=(12, 4))
    plt.plot(t, euler_state[:, 0], 'o--', markersize=6, label="Euler")
    plt.plot(t, rk4_state[:, 0], 'o--', markersize=6, label="RK4")
    plt.plot(t_exact, _free_fall_exact(init_state, g, t_exact), label="Exact")
    plt.xlabel("Time (s)")
    plt.ylabel("Height (m)")
    plt.title(f"Free Fall - Height with step size = {step_size} sec.")
    plt.grid(True)
    plt.legend()
    plt.show()

    h_final = _free_fall_exact(init_state, g, t[-1])
    print("Euler final error: ", abs(euler_state[-1, 0] - h_final))
    print("RK4 final error:   ", abs(rk4_state[-1, 0] - h_final))


def rk45_free_fall(init_state, g, time_span, max_step_size):
    """Free fall integrated with the adaptive RK45 method. init_state = [h0 (m), v0 (m/s)]."""
    t, rk45_state = rk45(_free_fall(g), time_span, init_state, rtol=1e-6, atol=1e-9, h_max=max_step_size)
    t_exact = np.linspace(time_span[0], t[-1], 100)

    plt.figure(figsize=(12, 4))
    plt.plot(t, rk45_state[:, 0], 'o--', markersize=6, label="RK45")
    plt.plot(t_exact, _free_fall_exact(init_state, g, t_exact), label="Exact")
    plt.xlabel("Time (s)")
    plt.ylabel("Height (m)")
    plt.title("Free Fall - Height with variable step size.")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_sma_vs_delta_v():
    """Semimajor axis reached with a tangential Delta V from a 6900 km circular orbit."""
    r = 6900                 # km
    v0 = 7.6                 # km/s
    mu = 398600.4418         # km^3/s^2
    delta_v = np.linspace(0, 4., 1000)
    sma = 1 / ((2 / r) - ((v0 + delta_v)**2 / mu))

    plt.figure(figsize=(12, 4))
    plt.plot(delta_v, sma, lw=2)
    plt.xlabel('Delta V [km/s]')
    plt.ylabel('SMA [km]')
    plt.ylim(-45000, 45000)
    plt.title('Semi-major Axis vs Delta V')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

########################################################### STABILITY #############################################################

def _plot_time_history_and_phase(solution, labels, phase_labels, phase_title="Phase portrait", time_title="Time history"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for values, label in zip(solution.y, labels):
        axes[0].plot(solution.t, values, label=label)
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("State")
    axes[0].set_title(time_title)
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(solution.y[0], solution.y[1])
    axes[1].plot(0.0, 0.0, "ro", label="Equilibrium")
    axes[1].set_xlabel(phase_labels[0])
    axes[1].set_ylabel(phase_labels[1])
    axes[1].set_title(phase_title)
    axes[1].grid(True)
    axes[1].legend()
    plt.tight_layout()
    plt.show()


def solve_oscillator(m, k, c):
    """Mass-spring-damper x'' = -(k/m) x - (c/m) x' starting from x = 1, v = 0."""
    A = np.array([[0.0, 1.0],
                  [-k / m, -c / m]])
    print("System matrix:")
    print(A)
    print("\nEigenvalues:")
    print(np.linalg.eigvals(A))

    t_span = (0.0, 20.0)
    solution = solve_ivp(lambda t, s: A @ s, t_span, [1.0, 0.0],
                         t_eval=np.linspace(*t_span, 2000), rtol=1e-9, atol=1e-12)
    _plot_time_history_and_phase(solution, ["Position", "Velocity"], ["Position x", "Velocity v"])


def solve_saddle_system():
    """Linear system with a saddle point at the origin."""
    A = np.array([[1.0, 0.0],
                  [0.0, -2.0]])
    print("Eigenvalues:")
    print(np.linalg.eigvals(A))

    t_span = (0.0, 4.0)
    solution = solve_ivp(lambda t, s: A @ s, t_span, [0.05, 1.0],
                         t_eval=np.linspace(*t_span, 500), rtol=1e-9, atol=1e-12)
    _plot_time_history_and_phase(solution, ["x", "y"], ["x", "y"])


def solve_vanderpol(mu, x0, v0):
    """Van der Pol oscillator x'' = mu (1 - x^2) x' - x."""
    def van_der_pol(t, state):
        x, v = state
        return [v, mu * (1.0 - x**2) * v - x]

    A = np.array([[0.0, 1.0],
                  [-1.0, mu]])
    print("Eigenvalues at the origin:")
    print(np.linalg.eigvals(A))

    t_span = (0.0, 30.0)
    solution = solve_ivp(van_der_pol, t_span, [x0, v0], t_eval=np.linspace(*t_span, 5000),
                         method="DOP853", rtol=1e-9, atol=1e-11)
    _plot_time_history_and_phase(solution, [f"x (x0={x0})", f"v (v0={v0})"], ["x", "v"],
                                 phase_title="Convergence toward a limit cycle",
                                 time_title="Van der Pol time histories")

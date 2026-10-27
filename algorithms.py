"""Root finders and ODE integrators used by the Dynamic Systems notebooks."""
import numpy as np


def _rel_error(new, old):
    return abs(new - old) / max(abs(new), np.finfo(float).tiny)


def bisection(f, a, b, tol=1e-8, max_iter=100):

    fa = f(a)
    if fa * f(b) >= 0:
        raise ValueError(
            "The interval does not bracket a root "
            "(f(a) and f(b) must have opposite signs)."
        )

    print(
        f"{'Iter':>4} | {'a':>12} | {'b':>12} | "
        f"{'c':>12} | {'f(c)':>12} | {'Rel.Error':>12}"
    )
    print("-" * 80)

    c_old = None

    for k in range(1, max_iter + 1):

        c = 0.5 * (a + b)
        fc = f(c)

        if c_old is None:
            rel_err = float("nan")
        else:
            rel_err = _rel_error(c, c_old)

        print(
            f"{k:4d} | "
            f"{a:12.6f} | "
            f"{b:12.6f} | "
            f"{c:12.6f} | "
            f"{fc:12.6e} | "
            f"{rel_err:12.6e}"
        )

        # Convergence tests
        if abs(fc) < tol:
            print("\nConverged on function tolerance.")
            return c

        if c_old is not None and rel_err < tol:
            print("\nConverged on relative error.")
            return c

        # Update interval
        if fa * fc < 0:
            b = c
        else:
            a, fa = c, fc

        c_old = c

    raise RuntimeError("Maximum number of iterations reached.")

def newton(f, df, x0, tol=1e-8, max_iter=100):

    print(
        f"{'Iter':>4} | {'x':>14} | "
        f"{'f(x)':>14} | {'Rel.Error':>14}"
    )
    print("-" * 60)

    x = x0

    for k in range(1, max_iter + 1):

        fx = f(x)
        dfx = df(x)

        if abs(dfx) < 1e-14:
            raise ZeroDivisionError(
                f"Derivative nearly zero at x = {x}"
            )

        x_new = x - fx / dfx

        rel_err = _rel_error(x_new, x)

        print(
            f"{k:4d} | "
            f"{x_new:14.10f} | "
            f"{f(x_new):14.6e} | "
            f"{rel_err:14.6e}"
        )

        if abs(f(x_new)) < tol:
            print("\nConverged on function tolerance.")
            return x_new

        if rel_err < tol:
            print("\nConverged on relative error.")
            return x_new

        x = x_new

    raise RuntimeError("Maximum number of iterations reached")

def secant(f, x0, x1, tol=1e-8, max_iter=100):

    print(
        f"{'Iter':>4} | {'x':>14} | "
        f"{'f(x)':>14} | {'Rel.Error':>14}"
    )
    print("-" * 60)

    for k in range(1, max_iter + 1):

        f0 = f(x0)
        f1 = f(x1)

        denominator = (f1 - f0)

        if abs(denominator) < 1e-14:
            raise ZeroDivisionError(
                "f(x_k) - f(x_k-1) is too small."
            )

        x2 = x1 - f1 * (x1 - x0) / denominator

        rel_err = _rel_error(x2, x1)

        print(
            f"{k:4d} | "
            f"{x2:14.10f} | "
            f"{f(x2):14.6e} | "
            f"{rel_err:14.6e}"
        )

        if abs(f(x2)) < tol:
            print("\nConverged on function tolerance.")
            return x2

        if rel_err < tol:
            print("\nConverged on relative error.")
            return x2

        x0 = x1
        x1 = x2

    raise RuntimeError("Maximum number of iterations reached")

def rk45(f, t_span, y0, rtol=1e-6, atol=1e-9, h0=None, h_min=1e-12, h_max=np.inf):
    """
    Adaptive Runge-Kutta-Fehlberg (RK45) integrator.

    Parameters
    ----------
    f : function - Derivative function f(t, y).
    t_span : ndarray - Initial and final time.
    y0 : ndarray - Initial state.
    rtol : float - Relative tolerance.
    atol : float - Absolute tolerance.
    h0 : float or None - Initial step size. If None, estimated automatically.
    h_min : float - Minimum allowed step size.
    h_max : float - Maximum allowed step size.

    Returns
    -------
    t : ndarray
        Time points.
    y : ndarray - Solution values.
    """

    t0, tf = t_span
    t = t0
    y = np.asarray(y0, dtype=float)

    if h0 is None:
        h = (tf - t0) / 100.0
    else:
        h = h0
    h = min(h, h_max)

    ts = [t]
    ys = [y.copy()]

    # Fehlberg coefficients
    while tf - t > 1e-12 * max(1.0, abs(tf)):

        if t + h > tf:
            h = tf - t

        k1 = h * f(t, y)
        k2 = h * f(t + 1/4*h, y + 1/4*k1)
        k3 = h * f(t + 3/8*h, y + 3/32*k1 + 9/32*k2)
        k4 = h * f(t + 12/13*h, y + 1932/2197*k1 - 7200/2197*k2 + 7296/2197*k3)
        k5 = h * f(t + h, y + 439/216*k1 - 8*k2 + 3680/513*k3- 845/4104*k4)
        k6 = h * f(t + 1/2*h, y - 8/27*k1 + 2*k2 - 3544/2565*k3 + 1859/4104*k4 - 11/40*k5)

        # 4th-order solution
        y4 = (y + 25/216*k1 + 1408/2565*k3 + 2197/4104*k4 - 1/5*k5)

        # 5th-order solution
        y5 = (y + 16/135*k1 + 6656/12825*k3 + 28561/56430*k4 - 9/50*k5 + 2/55*k6)

        # Error estimate
        err = np.abs(y5 - y4)
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
        err_norm = np.max(err / scale)

        # Accept step
        if err_norm <= 1.0:
            t += h
            y = y5

            ts.append(t)
            ys.append(y.copy())

        # Step-size controller
        if err_norm == 0:
            factor = 5.0
        else:
            factor = 0.84 * err_norm**(-0.2)

        factor = np.clip(factor, 0.1, 5.0)
        h_new = h * factor
        if h_new < h_min and err_norm > 1.0:
            raise RuntimeError("Step size underflow.")
        h = min(h_new, h_max)

    return np.array(ts), np.array(ys)

def euler(f, t_span, y0, dt):
    """
    Generic Euler integrator for systems of ODEs.

    Parameters
    ----------
    f : function - Derivative function f(t, y).
    t_span : ndarray - Initial and final time.
    y0 : ndarray - Initial state.
    h : dt - Time step.

    Returns
    -------
    t : ndarray - Time points.
    y : ndarray - Solution values.
    """

    t0, tf = t_span
    nsteps = int(np.ceil((tf - t0) / dt))

    t = np.zeros(nsteps + 1)
    y = np.zeros((nsteps + 1, len(y0)))

    t[0] = t0
    y[0] = y0

    for n in range(nsteps):
        h = min(dt, tf - t[n])  # the last step stops exactly at tf
        y[n + 1] = y[n] + h * f(t[n], y[n])
        t[n + 1] = t[n] + h

    return t, y

def rk4(f, t_span, y0, dt):
    """
    4th-order Runge-Kutta solver.

    Parameters
    ----------
    f : function - Derivative function f(t, y).
    t_span : ndarray - Initial and final time.
    y0 : ndarray - state.
    h : dt - Time step.

    Returns
    -------
    t : ndarray - Time points.
    y : ndarray - Solution values.
    """

    t0, tf = t_span
    nsteps = int(np.ceil((tf - t0) / dt))

    t = np.zeros(nsteps + 1)
    y = np.zeros((nsteps + 1, len(y0)))

    t[0] = t0
    y[0] = y0

    for i in range(nsteps):
        h = min(dt, tf - t[i])  # the last step stops exactly at tf
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + h*k1/2)
        k3 = f(t[i] + h/2, y[i] + h*k2/2)
        k4 = f(t[i] + h, y[i] + h*k3)

        y[i+1] = y[i] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        t[i+1] = t[i] + h

    return t, y



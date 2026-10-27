"""Gooding's algorithm for solving Lambert's problem.

R. H. Gooding, "On the Solution of Lambert's Orbital Boundary-Value Problem" (1988)
    https://apps.dtic.mil/sti/pdfs/ADA200383.pdf
ported from the Fortran implementation in
    https://github.com/jacobwilliams/Fortran-Astrodynamics-Toolkit
"""
import numpy as np


def solve(mu, r1, r2, t_sec, orb_type='prograde', m=0):
    """Solve Lambert's problem.

    Args:
        mu:       gravitational parameter (km^3/s^2)
        r1, r2:   initial and final position vectors (km)
        t_sec:    time of flight (s)
        orb_type: 'prograde' or 'retrograde'
        m:        number of complete revolutions
    Returns:
        v1_list, v2_list: velocities at r1 and r2 (km/s), one entry per solution.
        Zero-revolution transfers have a single solution; multi-revolution transfers
        can have two (the low- and high-energy paths).
    """
    h, psi = get_transfer_angle(orb_type, r1, r2)
    psi = m * 2.0 * np.pi + psi
    n, vr1_list, vt1_list, vr2_list, vt2_list = vlamb(mu, r1, r2, psi, t_sec)

    v1_list, v2_list = [], []
    for vr1, vt1, vr2, vt2 in zip(vr1_list, vt1_list, vr2_list, vt2_list):
        v1, v2 = calculate_velocities(h, r1, r2, vr1, vt1, vr2, vt2)
        v1_list.append(v1)
        v2_list.append(v2)
    return v1_list, v2_list


def get_transfer_angle(orb_type, r1, r2):
    """Unit angular momentum and transfer angle (rad) of the transfer orbit.

    See "The Superior Lambert Algorithm" (https://amostech.com/TechnicalPapers/2011/Poster/DER.pdf):
    for a prograde transfer psi = 2*pi - psi0 if alpha < 0, for a retrograde one if alpha > 0,
    alpha being the z component of r1 x r2.
    """
    r1_n = np.linalg.norm(r1)
    r2_n = np.linalg.norm(r2)
    h = np.cross(r1, r2) / np.linalg.norm(np.cross(r1, r2))
    alpha = h[2]
    psi = np.arccos(np.clip(np.dot(r1, r2) / (r1_n * r2_n), -1.0, 1.0))

    if (orb_type == 'prograde' and alpha < 0) or (orb_type == 'retrograde' and alpha >= 0):
        psi = 2 * np.pi - psi
        h = -h
    return h, psi


def calculate_velocities(h, r1, r2, vr1, vt1, vr2, vt2):
    """Velocity vectors at r1 and r2 from their radial and tangential components."""
    r1_n = np.linalg.norm(r1)
    r2_n = np.linalg.norm(r2)
    h_x_r1 = np.cross(h, r1)
    h_x_r2 = np.cross(h, r2)
    v1 = vr1 * (r1 / r1_n) + vt1 * (h_x_r1 / np.linalg.norm(h_x_r1))
    v2 = vr2 * (r2 / r2_n) + vt2 * (h_x_r2 / np.linalg.norm(h_x_r2))
    return v1, v2


def vlamb(mu, r1, r2, th, t_sec):
    """Gooding support routine VLAMB (appendix C).

    Args:
        mu:     gravitational parameter
        r1, r2: position vectors
        th:     transfer angle including complete revolutions (rad), th >= 0
        t_sec:  time of flight (s)
    Returns:
        n: number of solutions, then the lists of radial and tangential velocity
        components at r1 (vr1, vt1) and r2 (vr2, vt2)
    """
    # yields m = 0 when th = 2 pi exactly
    thr2 = th
    m = 0
    while thr2 > 2 * np.pi:
        thr2 = thr2 - 2 * np.pi
        m = m + 1
    thr2 = thr2 / 2.0

    r1_n = np.linalg.norm(r1)
    r2_n = np.linalg.norm(r2)

    dr = r1_n - r2_n
    r1r2 = r1_n * r2_n
    r1r2th = 4.0 * r1r2 * np.sin(thr2)**2
    csq = dr**2 + r1r2th
    c = np.sqrt(csq)
    s = (r1_n + r2_n + c) / 2.0
    gms = np.sqrt(mu * s / 2.0)
    qsqfm1 = c / s
    q = np.sqrt(r1r2) * np.cos(thr2) / s

    if c != 0.0:
        rho = dr / c
        sig = r1r2th / csq
    else:
        rho = 0.0
        sig = 1.0

    t = 4.0 * gms * t_sec / s**2

    n, x1, x2 = xlamb(m, q, qsqfm1, t)

    vr1_list, vt1_list, vr2_list, vt2_list = [], [], [], []
    for i in range(n):
        x = x1 if i == 0 else x2

        _, qzminx, qzplx, zplqx = tlamb(m, q, qsqfm1, x, -1)

        vt2 = gms * zplqx * np.sqrt(sig)
        vr1 = gms * (qzminx - qzplx * rho) / r1_n
        vt1 = vt2 / r1_n
        vr2 = -gms * (qzminx + qzplx * rho) / r2_n
        vt2 = vt2 / r2_n

        vr1_list.append(vr1)
        vt1_list.append(vt1)
        vr2_list.append(vr2)
        vt2_list.append(vt2)

    return n, vr1_list, vt1_list, vr2_list, vt2_list


def d8rt(x):
    """Eighth root, used by xlamb."""
    return np.sqrt(np.sqrt(np.sqrt(x)))


def xlamb(m, q, qsqfm1, tin):
    """Gooding support routine XLAMB.

    Args:
        m:      number of complete revolutions
        q:      transfer geometry parameter
        qsqfm1: 1 - q^2
        tin:    non-dimensional time of flight
    Returns:
        n: number of solutions, x: first solution, xpl: second solution
    """
    tol = 3.0e-7
    xpl = 0
    c0, c1, c2, c3, c41, c42 = 1.7, 0.5, 0.03, 0.15, 1.0, 0.24
    thr2 = np.arctan2(qsqfm1, 2.0 * q) / np.pi

    goto3 = False

    if m == 0:
        # single-rev starter from t (at x = 0) & bilinear (usually)
        n = 1
        t0, dt, d2t, d3t = tlamb(m, q, qsqfm1, 0.0, 0)
        tdiff = tin - t0
        if tdiff <= 0.0:
            # (-4 is the value of dt, for x = 0)
            x = t0 * tdiff / (-4.0 * tin)
        else:
            x = -tdiff / (tdiff + 4.0)
            w = x + c0 * np.sqrt(2.0 * (1.0 - thr2))
            if w < 0.0:
                x = x - np.sqrt(d8rt(-w)) * (x + np.sqrt(tdiff / (tdiff + 1.5 * t0)))
            w = 4.0 / (4.0 + tdiff)
            x = x * (1.0 + x * (c1 * w - c2 * x * np.sqrt(w)))
    else:
        # with multirevs, first get t(min) as basis for starter
        xm = 1.0 / (1.5 * (m + 0.5) * np.pi)
        if thr2 < 0.5:
            xm = d8rt(2.0 * thr2) * xm
        if thr2 > 0.5:
            xm = (2.0 - d8rt(2.0 - 2.0 * thr2)) * xm

        # Halley iterations for tmin
        for _ in range(12):
            tmin, dt, d2t, d3t = tlamb(m, q, qsqfm1, xm, 3)
            if d2t == 0.0:
                break
            xmold = xm
            xm = xm - dt * d2t / (d2t * d2t - dt * d3t / 2.0)
            if np.abs(xmold / xm - 1.0) <= tol:
                break
        else:
            # tmin not located - should never happen
            return -1, None, None

        tdiffm = tin - tmin
        if tdiffm < 0.0:
            raise ValueError(f"No Lambert solution with {m} revolutions for this time of flight.")
        elif tdiffm == 0.0:
            # unique solution already from x(tmin)
            return 1, xm, None
        else:
            n = 3
            if d2t == 0.0:
                d2t = 6.0 * m * np.pi
            x = np.sqrt(tdiffm / (d2t / 2.0 + tdiffm / (1.0 - xm)**2))
            w = xm + x
            w = w * 4.0 / (4.0 + tdiffm) + (1.0 - w)**2
            x = (x * (1.0 - (1.0 + m + c41 * (thr2 - 0.5))
                      / (1.0 + c3 * m) * x * (c1 * w + c2 * x * np.sqrt(w))) + xm)
            d2t2 = d2t / 2.0
            if x >= 1.0:
                # no finite solution with x > xm
                n = 1
                goto3 = True

    # now have a starter, so proceed by Halley
    while True:
        if not goto3:
            for _ in range(3):
                t, dt, d2t, d3t = tlamb(m, q, qsqfm1, x, 2)
                t = tin - t
                if dt != 0.0:
                    x = x + t * dt / (dt * dt + t * d2t / 2.0)
            # exit if only one solution, normally when m = 0
            if n != 3:
                return n, x, xpl
            n = 2
            xpl = x
        else:
            goto3 = False

        # second multi-rev starter
        t0, dt, d2t, d3t = tlamb(m, q, qsqfm1, 0.0, 0)
        tdiff0 = t0 - tmin
        tdiff = tin - t0
        if tdiff <= 0.0:
            x = xm - np.sqrt(tdiffm / (d2t2 - tdiffm * (d2t2 / tdiff0 - 1.0 / xm**2)))
        else:
            x = -tdiff / (tdiff + 4.0)
            w = x + c0 * np.sqrt(2.0 * (1.0 - thr2))
            if w < 0.0:
                x = x - np.sqrt(d8rt(-w)) * (x + np.sqrt(tdiff / (tdiff + 1.5 * t0)))
            w = 4.0 / (4.0 + tdiff)
            x = x * (1.0 + (1.0 + m + c42 * (thr2 - 0.5)) / (1.0 + c3 * m) * x * (c1 * w - c2 * x * np.sqrt(w)))
            if x <= -1.0:
                # no finite solution with x < xm
                n = n - 1
                if n == 1:
                    x = xpl


def tlamb(m, q, qsqfm1, x, n):
    """Gooding support routine TLAMB.

    Args:
        m:      number of complete revolutions
        q:      transfer geometry parameter
        qsqfm1: 1 - q^2
        x:      independent variable
        n:      number of derivatives to compute (-1 returns the velocity terms instead)
    Returns:
        t, dt, d2t, d3t: non-dimensional time of flight and its derivatives at x
    """
    sw = 0.4
    lm1 = n == -1
    l1 = n >= 1
    l2 = n >= 2
    l3 = n == 3
    qsq = q * q
    xsq = x * x
    u = (1.0 - x) * (1.0 + x)

    # needed if series, and otherwise useful when z = 0
    if not lm1:
        dt, d2t, d3t = 0.0, 0.0, 0.0

    if lm1 or m > 0 or x < 0.0 or np.abs(u) > sw:
        # direct computation (not series)
        y = np.sqrt(np.abs(u))
        z = np.sqrt(qsqfm1 + qsq * xsq)
        qx = q * x

        if qx <= 0.0:
            a = z - qx
            b = q * z - x
        if qx <= 0.0 and lm1:
            aa = qsqfm1 / a
            bb = qsqfm1 * (qsq * u - xsq) / b
        if (qx == 0.0 and lm1) or (qx > 0.0):
            aa = z + qx
            bb = q * z + x
        if qx > 0.0:
            a = qsqfm1 / aa
            b = qsqfm1 * (qsq * u - xsq) / bb

        if lm1:
            t, dt, d2t, d3t = 0, b, bb, aa
        else:
            if qx * u >= 0.0:
                g = x * z + q * u
            else:
                g = (xsq - qsq * u) / (x * z - q * u)
            f = a * y

            if x <= 1.0:
                t = m * np.pi + np.arctan2(f, g)
            elif f > sw:
                t = np.log(f + g)
            else:
                # series for the inverse tanh
                fg1 = f / (g + 1.0)
                term = 2.0 * fg1
                fg1sq = fg1 * fg1
                t = term
                twoi1 = 1.0
                while True:
                    twoi1 = twoi1 + 2.0
                    term = term * fg1sq
                    told = t
                    t = t + term / twoi1
                    if t == told:
                        break

            t = 2.0 * (t / y + b) / u

            if l1 and z != 0.0:
                qz = q / z
                qz2 = qz * qz
                qz = qz * qz2
                dt = (3.0 * x * t - 4.0 * (a + qx * qsqfm1) / z) / u
                if l2:
                    d2t = (3.0 * t + 5.0 * x * dt + 4.0 * qz * qsqfm1) / u
                if l3:
                    d3t = (8.0 * dt + 7.0 * x * d2t - 12.0 * qz * qz2 * x * qsqfm1) / u
    else:
        # compute by series
        u0i = 1.0
        if l1:
            u1i = 1.0
        if l2:
            u2i = 1.0
        if l3:
            u3i = 1.0

        term = 4.0
        tq = q * qsqfm1
        i = 0
        if q < 0.5:
            tqsum = 1.0 - q * qsq
        else:
            tqsum = (1.0 / (1.0 + q) + q) * qsqfm1
        ttmold = term / 3.0
        t = ttmold * tqsum

        while True:
            i = i + 1
            p = i
            u0i = u0i * u
            if l1 and i > 1:
                u1i = u1i * u
            if l2 and i > 2:
                u2i = u2i * u
            if l3 and i > 3:
                u3i = u3i * u

            term = term * (p - 0.5) / p
            tq = tq * qsq
            tqsum = tqsum + tq
            told = t
            tterm = term / (2.0 * p + 3.0)
            tqterm = tterm * tqsum
            t = t - u0i * ((1.5 * p + 0.25) * tqterm / (p * p - 0.25) - ttmold * tq)
            ttmold = tterm
            tqterm = tqterm * p

            if l1:
                dt = dt + tqterm * u1i
            if l2:
                d2t = d2t + tqterm * u2i * (p - 1.0)
            if l3:
                d3t = d3t + tqterm * u3i * (p - 1.0) * (p - 2.0)

            if not (i < n or t != told):
                break

        if l3:
            d3t = 8.0 * x * (1.5 * d2t - xsq * d3t)
        if l2:
            d2t = 2.0 * (2.0 * xsq * d2t - dt)
        if l1:
            dt = -2.0 * x * dt

        t = t / xsq

    return t, dt, d2t, d3t

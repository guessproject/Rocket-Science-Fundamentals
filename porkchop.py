"""Porkchop plots for interplanetary transfers."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import astrodynamics as ad
import lambert


def plot_porkchop(dep_planet, arr_planet, first_dep_date, last_dep_date, first_arr_date, last_arr_date,
                  plot_type, dep_step=2, arr_step=5):
    """Porkchop plot of a planet-to-planet transfer.

    Args:
        dep_planet, arr_planet:          departure and arrival planet names
        first_dep_date, last_dep_date:   departure window, dd-mm-yyyy
        first_arr_date, last_arr_date:   arrival window, dd-mm-yyyy
        plot_type:                       'delv_plot' (total Delta V) or 'c3_plot' (departure C3)
        dep_step, arr_step:              grid resolution (days)
    """
    if plot_type not in ('delv_plot', 'c3_plot'):
        raise ValueError("plot_type should be 'c3_plot' or 'delv_plot'")

    jd_dep_first, jd_dep_last, jd_arr_first, jd_arr_last = (
        ad.jd_from_date_string(d) for d in (first_dep_date, last_dep_date, first_arr_date, last_arr_date))
    if jd_arr_first < jd_dep_last:
        raise ValueError("The latest departure epoch is later than the earliest arrival epoch.")

    jd_dep = np.arange(jd_dep_first, jd_dep_last, dep_step)
    jd_arr = np.arange(jd_arr_first, jd_arr_last, arr_step)
    tof_days, c3_dep, delv_total = porkchop_data(dep_planet, arr_planet, jd_dep, jd_arr)

    if plot_type == 'delv_plot':
        title = ('Porkchop plot (∆V Total = ||$ΔV_{' + dep_planet + '}$|| + ||$ΔV_{' + arr_planet + '}$||, km/s)')
        _draw_porkchop(title, jd_dep, jd_arr, delv_total, [5, 5.5, 6, 6.5, 7, 7.5, 8, 9, 10, 12, 15, 20],
                       tof_days, [100, 150, 200, 250, 300, 350, 400, 450, 500, 550])
    else:
        title = 'Porkchop plot (C3 characteristic energy = $v_{\\infty}^{2}$, km$^2$/s$^2$)'
        _draw_porkchop(title, jd_dep, jd_arr, c3_dep, [4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 30, 50],
                       tof_days, [100, 150, 200, 250, 300, 350, 400, 450, 500, 550])


def porkchop_data(dep_planet, arr_planet, jd_dep, jd_arr):
    """Time of flight (days), departure C3 (km^2/s^2) and total Delta V (km/s) on a departure x arrival grid.

    Rows of the returned arrays correspond to arrival dates, columns to departure dates.
    Zero-revolution prograde transfers are considered: the single Lambert solution
    is Type I or Type II depending on the transfer angle.
    """
    mu_sun = ad.planet_mu('Sun')
    dep_states = [ad.planet_state_vector(dep_planet, jd) for jd in jd_dep]
    arr_states = [ad.planet_state_vector(arr_planet, jd) for jd in jd_arr]

    shape = (len(jd_arr), len(jd_dep))
    tof_days = np.zeros(shape)
    c3_dep = np.zeros(shape)
    delv_total = np.zeros(shape)
    for ia, (r_arr, v_arr) in enumerate(arr_states):
        for id_, (r_dep, v_dep) in enumerate(dep_states):
            tof = jd_arr[ia] - jd_dep[id_]
            v1_list, v2_list = lambert.solve(mu_sun, r_dep, r_arr, tof * 86400.0, 'prograde')
            v_inf_dep = np.linalg.norm(v1_list[0] - v_dep)
            v_inf_arr = np.linalg.norm(v2_list[0] - v_arr)
            tof_days[ia, id_] = tof
            c3_dep[ia, id_] = v_inf_dep**2
            delv_total[ia, id_] = v_inf_dep + v_inf_arr
    return tof_days, c3_dep, delv_total


def _draw_porkchop(title, jd_dep, jd_arr, data, levels, tof_days, tof_levels):
    fig, ax = plt.subplots(figsize=(10, 9))

    # axes in days from the first date, labelled with calendar dates
    x = jd_dep - jd_dep[0]
    y = jd_arr - jd_arr[0]

    cp = ax.contour(x, y, data, levels, cmap="rainbow")
    ax.clabel(cp, inline=True, fontsize=7)
    tp = ax.contour(x, y, tof_days, tof_levels, colors='k', linestyles=':')
    ax.clabel(tp, inline=True, fmt=lambda v: f"{v:.0f} days", fontsize=7)

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: ad.date_string_from_jd(jd_dep[0] + v)))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: ad.date_string_from_jd(jd_arr[0] + v)))
    ax.xaxis.set_tick_params(labelsize=7, rotation=90)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.set_xlabel("Departure date (dd-mm-yyyy)", fontsize=8)
    ax.set_ylabel("Arrival date (dd-mm-yyyy)", fontsize=8)
    ax.grid(which='major', linestyle='dashdot', linewidth=0.5, color='gray')
    ax.grid(which='minor', linestyle='dotted', linewidth=0.5, color='gray')
    ax.minorticks_on()
    ax.set_title(title, fontsize=8)
    ax.set_aspect(1)
    fig.tight_layout()
    plt.show()

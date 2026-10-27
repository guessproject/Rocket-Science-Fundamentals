"""Plotting helpers used by the Orbit Determination notebooks."""
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# columns of the GEO tracking dataset holding (RA, Dec) for each maneuver case
_GEO_RA_DEC_COLUMNS = {None: (1, 2), 'intrack': (3, 4), 'crosstrack': (5, 6), 'radial': (7, 8)}
_GEO_TITLES = {None: "Ra/Dec Evolution without maneuvers", 'intrack': "Ra/Dec Evolution with In-Track maneuver",
               'crosstrack': "Ra/Dec Evolution with Cross-Track maneuver", 'radial': "Ra/Dec Evolution with Radial maneuver"}


def plot_geo_ra_dec(df, maneuver=None):
    """Plot the RA/Dec of a GEO satellite (maneuver=None) or their change after a maneuver.

    df columns: time (s), then (RA, Dec) in deg without maneuver, with an in-track,
    a cross-track and a radial maneuver. maneuver: None, 'intrack', 'crosstrack' or 'radial'.
    """
    data = df.to_numpy()
    hours = data[:, 0] / 3600
    ra_col, dec_col = _GEO_RA_DEC_COLUMNS[maneuver]
    ra, dec = data[:, ra_col], data[:, dec_col]
    if maneuver is None:
        labels = ('Right Ascension', 'Declination')
    else:
        ra, dec = ra - data[:, 1], dec - data[:, 2]
        labels = (' Delta RA (deg)', ' Delta Dec (deg)')

    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_subplot(111)
    ax.set_title(_GEO_TITLES[maneuver])
    ax.scatter(hours, ra, label=labels[0], s=5)
    ax.scatter(hours, dec, label=labels[1], s=5)
    ax.grid(True)
    ax.legend()
    ax.set_xlabel("Time (hr)")
    ax.set_ylabel("Value (deg)")


def plot_flux_index(file_path):
    """Plot observed and predicted F10.7 solar flux from a CelesTrak space weather file."""
    blocks = {'OBSERVED': [], 'DAILY_PREDICTED': []}
    current = None
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            stripped = line.strip()
            if stripped.startswith('BEGIN ') and stripped[6:] in blocks:
                current = stripped[6:]
            elif stripped.startswith('END ') and stripped[4:] in blocks:
                current = None
            elif current is not None:
                blocks[current].append(line.rstrip())

    observed_dates, observed_flux = _parse_space_weather(blocks['OBSERVED'])
    predicted_dates, predicted_flux = _parse_space_weather(blocks['DAILY_PREDICTED'])

    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_subplot(111)
    ax.plot(observed_dates, observed_flux, label='Observed', marker='o', linestyle='-')
    ax.plot(predicted_dates, predicted_flux, label='Predicted', marker='x', linestyle='--')
    ax.grid(True)
    ax.legend()
    ax.set_title("Observed vs Predicted Data")
    ax.set_xlabel("Date")
    ax.set_ylabel("F 10.7")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def _parse_space_weather(lines):
    # dates and F10.7 values (31st column) of a block of space weather data lines
    dates, flux = [], []
    for line in lines:
        columns = line.split()
        if len(columns) < 31:
            continue
        try:
            date = datetime(int(columns[0]), int(columns[1]), int(columns[2]))
            value = float(columns[30])
        except ValueError:
            print(f"Skipping invalid line: {line}")
            continue
        dates.append(date)
        flux.append(value)
    return dates, flux

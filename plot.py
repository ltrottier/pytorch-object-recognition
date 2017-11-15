#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import glob
import json
import seaborn as sns
from argparse import ArgumentParser

sns.set()
sns.set_context("paper", font_scale=2.0, rc={"lines.linewidth": 2.5})


def tfsmooth(x, smoothing_weight):
    y = x.copy()
    for i, v in enumerate(y):
        if i == 0:
            prev = y[0]
        else:
            prev = y[i - 1]
        y[i] = prev * smoothing_weight + (1 - smoothing_weight) * x[i]

    return y


def plot_results(filename):
    smoothing_weight = 0.8
    dirname = os.path.dirname(filename)

    with open(filename, 'r') as fid:
        results = json.load(fid)

    n_epochs = len(results)

    for stats_type in ['Train Stats', 'Test Stats']:

        for key in results[0][stats_type].keys():

            # Get values
            values = [results[epoch][stats_type][key] for epoch in range(n_epochs)]
            values = np.array(values)

            # Plot smoothed values
            values_smooth = tfsmooth(values, smoothing_weight)
            fig, ax = plt.subplots()
            plots = []
            plots = plots + ax.plot(
                values_smooth,
                linestyle='solid',
                label='Smoothing {:0.1f}'.format(smoothing_weight)
            )
            ax.legend(handles=plots)
            ax.set_title(stats_type)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(key)
            fig.set_size_inches(8, 8)
            fig.tight_layout()

            savefile = os.path.join(dirname, "{} - {}.jpg".format(stats_type, key))
            fig.savefig(savefile, tight_boxes=True)
            plt.close(fig)

            # Write report
            values_min = np.min(values)
            values_last = values[-1]
            if n_epochs > 5:
                values_mean_last_five = np.mean(values[-5:])
            else:
                values_mean_last_five = None

            values_smooth_min = np.min(values_smooth)
            values_smooth_last = values_smooth[-1]
            if n_epochs > 5:
                values_smooth_mean_last_five = np.mean(values_smooth[-5:])
            else:
                values_smooth_mean_last_five = None

            savefile = os.path.join(dirname, "{} - {}.report.txt".format(stats_type, key))
            with open(savefile, 'w') as fid:
                lines = []
                lines.append("Original Values")
                lines.append("     Min: {}".format(values_min))
                lines.append("     Last: {}".format(values_last))
                lines.append("     Mean Last 5 Epochs: {}".format(values_mean_last_five))
                lines.append("")
                lines.append("Smoothed Values")
                lines.append("     Min: {}".format(values_smooth_min))
                lines.append("     Last: {}".format(values_smooth_last))
                lines.append("     Mean Last 5 Epochs: {}".format(values_smooth_mean_last_five))
                lines.append("")
                lines = "\n".join(lines)
                fid.write(lines)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--filepath', '-f', help='json file')
    args = parser.parse_args()
    plot_results(args.filepath)

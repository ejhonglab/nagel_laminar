#!/usr/bin/env python3

import argparse
import os
from os.path import join

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    csv_basename = 'pid.csv'
    parser.add_argument('csv_dir', nargs='?', default=os.getcwd(),
        help='should contain a file {}'.format(csv_basename)
    )
    args = parser.parse_args()
    csv_dir = args.csv_dir

    csv_fname = join(csv_dir, csv_basename)
    df = pd.read_csv(csv_fname)

    # TODO delete
    #df = df.loc[(df.time_s >= 10.5) & (df.time_s <= 112)].copy()
    #df.time_s = df.time_s - df.time_s.iloc[0]
    #

    show_valve = True

    fig, ax = plt.subplots()
    ax.plot(df.time_s, df.pid_v, label='PID')

    if show_valve:
        df.valve_control = (df.valve_control > 2.5) * 2

        # TODO either rescale (ideally just at matplotlib level) or just show >2.5 as
        # region behind (maybe CLI flag to hide entirely)
        ax.plot(df.time_s, df.valve_control, label='Valve')

    ax.set_xlabel('Seconds')
    ax.set_ylabel('PID (a.u.)')

    if show_valve:
        ax.legend()

    fig_fname = join(csv_dir, 'pid.svg')
    fig.savefig(fig_fname)
    print('wrote PID figure to {}'.format(fig_fname))

    plt.show()


if __name__ == '__main__':
    main()


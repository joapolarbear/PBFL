import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from typing import List

def smooth(scalars: List[float], weight: float) -> List[float]:
    # One of the easiest implementations I found was to use that Exponential Moving Average the Tensorboard uses, https://stackoverflow.com/questions/5283649/plot-smooth-line-with-pyplot
    # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return np.array(smoothed)

fontsize = 28
policy2attr = {
    "GPFL": ('r'), 
    "Pow-d": ('g'), 
    "Random": ('b'), 
    "FedCor": ('black')
}

def handle_one_file(_file):
    dir_name = os.path.dirname(_file)

    df = pd.read_csv(_file)

    fig = plt.figure(figsize=(8, 5))
    plt.grid(axis="both")
    for policy in ["GPFL", "Pow-d", "Random", "FedCor"]:
        head = f"{policy} - Test/Acc"
        plt.plot(df["Step"], smooth(df[head] * 100, 0.9), label=policy, color=policy2attr[policy])
    plt.xlabel("Round", fontsize=fontsize)
    plt.ylabel("Accuracy (%)", fontsize=fontsize)
    plt.legend(fontsize=fontsize-6, ncol=2)
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_name, os.path.basename(_file).split(".")[0] + "-rst.pdf"))

dir_name = sys.argv[1]
for _file in os.listdir(dir_name):
    if not _file.endswith(".csv"):
        continue
    handle_one_file(os.path.join(dir_name, _file))

import os
import sys
import re
import numpy as np
from typing import List

import matplotlib.pyplot as plt

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

def handle_one_dir(dir_name):
    ucb_alpha_to_rst = {}
    for _file in os.listdir(dir_name):
        match = re.search(r"(?P<policy>(FedCor|PBFL|Pow-d|Random))_policy-(?P<dist_type>(one_shard|two_shard|dir))-(?P<ucb_alpha_fn>\w+)_ucb_alpha-(\d+)to(\d+)-(?P<data_set>\w+)", _file)
        if match is None:
            continue
        ucb_alpha_fn = match["ucb_alpha_fn"].replace("bslash", "/")
        # print(match)
        with open(os.path.join(dir_name, _file), 'r') as fp:
            rst_per_trial = []
            for line in fp.readlines():
                '''
                [2024-01-16 13:22:02(+6.668793439865112s)] [server.py:275] INFO - [ROUND 1] Testing: Loss 2.295965 Acc 0.2662
                '''
                line_match = re.search(r"\[\d+-\d+-\d+ \d+:\d+:\d+\(\+(?P<time>\d+(.\d+)?)s\)\].*\[ROUND (?P<step>\d+)\] Testing: Loss (?P<loss>\d+(.\d+)?) Acc (?P<accu>\d+(.\d+)?)", line)
                if line_match is None:
                    continue
                # print(line_match["time"], line_match["selected_client_num"])
                rst_per_trial.append([float(line_match["time"]), int(line_match["step"]), float(line_match["loss"]), float(line_match["accu"])])
        rst_per_trial = np.array(rst_per_trial)
        ucb_alpha_to_rst[ucb_alpha_fn] = rst_per_trial
    
    fontsize = 28
    def ucb_alpha_to_label(ucb_alpha_fn):
        if ucb_alpha_fn.startswith("const_"):
            ucb_alpha_str = f"alpha={ucb_alpha_fn.split('const_')[1]}"
        elif ucb_alpha_fn.startswith("linear_"):
            ucb_alpha_str = f"alpha={ucb_alpha_fn.split('linear_')[1]} * step"
        else:
            raise
        return ucb_alpha_str
    
    # Const
    fig = plt.figure(figsize=(8, 5))
    plt.grid(axis='both')
    for ucb_alpha_fn in sorted(ucb_alpha_to_rst.keys()):
        if not ucb_alpha_fn.startswith("const_"):
            continue
        rst_per_trial = ucb_alpha_to_rst[ucb_alpha_fn]
        ucb_alpha_str = ucb_alpha_to_label(ucb_alpha_fn)
        plt.plot(rst_per_trial[:, 1], smooth(rst_per_trial[:, 3] * 100, .95), label=ucb_alpha_str)
    plt.xlabel("Round", fontsize=fontsize)
    plt.ylabel("Test Accuracy (%)", fontsize=fontsize)
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)
    plt.legend(fontsize=fontsize-6)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_name, f"{os.path.basename(dir_name)}_const.pdf"))
    plt.close()
    
    # Linear
    fig = plt.figure(figsize=(8, 5))
    plt.grid(axis='both')
    for ucb_alpha_fn in sorted(ucb_alpha_to_rst.keys()):
        if not ucb_alpha_fn.startswith("linear_"):
            continue
        rst_per_trial = ucb_alpha_to_rst[ucb_alpha_fn]
        ucb_alpha_str = ucb_alpha_to_label(ucb_alpha_fn)
        plt.plot(rst_per_trial[:, 1], smooth(rst_per_trial[:, 3] * 100, .95), label=ucb_alpha_str)
    plt.xlabel("Round", fontsize=fontsize)
    plt.ylabel("Test Accuracy (%)", fontsize=fontsize)
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)
    plt.legend(fontsize=fontsize-6)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_name, f"{os.path.basename(dir_name)}_linear.pdf"))
    plt.close()
    
    # overall
    fig = plt.figure(figsize=(8, 5))
    plt.grid(axis='both')
    for ucb_alpha_fn in ["const_0", "const_2", "linear_1/500"]:
        rst_per_trial = ucb_alpha_to_rst[ucb_alpha_fn]
        ucb_alpha_str = ucb_alpha_to_label(ucb_alpha_fn)
        plt.plot(rst_per_trial[:, 1], smooth(rst_per_trial[:, 3] * 100, .95), label=ucb_alpha_str)
    plt.xlabel("Round", fontsize=fontsize)
    plt.ylabel("Test Accuracy (%)", fontsize=fontsize)
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)
    plt.legend(fontsize=fontsize-6)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_name, f"{os.path.basename(dir_name)}_cross.pdf"))
    plt.close()


handle_one_dir(os.path.abspath(sys.argv[1]))
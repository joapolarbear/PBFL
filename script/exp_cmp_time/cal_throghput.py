import os
import sys
import re
import numpy as np

def my_cmp(a, b):
    if a[0] > b[0]:
        return -1
    elif a[0] == b[0]:
        if a[0] > b[0]:
            return 1
        else:
            return -1
    else:
        return 1

TARGET_ROUND_NUM = 200
dir_name = os.path.abspath(sys.argv[1])
policy_list = ["fedcor", "pbfl", "pwd-d", "random"]
all_rst = []
for _file in os.listdir(dir_name):
    match = re.search(r"(?P<policy>(FedCor|PBFL|Pow-d|Random))_policy-(?P<dist_type>(one_shard|two_shard|dir))-(\d+)to(\d+)-(?P<data_set>\w+)", _file)
    policy = match["policy"].lower()
    if policy not in policy_list:
        policy_list.append(policy)
    # print(match)
    with open(os.path.join(dir_name, _file), 'r') as fp:
        time_client_num_list = []
        for line in fp.readlines():
            '''
            [2024-01-16 03:05:57(+5s)] [server.py:130] INFO - ROUND 0
            [2024-01-16 03:05:57(+5s)] [fedcor.py:103] INFO - > FedCor warmup 0
            [2024-01-16 03:05:57(+5s)] [server.py:188] INFO - Pre-client selection 100 -> 5
            [2024-01-16 03:05:57(+5s)] [server.py:193] INFO - Selected clients: [0, 14, 40, 56, 59]
            [2024-01-16 03:05:57(+6s)] [server.py:466] INFO - 5 Clients Training: Loss 0.134344 Acc 0.9758
            [2024-01-16 03:05:58(+6s)] [server.py:274] INFO - Testing: Loss 2.755304 Acc 0.1365
            ...
            Power-d
            [2024-01-16 03:32:55(+102.51574158668518s)] [server.py:166] INFO - Candidate client selection 10/100
            '''
            if policy == "pow-d":
                line_match = re.search(r"\[\d+-\d+-\d+ \d+:\d+:\d+\(\+(?P<time>\d+(.\d+)?)s\)\].*Candidate client selection (?P<selected_client_num>\d+)/\d+", line)
            else:
                line_match = re.search(r"\[\d+-\d+-\d+ \d+:\d+:\d+\(\+(?P<time>\d+(.\d+)?)s\)\].*client selection \d+ -> (?P<selected_client_num>\d+)", line)
            if line_match is None:
                continue
            # print(line_match["time"], line_match["selected_client_num"])
            time_client_num_list.append([float(line_match["time"]), int(line_match["selected_client_num"])])
        
        time_client_num_array = np.array(time_client_num_list)
        # print(time_client_num_array.shape)
        try:
            time_per_client = np.mean((time_client_num_array[1:, 0] - time_client_num_array[:-1, 0]) / time_client_num_array[:-1, 1])
        except:
            import pdb; pdb.set_trace()
        selected_client_num_per_step = time_client_num_array[:-1, 1]
        all_consumed_time = time_per_client * (np.sum(selected_client_num_per_step) + (TARGET_ROUND_NUM - len(selected_client_num_per_step)) * selected_client_num_per_step[-1])
        all_rst.append([match['data_set'], match['dist_type'], policy, time_per_client * TARGET_ROUND_NUM, all_consumed_time])
    
all_rst = sorted(all_rst, key=lambda x: (x[0], x[1], x[2]))
for data_set, dist_type, policy, time, all_consumed_time in all_rst:
    print(f"{data_set} {dist_type} {policy} --> {time:.3f}s per {TARGET_ROUND_NUM} steps, {all_consumed_time=:.3f}s")
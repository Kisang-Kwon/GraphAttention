import os
import re


def default_args():
    args = dict()

    return args


def read_config(fpath):
    args = default_args()

    Fopen = open(fpath)
    for line in Fopen:
        arr = re.split(' = ', line.rstrip())

        if arr[0] == 'epochs':
            args[arr[0]] = int(arr[1])
        elif arr[0] == 'batch_size':
            args[arr[0]] = int(arr[1])
        elif arr[0] == 'learning_rate':
            args[arr[0]] = float(arr[1])
        elif arr[0] == 'l2_params':
            args[arr[0]] = float(arr[1])
        else:
            args[arr[0]] = arr[1]
    Fopen.close()

    return args


def print_all_hyper_parameters(args, LOG):
    LOG.write('==== Parameters ====\n')
    for p in args:
        LOG.write(f'{p} = {args[p]}\n')
    LOG.write('====================\n')
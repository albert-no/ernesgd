import argparse
import json
import subprocess


def sweep(gpucode, lr_list, option_dict):
    cmd_template = f'(CUDA_VISIBLE_DEVICES={gpucode} python run_dcgan.py'
    for key in option_dict:
        cmd_template += f' --{key}={option_dict[key]}'
    for lr in lr_list:
        cmd_run = cmd_template + f' --lr={lr})'
        print(cmd_run)
        subprocess.call(cmd_run, shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default='sample_parameters.json',
        help='name of parameters json file')
    opt = parser.parse_args()

    with open(opt.json) as f:
        json_data = json.load(f)

    gpucode = json_data['gpucode']
    lr_list = json_data['lr_list']
    exceptions = ['gpucode', 'lr_list']
    option_dict = {}
    for key in json_data:
        if key not in exceptions:
            option_dict[key] = json_data[key]

    sweep(gpucode, lr_list, option_dict)



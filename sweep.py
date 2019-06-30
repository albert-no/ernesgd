import subprocess


def sweep(gpucode, pycode, lr_list, option_dict):
    cmd_template = f'(CUDA_VISIBLE_DEVICES={gpucode} python {pycode}.py'
    for key in option_dict:
        cmd_template += f' --{key}={option_dict[key]}'
    cmd_template += ')'
    for lr in lr_list:
        subprocess.call(cmd_template, shell=True)


if __name__ == '__main__':
    gpucode = 1
    pycode = 'dcgan_ernest'
    lr_list = [0.02, 0.04, 0.06, 0.08]
    option_dict = {'n_cpu': 16,
            'dataset_name': 'CIFAR',
            'batch_size': 128,
            'FID_epochs': 20,
            'img_size': 32,
    }
    sweep(gpucode, pycode, lr_list, option_dict)



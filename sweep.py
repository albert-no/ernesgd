import subprocess


def sweep(gpucode, pycode, lr_list):
    for lr in lr_list:
        subprocess.call(
            f'(CUDA_VISIBLE_DEVICES={gpucode} python {pycode}.py --lr={lr} --n_cpu=16 --dataset_name=CIFAR --batch_size=128)',
            shell=True)


if __name__ == '__main__':
    gpucode = 1
    pycode = 'dcgan_sgd'
    lr_list = [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5]
    sweep(gpucode, pycode, lr_list)



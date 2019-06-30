# ERNESGD
ERNESGD is 

python3 is required.

Under virtual environment, run setup command.

```
python setup.py install
```



## Data
The following command will download and create real images,
as well as extract statistics.
Valid options are ``CIFAR`` and ``MNIST``.

```
cd ernesgd_data
python get_real_images.py
```

If you alreaday donwloaded dataset, you can run the command in ``extract statisics only`` mode.
set ``downloaded=True``



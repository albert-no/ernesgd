# ERNESGD
ERNESGD is 

python3 is required.




## Data
The following command will download and create real images,
as well as extract statistics.

```
python get_real_images.py
```

Before you run the command, you need to modify``fid_score.py``.
``from .inception`` to ``from inception``. 
This will be fixed .

Valid options are ``CIFAR`` and ``MNIST``.


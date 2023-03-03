# fair_ensemble

Official Repo for "FAIR-ENSEMBLE: WHEN FAIRNESS NATURALLY
EMERGES FROM DEEP ENSEMBLING" project at Cohere4AI

Install dependencies by : 

```
conda create -y -n fair_ensemble_env python=3.9 enlighten cupy pkg-config compilers libjpeg-turbo cudatoolkit=11.3 numba -c pytorch -c conda-forge
conda activate fair_ensemble_env
pip install -r requirements.txt
pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

Enter the desired dataset folder (CIFAR100/TinyImagenet) and execute the following commands :

1. Create Datasets (only required the first time you set things up)
```
python scripts/make_dataset.py 
```

2. Train Networks with various ablations
```
bash c100_launcher.sh
```


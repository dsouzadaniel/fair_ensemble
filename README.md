# fair_ensemble

Official Repo for "FAIR-ENSEMBLE: WHEN FAIRNESS NATURALLY
EMERGES FROM DEEP ENSEMBLING" project at Cohere4AI

Install dependencies by : 
```
conda create -y -n beyond_env python=3.9 enlighten cupy pkg-config compilers libjpeg-turbo opencv pytorch torchvision cudatoolkit=11.3 numba -c pytorch -c conda-forge
conda activate beyond_env
pip install beautifultable
pip install ffcv
```

Enter the desired dataset folder (CIFAR100/TinyImagenet) and execute the following commands :

1. Create Datasets (only required the first time you set things up)
```
python make_dataset.py 
```

2. Train Networks with various ablations
```
bash c100_launcher.sh
```

3. Use **Plots.ipynb** to visualize results

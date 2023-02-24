#!/bin/bash
echo arg 1 - 'Name of model'
echo arg 2 - '(Optional) config yaml path, defaults to default_config_<model_name>.yaml'

model_name=$1
default_yaml=default_config_${model_name}.yaml
yaml_pth=${2:-$default_yaml}

## Run 5 variations of 4 Ablations
for seed in {1..5}
do
#  # Run all random seeds (enforce random seeds for Data Augmentation, Model Init, and Batch Order)
  python train_cifar100.py --config-file ${yaml_pth} --exp.ix $seed --exp.ablation "RANDOM" --seed.DA $seed --seed.model_init $seed --seed.batch_order $seed
#  # Run fixed seeds for Model Init and Batch Order (enforce random seeds for Data Augmentation)
  python train_cifar100.py --config-file ${yaml_pth} --exp.ix $seed --exp.ablation "MS_BS" --seed.DA $seed
#  # Run Data Augmentation ( enforce random seeds for Model Init and Batch Order)
#  python train_cifar100.py --config-file ${yaml_pth} --exp.ix $seed --exp.ablation "DA" --seed.model_init $seed --seed.batch_order $seed
#  # Run Data Augmentation with fixed seeds for Model Init and Batch Order (fixed by default in train_cifar100.py)
#  python train_cifar100.py --config-file ${yaml_pth} --exp.ix $seed --exp.ablation "DA_MS_BS"
#  # Run Data Augmentation with fixed seeds for Model Init (enforce random seeds for Batch Order)
#  python train_cifar100.py --config-file ${yaml_pth} --exp.ix $seed --exp.ablation "DA_MS" --seed.batch_order $seed
#  # Run Data Augmentation with fixed seeds for Batch Order (enforce random seeds for Model Init)
#  python train_cifar100.py --config-file ${yaml_pth} --exp.ix $seed --exp.ablation "DA_BS" --seed.model_init $seed
done

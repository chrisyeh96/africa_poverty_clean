TODO
- instructions for loading ImageNet weights

## Using publicly available satellite imagery and deep learning to understand economic well-being in Africa

This repository includes the code and data necessary to reproduce the results and figures for the article "Using publicly available satellite imagery and deep learning to understand economic well-being in Africa" published in *Nature Communications*.

[https://doi.org/10.1038/s41467-020-16185-w](Nature Communications)  # TODO

BibTex entry  # TODO

```tex
@article {

}
```


## Hardware and Software Requirements

This code was tested on a system with the following specifications:

- operating system: Ubuntu 16.04.6 LTS
- memory: 125GB RAM
- disk storage: 500GB
- GPU: TODO

The software requirements are listed in the `env.yml` file, which is meant to be used with `conda` (version 4.8.3). See [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) for instructions on installing conda. Once conda is installed, run the following command to set up the conda environment:

```bash
conda env create -f env.yml
```


## Instructions

TODO


## Training DHS models

The generic script for training ResNet-18 models for predicting DHS asset-wealth is as follows:

```bash
python train_dhs.py \
    --label_name wealthpooled --batcher base \
    --model_name resnet --num_layers 18 \
    --lr_decay 0.96 --batch_size 64 \
    --gpu 0 --num_threads 5 \
    --cache train,train_eval,val \
    --augment True --eval_every 1 --print_every 40 \
    --ooc {ooc} --max_epochs {max_epochs} \
    --out_dir {out_dir} \
    --keep_frac {keep_frac} --seed {seed} \
    --experiment_name {experiment_name} \
    --dataset {dataset} \
    --ls_bands {ls_bands} --nl_band {nl_band} \
    --lr {lr} --fc_reg {reg} --conv_reg {reg} \
    --imagenet_weights_path {imagenet_weights_path} \
    --hs_weight_init {hs_weight_init}
```

Settings for "out-of-country" vs. "in-country" experiments:

Setting      | "out-of-country"     | "in-country"
-------------|----------------------|-------------
`max_epochs` | 200                  | 150
`ooc`        | `True`               | `False`
`out_dir`    | `./outputs/DHS_OOC/` | `./outputs/DHS_Incountry/`

Settings for different CNN models:

Setting                 | `CNN MS`                                    | `CNN NL`
------------------------|---------------------------------------------|-------------
`ls_bands`              | `ms`                                        | `None`
`nl_band`               | `None`                                      | `split`
`imagenet_weights_path` | `./models/imagenet_resnet18_tensorpack.npz` | `None`
`hs_weight_init`        | `samescaled`                                | `None`

For cross-validation hyper-parameter tuning, we tested the following values:

Setting                        | Values Tested
-------------------------------|--------------------------------------------------------------------
`dataset` (for out-of-country) | `'DHS_OOC_X'` where `X` is one of `['A', 'B', 'C', 'D', 'E']`
`dataset` (for in-country)     | `'DHS_Incountry_X'` where `X` is one of `['A', 'B', 'C', 'D', 'E']`
`reg`                          | `[1e-0, 1e-1, 1e-2, 1e-3]`
`lr`                           | `[1e-2, 1e-3, 1e-4, 1e-5]`

Once the optimal hyperparameters for each cross-validation fold were determined, we then experimented with training the models on subsets of the data:

Setting     | Values Tested
------------|----------------------------
`keep_frac` | `[0.05, 0.1, 0.25, 0.5, 1]`
`seed`      | `[123, 456, 789]`

Here is an example of a complete training run:

```bash
python train_dhs.py \
    --label_name wealthpooled --batcher base \
    --model_name resnet --num_layers 18 \
    --lr_decay 0.96 --batch_size 64 \
    --gpu 0 --num_threads 5 \
    --cache train,train_eval,val \
    --augment True --eval_every 1 --print_every 40 \
    --ooc True --max_epochs 200 \
    --out_dir ./outputs/DHS_OOC/ \
    --keep_frac 0.25 --seed 456 \
    --experiment_name DHS_OOC_D_ms_samescaled_keep0.25_seed456 \
    --dataset DHS_OOC_D \
    --ls_bands ms --nl_band None \
    --lr 1e-2 --fc_reg 1e-3 --conv_reg 1e-3 \
    --imagenet_weights_path ./models/imagenet_resnet18_tensorpack.npz \
    --hs_weight_init samescaled
```


## Training Transfer-learning Model

TODO


## Training LSMS Models

TODO

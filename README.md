## Using publicly available satellite imagery and deep learning to understand economic well-being in Africa

This repository includes the code and data necessary to reproduce the results and figures for the article "Using publicly available satellite imagery and deep learning to understand economic well-being in Africa" published in *Nature Communications* on May 22, 2020 ([link](https://www.nature.com/articles/s41467-020-16185-w)).

Please cite this article as follows, or use the BibTeX entry below.

> Yeh, C., Perez, A., Driscoll, A. *et al*. Using publicly available satellite imagery and deep learning to understand economic well-being in Africa. *Nat Commun* **11**, 2583 (2020). https://doi.org/10.1038/s41467-020-16185-w

```tex
@article{yeh2020using,
    author = {Yeh, Christopher and Perez, Anthony and Driscoll, Anne and Azzari, George and Tang, Zhongyi and Lobell, David and Ermon, Stefano and Burke, Marshall},
    day = {22},
    doi = {10.1038/s41467-020-16185-w},
    issn = {2041-1723},
    journal = {Nature Communications},
    month = {5},
    number = {1},
    title = {{Using publicly available satellite imagery and deep learning to understand economic well-being in Africa}},
    url = {https://www.nature.com/articles/s41467-020-16185-w},
    volume = {11},
    year = {2020}
}
```

For a list of known errata discovered since the paper was published, please consult the [errata.md](https://github.com/sustainlab-group/africa_poverty/tree/master/errata.md) file. To the best of our knowledge, the errata do not affect the main findings of the paper.


## Table of Contents

* [Computing Requirements](#computing-requirements)
* [Running trained CNN models](#running-trained-cnn-models)
* [Training baseline models](#training-baseline-models)
* [Training DHS models](#training-dhs-models)
* [Training transfer learning models](#training-transfer-learning-models)
* [Training LSMS models](#training-lsms-models)
* [Code Formatting and Type Checking](#code-formatting-and-type-checking)


## Computing Requirements

This code was tested on a system with the following specifications:

- operating system: Ubuntu 16.04.6 LTS
- CPU: Intel Xeon Silver 4110
- memory (RAM): 125GB
- disk storage: 500GB
- GPU: 1x NVIDIA Titan Xp

The main software requirements are Python 3.7 with TensorFlow r1.15, and R 3.6. The complete list of required packages and library are listed in the `env.yml` file, which is meant to be used with `conda` (version 4.8.3). See [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) for instructions on installing conda via Miniconda. Once conda is installed, run the following command to set up the conda environment:

```bash
conda env update -f env.yml --prune
```

If you are using a GPU, you may need to also install CUDA 10 and cuDNN 7. If you do not have a GPU, replace the `tensorflow-gpu=1.15.*` line in the `env.yml` file with `tensorflow=1.15.*`.


## Running trained CNN models

Steps:

1. Follow the numbered files in the `preprocessing/` directory to download the necessary satellite images, shapefiles, and trained model checkpoints.

2. Identify interesting feature activation maps using the `max_activating.ipynb` (TODO) notebook.

3. Run the `extract_features.py` script to use the trained CNN models to extract 512-dimensional feature vectors from each satellite image.

4. Use the `models/dhs_resnet_ridge.py` and TODO notebooks to train ridge regression models on top of the extracted feature vectors. Training these ridge regression models is roughly equivalent to "fine-tuning" the final layer of the trained CNN models.

5. Use the following notebooks to analyze the results:
  - `model_analysis/dhs_ooc.py`
  - `model_analysis/dhs_incountry.py` (TODO)
  - keep-frac models (TODO)
  - LSMS models (TODO)


## Training baseline models

TODO
- KNN
- Ridge on histograms


## Training DHS models

Follow the numbered files in the `preprocessing/` directory to download the necessary satellite images and shapefiles.

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

Training progress can be monitored by TensorBoard:

```bash
tensorboard --logdir {out_dir}
```

Settings for "out-of-country" vs. "in-country" experiments:

Setting      | "out-of-country"     | "in-country"
-------------|----------------------|-------------
`max_epochs` | 200                  | 150
`ooc`        | `True`               | `False`
`out_dir`    | `./outputs/dhs_ooc/` | `./outputs/dhs_incountry/`

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
    --out_dir ./outputs/dhs_ooc/ \
    --keep_frac 0.25 --seed 456 \
    --experiment_name DHS_OOC_D_ms_samescaled_keep0.25_seed456 \
    --dataset DHS_OOC_D \
    --ls_bands ms --nl_band None \
    --lr 1e-2 --fc_reg 1e-3 --conv_reg 1e-3 \
    --imagenet_weights_path ./models/imagenet_resnet18_tensorpack.npz \
    --hs_weight_init samescaled
```


## Training transfer learning models

TODO


## Training LSMS models

TODO


TODO
- instructions for loading ImageNet weights


## Code Formatting and Type Checking

This repo uses [flake8](https://flake8.pycqa.org/) for Python linting and [mypy](https://mypy.readthedocs.io/) for type-checking. Configuration files for each are included in this repo: `.flake8` and `mypy.ini`.

To run either code linting or type checking, set the current directory to the repo root directory. Then run any of the following commands:

```bash
# LINTING
# =======

# entire repo
flake8

# all modules within utils directory
flake8 utils

# a single module
flake8 utils/analysis.py

# a jupyter notebook - we ignore these error codes:
# - E305: expected 2 blank lines after class or function definition
# - E402: Module level import not at top of file
# - F404: from __future__ imports must occur at the beginning of the file
# - W391: Blank line at end of file
jupyter nbconvert preprocessing/1_process_tfrecords.ipynb --stdout --to script | flake8 - --extend-ignore=E305,E402,F404,W391


# TYPE CHECKING
# =============

# entire repo
mypy .

# all modules within utils directory
mypy -p utils

# a single module
mypy utils/analysis.py

# a jupyter notebook
mypy -c "$(jupyter nbconvert preprocessing/1_process_tfrecords.ipynb --stdout --to script)"
```

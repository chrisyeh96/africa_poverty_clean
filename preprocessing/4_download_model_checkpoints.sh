#!/bin/bash

# This Bash script downloads trained TensorFlow model checkpoints
# from GitHub into the outputs/ directory.
#
# Run this script from within the preprocessing/ directory.
#
# Prerequisites: None.

mkdir -p ../outputs
cd ../outputs

BASE_GITHUB_URL="https://github.com/sustainlab-group/africa_poverty/releases/download/v1.0.1/"

DHS_OOC_MODELS=(
    "DHS_OOC_A_ms_samescaled_b64_fc01_conv01_lr0001"
    "DHS_OOC_B_ms_samescaled_b64_fc001_conv001_lr0001"
    "DHS_OOC_C_ms_samescaled_b64_fc001_conv001_lr001"
    "DHS_OOC_D_ms_samescaled_b64_fc001_conv001_lr01"
    "DHS_OOC_E_ms_samescaled_b64_fc01_conv01_lr001"

    "DHS_OOC_A_nl_random_b64_fc1.0_conv1.0_lr0001"
    "DHS_OOC_B_nl_random_b64_fc1.0_conv1.0_lr0001"
    "DHS_OOC_C_nl_random_b64_fc1.0_conv1.0_lr0001"
    "DHS_OOC_D_nl_random_b64_fc1.0_conv1.0_lr01"
    "DHS_OOC_E_nl_random_b64_fc1.0_conv1.0_lr0001"

    "DHS_OOC_A_rgb_same_b64_fc001_conv001_lr01"
    "DHS_OOC_B_rgb_same_b64_fc001_conv001_lr0001"
    "DHS_OOC_C_rgb_same_b64_fc001_conv001_lr0001"
    "DHS_OOC_D_rgb_same_b64_fc1.0_conv1.0_lr01"
    "DHS_OOC_E_rgb_same_b64_fc001_conv001_lr0001"
)

DHS_INCOUNTRY_MODELS=(
    "DHS_Incountry_A_ms_samescaled_b64_fc01_conv01_lr001"
    "DHS_Incountry_B_ms_samescaled_b64_fc1_conv1_lr001"
    "DHS_Incountry_C_ms_samescaled_b64_fc1.0_conv1.0_lr0001"
    "DHS_Incountry_D_ms_samescaled_b64_fc001_conv001_lr0001"
    "DHS_Incountry_E_ms_samescaled_b64_fc001_conv001_lr0001"

    "DHS_Incountry_A_nl_random_b64_fc1.0_conv1.0_lr0001"
    "DHS_Incountry_B_nl_random_b64_fc1.0_conv1.0_lr0001"
    "DHS_Incountry_C_nl_random_b64_fc1.0_conv1.0_lr0001"
    "DHS_Incountry_D_nl_random_b64_fc1.0_conv1.0_lr0001"
    "DHS_Incountry_E_nl_random_b64_fc01_conv01_lr001"
)

TRANSFER_MODELS=(
    "transfer_nlcenter_ms_b64_fc001_conv001_lr0001"
    "transfer_nlcenter_rgb_b64_fc001_conv001_lr0001"
)


echo "Downloading and unzipping DHS_OOC model checkpoints"
mkdir dhs_ooc
for model in ${DHS_OOC_MODELS[@]}
do
    echo "Downloading model ${model}"
    url="${BASE_GITHUB_URL}/${model}.zip"
    wget --no-verbose --show-progress -P dhs_ooc ${url}

    echo "Unzipping model ${model}"
    unzip "dhs_ooc/${model}.zip" -d "dhs_ooc/${model}"
done


echo "Downloading and unzipping DHS_Incountry model checkpoints"
mkdir dhs_incountry
for model in ${DHS_INCOUNTRY_MODELS[@]}
do
    echo "Downloading model ${model}"
    url="${BASE_GITHUB_URL}/${model}.zip"
    wget --no-verbose --show-progress -P dhs_incountry ${url}

    echo "Unzipping model ${model}"
    unzip "dhs_incountry/${model}.zip" -d "dhs_incountry/${model}"
done


echo "Downloading and unzipping transfer learning model checkpoints"
mkdir transfer
for model in ${TRANSFER_MODELS[@]}
do
    echo "Downloading model ${model}"
    url="${BASE_GITHUB_URL}/${model}.zip"
    wget --no-verbose --show-progress -P transfer ${url}

    echo "Unzipping model ${model}"
    unzip "transfer/${model}.zip" -d "transfer/${model}"
done

#!/bin/bash

python3 src/multi_linear_regression.py \
    --exp_names_str_list MedInc HouseAge AveRooms AveBedrms Population AveOccup Latitude Longitude \
    --output_dir_path workdir
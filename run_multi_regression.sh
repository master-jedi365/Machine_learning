#!/bin/bash

#==================#
#     変数説明     #
#==================#
# exp_names_str_list: 説明変数に使う項目をスペース区切りで記述する


python3 src/multi_linear_regression.py \
    --exp_names_str_list MedInc HouseAge AveRooms AveBedrms Population AveOccup Latitude Longitude \
    --output_dir_path workdir
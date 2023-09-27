import os
import sys
import yaml
import argparse
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

sys.path.append(os.path.join("src", "utils"))
from data_utils import get_calif_dataset, get_data_from_name

OUTPUT_NAME="multi_linear_result"

def arg_parse():
    ap = argparse.ArgumentParser()  
    ap.add_argument('--tran_test_split_ratio', type=float, default=1/10)
    ap.add_argument('--exp_names_str_list', type=str, nargs="*")  
    ap.add_argument('--output_dir_path', type=str, default="workdir")  
    return ap.parse_args()    

def main():
    args = arg_parse()

    calif_ds = get_calif_dataset(args.tran_test_split_ratio)
    lr = LinearRegression()

    # 重線形回帰
    train_X, test_X = get_data_from_name(args.exp_names_str_list, calif_ds)
    lr.fit(train_X, calif_ds.pred_train_data)
    a_list = lr.coef_[0]
    b = lr.intercept_[0]
    R2 = lr.score(test_X, calif_ds.pred_test_data)

    print("a: ", a_list)
    print("b: ", b)
    print("test R2: ", R2)
    #出力
    os.makedirs(args.output_dir_path, exist_ok=True)
    # visualize
    coef_list = list(map(float, a_list))
    plt.bar(list(range(0, len(args.exp_names_str_list))), coef_list)
    plt.savefig(os.path.join(args.output_dir_path, f"{OUTPUT_NAME}.jpg"))
    # parameters as yaml
    out_dict = {
        "coefs": coef_list,
        "intercept": float(lr.intercept_[0]),
        "test_R2": float(R2)
    }
    with open(os.path.join(args.output_dir_path, f"{OUTPUT_NAME}.yaml"), "w") as f:
        yaml.dump(out_dict, f, encoding='utf-8')

if __name__ == "__main__":
    main()
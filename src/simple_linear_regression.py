import os
import sys
import yaml
import argparse
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

sys.path.append(os.path.join("src", "utils"))
from data_utils import get_calif_dataset, get_data_from_name

OUTPUT_NAME="simple_linear_result"

def arg_parse():
    ap = argparse.ArgumentParser()  
    ap.add_argument('--tran_test_split_ratio', type=float, default=1/10)
    ap.add_argument('--exp_name_str', type=str, default="MedInc")  
    ap.add_argument('--output_dir_path', type=str, default="workdir")  
    return ap.parse_args()    

def main():
    args = arg_parse()

    calif_ds = get_calif_dataset(args.tran_test_split_ratio)
    lr = LinearRegression()

    # 一次元線形回帰
    train_X, test_X = get_data_from_name([args.exp_name_str], calif_ds)
    lr.fit(train_X, calif_ds.pred_train_data)
    a = lr.coef_[0][0]
    b = lr.intercept_[0]
    R2 = lr.score(test_X, calif_ds.pred_test_data)

    print("a: ", a)
    print("b: ", b)
    print("test R2: ", R2)
    #出力
    # visualize
    os.makedirs(args.output_dir_path, exist_ok=True)
    plt.scatter(test_X.ravel(), calif_ds.pred_test_data.ravel())
    max_x = max(train_X.ravel())
    plt.plot([0, max_x], [b, a * max_x + b], color="red")
    plt.savefig(os.path.join(args.output_dir_path, f"{OUTPUT_NAME}.jpg"))

    # parameters as yaml
    coef_list = list(map(float, lr.coef_[0]))
    out_dict = {
        "coefs": coef_list,
        "intercept": float(lr.intercept_[0]),
        "test_R2": float(R2)
    }
    with open(os.path.join(args.output_dir_path, f"{OUTPUT_NAME}.yaml"), "w") as f:
        yaml.dump(out_dict, f, encoding='utf-8')

if __name__ == "__main__":
    main()
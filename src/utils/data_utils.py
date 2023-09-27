from dataclasses import dataclass
import random
import numpy as np

from sklearn.datasets import fetch_california_housing

@dataclass
class dataset:
    exp_train_data: np.ndarray
    exp_test_data: np.ndarray
    pred_train_data: np.ndarray
    pred_test_data: np.ndarray
    exp_names: list[str]
    pred_name: list[str]
    data_num: int
    train_data_idx: list[int]
    test_data_idx: list[int]

def get_exp_name_idx(name: str, ds: dataset) -> int:
    return ds.exp_names.index(name)

def get_data_from_name(name_list: list[str], ds: dataset) -> tuple[np.ndarray]:
    name_idx_list = [get_exp_name_idx(name, ds) for name in name_list]
    return ds.exp_train_data[:, name_idx_list], ds.exp_test_data[:, name_idx_list]

def get_calif_dataset(split_ratio: float) -> dataset:
    data_dict = fetch_california_housing()

    data_array = data_dict["data"]
    target_array = data_dict["target"]
    exp_names = data_dict["feature_names"]
    pred_name = data_dict["target_names"]
    data_num = data_array.shape[0]
    
    test_data_idx =random.sample(list(range(0, data_num)), int(data_num * split_ratio))
    train_data_idx = list(set(range(0, data_num)) - set(test_data_idx))
    print("#----------------------------------------------------------#")
    print("#             california_housing dataset                   #")
    print("#----------------------------------------------------------#")
    print("Explonation names: ", exp_names)
    print("Prediction names: ", pred_name)
    print("All data num: ", data_num)
    print("Test data num: ", len(test_data_idx))
    print("Train data num", len(train_data_idx))
    print("#----------------------------------------------------------#")

    exp_train_data = data_array[train_data_idx, :]
    exp_test_data = data_array[test_data_idx, :]
    pred_train_data = target_array[train_data_idx].reshape(-1, 1)
    pred_test_data = target_array[test_data_idx].reshape(-1, 1)

    ds = dataset(
        exp_train_data=exp_train_data,
        exp_test_data=exp_test_data,
        pred_train_data=pred_train_data,
        pred_test_data=pred_test_data,
        exp_names=exp_names,
        pred_name=pred_name,
        data_num=data_num,
        train_data_idx=train_data_idx,
        test_data_idx=test_data_idx
    )
    
    return ds
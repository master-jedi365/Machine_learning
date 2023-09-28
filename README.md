# 機器学習サンプルコード集

## 動作確認環境
| 項目 | 内容 |
| :--- | :--- | 
| OS | ubuntu 20.04, Linux version 5.15.0-52-generic |
| RAM | 32GB |
| CPU | Intel(R) Core(TM) i9-9900K |
| GPU  | NVIDIA GeForce RTX 2080 Ti |
| dcoker | 20.10.21 (nvidia-docker) |

---
## 環境構築

以下のコマンドを実行することでdockerfileからdocker imageのビルドを行う：
```
$ cd your/directory/path/Machine_learning/docker
$ docker build -t ml_sample:latest .
```

---
## 実行スクリプト説明
- run_simple_regression.sh: カリフォルニアデータセットの単線形回帰分析を行うスクリプト
- run_multi_regression.sh: カリフォルニアデータセットの重線形回帰分析を行うスクリプト
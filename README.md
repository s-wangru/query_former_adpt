# QueryFormer Forked Repo - README

Query representation learning plays a crucial role in various AI-based database tasks, such as query optimization and workload forecasting. QueryFormer is a method that uses attention to learn query representation. 

_Zhao, Yue, Gao Cong, Jiachen Shi, and Chunyan Miao. "Queryformer: A Tree Transformer Model for Query Plan Representation." Proceedings of the VLDB Endowment 15, no. 8 (2022): 1658-1670._

## Setting up a Python 3.12.2 Virtual Environment
1. check available python versions installed via pyenv
```shell
pyenv versions
```
2. set a python version for the current project
```shell
cd project_dir
pyenv local 3.12.2
```

confirm that the local python was set to the target python version:
```shell
python --version
```
3. create and activate a python virtual env:
```shell
python -m venv .venv
source .venv/bin/activate
```

4. install python dependencies
```shell
pip install -r requirements.txt
```

5. Generate tpc-h worklaod
```shell
export DSS_CONFIG = path to dbgen
export DSS_PATH = $DSS_CONFIG/output
./dbgen.sh
```

6. Generate encoding for queries
```shell
# python encode_dataset.py --file-name --dataset
python encode_dataset.py --file-name tpch-kit/dbgen/1 --dataset tpch10
```

7. Train on given quereis
```shell
# python train_query_former.py --file-name --dataset-name --topredict
python train_query_former.py --file-name tpch10_data

8. Train with 
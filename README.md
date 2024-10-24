# QueryFormer Forked Repo - README

Query representation learning plays a crucial role in various AI-based database tasks, such as query optimization and workload forecasting. QueryFormer is a method that uses attention to learn query representation. 

_Zhao, Yue, Gao Cong, Jiachen Shi, and Chunyan Miao. "Queryformer: A Tree Transformer Model for Query Plan Representation." Proceedings of the VLDB Endowment 15, no. 8 (2022): 1658-1670._

QueryFormer learns query representations that can be utilized for different prediction tasks. The QueryFormer code, designed for PyTorch, was released over 2 years ago. The python ecosystem has progressed since then. I updated the QueryFormer code to work with the latest Python version (3.12.2), PyTorch, and other required packages. I’ve finally managed to run the QueryFormer code on the latest Python runtime after investing several days of effort. In this README, I have detailed the steps for setting up the Python environment and making the code working.  

The training pipeline is in the Training V1.ipynb file. To make the pipeline run faster, I have changed the original batch size from 1024 to 128. Also, in the training dataset, instead of loading all 18 files, I loaded 2 files. You'll find them in the Training V1.ipynb file. 

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

encode_dataset.ipynb contains the streamline to encode a given dataset and SQL queries into EXPLAIN plans and the encoding dataset containing
1. min and max of each column
2. histogram file for the dataset
3. bitmap of 1000 samples

training_tpch.ipynb trains on the dataset tpc-h
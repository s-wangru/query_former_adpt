import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import time
import pandas as pd
from scipy.stats import pearsonr

from model.util import Normalizer
from model.database_util import get_hist_file, get_job_table_sample, collator
from model.model import QueryFormer
from model.database_util import Encoding
from model.dataset import PlanTreeDataset
from model.trainer import eval_workload, train

data_path = ''
histogram_file = 'histograms.csv'
to_predict = 'cost'
dataset = 'tpch10'
train_file = 'query_plans.csv'

test_file = 'query_plans.csv'

class Args:
    # bs = 1024
    # SQ: smaller batch size
    bs = 128
    lr = 0.001
    # epochs = 200
    epochs = 100
    clip_size = 50
    embed_size = 32
    pred_hid = 128
    ffn_dim = 128
    head_size = 12
    n_layers = 8
    dropout = 0.1
    sch_decay = 0.6
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    newpath = './results/full/cost/'
    to_predict = 'cost'
args = Args()

if not os.path.exists(args.newpath):
    os.makedirs(args.newpath)

def train_query_former():

    hist_file = get_hist_file(histogram_file)
    cost_norm = Normalizer(-3.61192, 12.290855)
    card_norm = Normalizer(1,100)

    encoding_ckpt = torch.load('checkpoints/'+ dataset + '_encoding.pt')
    encoding = encoding_ckpt['encoding']
    checkpoint = torch.load('checkpoints/cost_model.pt', map_location='cpu')

    from model.util import seed_everything
    seed_everything()

    model = QueryFormer(emb_size = args.embed_size ,ffn_dim = args.ffn_dim, head_size = args.head_size, \
                    dropout = args.dropout, n_layers = args.n_layers, \
                    use_sample = True, use_hist = True, \
                    pred_hid = args.pred_hid
                    )

    model.to(args.device)

    dfs = []  # list to hold DataFrames

    df = pd.read_csv(train_file)
    dfs.append(df)

    full_train_df = pd.concat(dfs)

    val_dfs = []  # list to hold DataFrames

    df = pd.read_csv(test_file)
    val_dfs.append(df)

    val_df = pd.concat(val_dfs)

    table_sample = get_job_table_sample(data_path + '/' + dataset)

    train_ds = PlanTreeDataset(full_train_df, None, encoding, hist_file, card_norm, cost_norm, to_predict, table_sample)
    val_ds = PlanTreeDataset(val_df, None, encoding, hist_file, card_norm, cost_norm, to_predict, table_sample)

    crit = nn.MSELoss()
    model, best_path = train(model, train_ds, val_ds, crit, cost_norm, args)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train QueryFormer model")
    parser.add_argument('--file-name', type=str, default='',
                        help=f"Path to the SQL queries file (default: )")
    parser.add_argument('--dataset-name', type=str, default='tpch',
                        help=f"Name of the sample dataset/database (default: tpch)")
    parser.add_argument('--topredict', type=str, default='cost',help='cost or card')

    input_args = parser.parse_args()

    data_path = input_args.file_name
    sample_db_name = input_args.dataset_name
    to_predict = input_args.topredict
    histogram_file = data_path + '/histograms.csv'
    train_file = data_path + '/query_plans.csv'
    test_file = data_path + '/query_plans.csv'

    train_query_former()
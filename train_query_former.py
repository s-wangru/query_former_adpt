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

import json

def print_plan(plan):
    # pretty print the json object

    # the following code parses the json string into a dictionary
    json_parsed = json.loads(plan)
    json_pretty = json.dumps(json_parsed, indent=4)
    print(json_pretty)

    with open('output.json', 'w') as f:
        f.write(json_pretty)


def print_qerror(preds_unnorm, labels_unnorm):
    qerror = []
    for i in range(len(preds_unnorm)):
        print("Predicted: {}, Actual: {}".format(preds_unnorm[i], labels_unnorm[i]))
        if preds_unnorm[i] > float(labels_unnorm[i]):
            qerror.append(preds_unnorm[i] / float(labels_unnorm[i]))
        else:
            qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))

    e_50, e_90 = np.median(qerror), np.percentile(qerror,90)    
    e_mean = np.mean(qerror)
    print("Median: {}".format(e_50))
    print("90th percentile: {}".format(e_90))
    print("Mean: {}".format(e_mean))
    return 

def get_corr(ps, ls): # unnormalised
    ps = np.array(ps)
    ls = np.array(ls)
    corr, _ = pearsonr(np.log(ps), np.log(ls))
    
    return corr

def evaluate(model, ds, bs, norm, device):
    model.eval()
    cost_predss = np.empty(0)

    with torch.no_grad():
        for i in range(0, len(ds), bs):
            batch, batch_labels = collator(list(zip(*[ds[j] for j in range(i,min(i+bs, len(ds)) ) ])))

            batch = batch.to(device)

            cost_preds, _ = model(batch)

            cost_preds = cost_preds.squeeze()

            cost_predss = np.append(cost_predss, cost_preds.cpu().detach().numpy())

            
    print_qerror(norm.unnormalize_labels(cost_predss), ds.costs)
    corr = get_corr(norm.unnormalize_labels(cost_predss), ds.costs)
    print('Corr: ', corr)

    return 

def eval_workload(workload, methods):

    get_table_sample = methods['get_sample']

    workload_file_name = data_path + '/' + workload
    output_file_name = data_path + '/{}_output.csv'.format(workload)

    table_sample = get_table_sample(workload_file_name)

    plan_df = pd.read_csv(f'{data_path}/test_query_plans.csv')
    print_plan(plan_df['json'][0])
    workload_csv = pd.read_csv(f'{data_path}/{workload}.csv',sep='@',header=None)
    workload_csv.columns = ['table','join','predicate','cardinality']

    workload_csv.to_csv(output_file_name, index=False)

    
    ds = PlanTreeDataset(plan_df, workload_csv, \
        methods['encoding'], methods['hist_file'], methods['cost_norm'], \
        methods['cost_norm'], 'cost', table_sample)
    

    evaluate(methods['model'], ds, methods['bs'], methods['cost_norm'], methods['device'])
    return 

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

    dfs = []  # list to hold train df

    df = pd.read_csv(train_file)
    dfs.append(df)

    full_train_df = pd.concat(dfs)

    val_dfs = []  # list to hold test df

    df = pd.read_csv(test_file)
    val_dfs.append(df)

    val_df = pd.concat(val_dfs)

    train_table_sample = get_job_table_sample(data_path + '/' + dataset + '_train')
    test_table_sample = get_job_table_sample(data_path + '/' + dataset + '_test')

    train_ds = PlanTreeDataset(full_train_df, None, encoding, hist_file, card_norm, cost_norm, to_predict, train_table_sample)
    val_ds = PlanTreeDataset(val_df, None, encoding, hist_file, card_norm, cost_norm, to_predict, test_table_sample)

    crit = nn.MSELoss()
    model, best_path = train(model, train_ds, val_ds, crit, cost_norm, args)
    methods = {
        'get_sample' : get_job_table_sample,
        'encoding': encoding,
        'cost_norm': cost_norm,
        'hist_file': hist_file,
        'model': model,
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'bs': 512,
    }
    eval_workload(dataset + '_test', methods)

        
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
    train_file = data_path + '/train_query_plans.csv'
    test_file = data_path + '/test_query_plans.csv'

    train_query_former()
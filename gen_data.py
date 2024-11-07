import os
import sys
import json
import csv
import argparse
import threading
from typing import List, Dict, Any, Optional

import psycopg2
import pandas as pd
import numpy as np
import torch
from sqlalchemy import create_engine
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from torch.utils.data import Dataset


model_name = "meta-llama/Llama-2-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("llama_2.log"),
        logging.StreamHandler()
    ]
)


def read_queries_from_file(file_path: str) -> List[str]:
    """
    Reads SQL queries from a file, ignoring comments and empty lines.
    Each query should end with a semicolon.
    """
    queries = []
    current_query = []

    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()

                if line.startswith('--') or not line:
                    continue

                current_query.append(line)

                if line.endswith(';'):
                    queries.append(' '.join(current_query))
                    current_query = []  # Reset for the next query

        logging.info(f"Total Queries Read: {len(queries)}")
        return queries
    except FileNotFoundError:
        logging.error(f"Error: File not found at {file_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred while reading queries: {e}")
        sys.exit(1)


def get_query_plan(query: str, db_config: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """
    Executes the given SQL query with EXPLAIN ANALYZE and returns the execution plan.
    """
    try:
        with psycopg2.connect(**db_config) as conn:
            with conn.cursor() as cur:
                cur.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {query}")
                plan = cur.fetchone()[0]  # fetchone() returns a tuple
                return plan[0]
    except Exception as e:
        logging.error(f"Error getting plan for query: {query}\n{e}")
        return None
    
def save_query_plans(queries: List[str], db_config: Dict[str, str], output_csv: str):
    query_plans = []
    for i, query in enumerate(queries):
        plan = get_query_plan(query, db_config)
        if plan:
            query_plans.append({"id": i, "json": json.dumps(plan), "cost": plan['Plan']['Total Cost']})

    full_train_df = pd.DataFrame(query_plans)
    full_train_df.to_csv(output_csv, index=False)
    logging.info(f"Saved query execution plans to '{output_csv}'. Total plans: {len(query_plans)}")
    logging.info(full_train_df.head())


def extract_minmax(table: str, column_min_max_vals: Dict[str, List[Any]]) -> int:
    full_path = os.path.join('tpch-kit/dbgen/outputs', f"{table}.tbl")
    try:
        # Read the table file
        df = pd.read_table(full_path, delimiter='|', header=None, index_col=False)
    except FileNotFoundError:
        logging.error(f"Error: Data file not found at {full_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred while reading {full_path}: {e}")
        sys.exit(1)

    # Extract min-max values for each column
    for col in df.columns:
        full_column_name = f"{table}.{col}"
        min_val = df[col].min()
        max_val = df[col].max()

        # Convert string values to integers using UTF-8 encoding
        if isinstance(min_val, str) or isinstance(max_val, str):
            min_val = int.from_bytes(min_val.encode('utf-8'), 'big')
            max_val = int.from_bytes(max_val.encode('utf-8'), 'big')

        column_min_max_vals[full_column_name] = [min_val, max_val]

    # Generate a formatted string for static context
    context_str = f"TABLE: {table}\n"
    context_str += "COLUMN MIN-MAX VALUES:\n"
    for column, (min_val, max_val) in column_min_max_vals.items():
        context_str += f"{column}: Min = {min_val}, Max = {max_val}\n"

    return context_str

def parse_histogram_file(file_path: str) -> str:

    # Read the file into a DataFrame
    df = pd.read_csv(file_path)

    # Initialize the context string
    context_str = "HISTOGRAM DATA:\n"

    for _, row in df.iterrows():
        table = row['table']
        column = row['column']
        bins = row['bins']
        table_column = row['table_column']
        freq = row['freq']
        
        # Decode bins and frequencies
        bins_list = bins.strip("'").split()  # Convert string of bins to a list
        freq_list = [int(freq[i:i+16], 16) for i in range(0, len(freq), 16)]  # Split freq string into 16-char chunks

        # Format the histogram data for the current column
        context_str += f"\nTABLE: {table}\nCOLUMN: {column}\n"
        context_str += "BINS & FREQUENCIES:\n"

        # Pair each bin with its frequency and format as text
        for bin_val, freq_val in zip(bins_list, freq_list):
            context_str += f"  Bin: {bin_val}, Frequency: {freq_val}\n"
    
    return context_str

def static_context():
    hist = parse_histogram_file('histogram.csv')
    tables = ['customer', 'lineitem', 'nation', 'orders', 'part', 'partsupp', 'region', 'supplier']
    minmax = ""
    for t in tables:
        minmax += extract_minmax(t, {})
    return hist + '\n' + minmax

static_tokens = tokenizer(static_context(), return_tensors="pt", truncation=True)
with torch.no_grad():
    static_embedding = base_model(**static_tokens).last_hidden_state.mean(dim=1)  # Mean pooling for fixed size


class QueryPlanDataset(Dataset):
    def __init__(self, csv_file):
        # Load the CSV file
        df = pd.read_csv(csv_file)
        self.query_plans = df['plan'].apply(json.loads)  # Convert JSON string to dictionary
        self.costs = df['cost'].values  # Costs as labels

    def __len__(self):
        return len(self.costs)

    def __getitem__(self, idx):
        query_plan = self.query_plans[idx]
        cost = self.costs[idx]

        plan_text = json.dumps(query_plan, indent=2)  # Pretty-print JSON

        query_tokens = tokenizer(plan_text, return_tensors="pt", truncation=True, max_length=4096, padding="max_length")
        query_embeddings = base_model.transformer.wte(query_tokens.input_ids).squeeze(0)  # Convert to embeddings

        combined_embeddings = torch.cat([static_embedding, query_embeddings], dim=1)
        
        # Attention mask: 1 for tokens in combined embeddings, 0 for padding
        attention_mask = torch.cat([torch.ones(static_embedding.shape[1]), query_tokens.attention_mask.squeeze(0)], dim=0)

        return {"inputs_embeds": query_embeddings, "attention_mask": attention_mask, "labels": torch.tensor(cost, dtype=torch.float)}


csv_file = "tpch-kit/dbgen/tpch-stream.csv"
dataset = QueryPlanDataset(csv_file)


class LLaMAForCostPrediction(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.regression_head = torch.nn.Linear(base_model.config.hidden_size, 1)  # Output for cost

    def forward(self, inputs_embeds, attention_mask, labels=None):
        outputs = self.base_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state[:, -1, :]

        # Predict cost
        cost_pred = self.regression_head(last_hidden_state)

        loss = None
        if labels is not None:
            loss = torch.nn.functional.mse_loss(cost_pred.view(-1), labels.view(-1))

        return {"loss": loss, "logits": cost_pred} if loss is not None else {"logits": cost_pred}

model = LLaMAForCostPrediction(base_model)


training_args = TrainingArguments(
    output_dir="./llama_cost_predictor",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_train_epochs=3,
    fp16=True,  # Mixed precision if supported
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    compute_metrics=None, 
)


trainer.train()
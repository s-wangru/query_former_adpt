# some problems: 
#   - out of memory issues
#   - runs kinda slow
#   - var length input, temporarily solved with padding
#   - how to add the metadata more efficiently

import json
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from typing import List, Dict, Any, Optional
import os, sys
import logging

def extract_minmax(table: str, column_min_max_vals: Dict[str, List[Any]]) -> int:
    full_path = os.path.join('tpch-kit/dbgen/output', f"{table}.tbl")
    try:
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


        if isinstance(min_val, str) or isinstance(max_val, str):
            min_val = int.from_bytes(min_val.encode('utf-8'), 'big')
            max_val = int.from_bytes(max_val.encode('utf-8'), 'big')

        column_min_max_vals[full_column_name] = [min_val, max_val]


    context_str = f"TABLE: {table}\n"
    context_str += "COLUMN MIN-MAX VALUES:\n"
    for column, (min_val, max_val) in column_min_max_vals.items():
        context_str += f"{column}: Min = {min_val}, Max = {max_val}\n"

    return context_str

def parse_histogram_file(file_path: str) -> str:
  
    df = pd.read_csv(file_path)

    context_str = "HISTOGRAM DATA:\n"

    for _, row in df.iterrows():
        table = row['table']
        column = row['column']
        bins = row['bins']
        table_column = row['table_column']
        freq = row['freq']
        
        bins_list = bins.strip("'").split()
        freq_list = [int(freq[i:i+16], 16) for i in range(0, len(freq), 16)]  


        context_str += f"\nTABLE: {table}\nCOLUMN: {column}\n"
        context_str += "BINS & FREQUENCIES:\n"

        for bin_val, freq_val in zip(bins_list, freq_list):
            context_str += f"  Bin: {bin_val}, Frequency: {freq_val}\n"
    
    return context_str

def static_context():
    hist = parse_histogram_file('tpch10_data/histograms.csv')
    tables = ['customer', 'lineitem', 'nation', 'orders', 'part', 'partsupp', 'region', 'supplier']
    minmax = ""
    for t in tables:
        minmax += extract_minmax(t, {})
    return {"histogram": hist, "minmax": minmax}

static_context = static_context()




class QueryPlanDataset(Dataset):
    def __init__(self, file_path, tokenizer, static_context_tokenized, max_length=1024):
        self.tokenizer = tokenizer
        self.static_context_tokenized = static_context_tokenized
        self.max_length = max_length

        df = pd.read_csv(file_path)
        self.data = []
        for _, row in df.iterrows():
            query_plan = json.dumps(json.loads(row["json"]))
            execution_time = json.loads(row["json"])["Execution Time"]
            self.data.append({
                "query_plan": query_plan,
                "cost": execution_time,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query_plan = item["query_plan"]
        cost = item["cost"]

        query_plan_tokenized = self.tokenizer(
            f"Query Plan:\n{query_plan}\nCost:",
            max_length=self.max_length - self.static_context_tokenized["input_ids"].shape[1],
            truncation=True,
            padding=False,
            return_tensors="pt"
        )


        input_ids = torch.cat(
            [self.static_context_tokenized["input_ids"][0], query_plan_tokenized["input_ids"].squeeze(0)],
            dim=0
        )
        attention_mask = torch.cat(
            [self.static_context_tokenized["attention_mask"][0], query_plan_tokenized["attention_mask"].squeeze(0)],
            dim=0
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(cost, dtype=torch.float)
        }


file_path = "tpch10_data/query_plans.csv"

model_name = "meta-llama/Llama-3.2-3B" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

static_context_tokenized = tokenizer(
    f"Static Context:\n{static_context}\n\n",
    return_tensors="pt",
    max_length=1024,
    padding="max_length",
    truncation=True
)

dataset = QueryPlanDataset(
    file_path = file_path,
    tokenizer=tokenizer,
    static_context_tokenized=static_context_tokenized
)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])


def data_collator(features):
    """
    Pads input_ids, attention_mask, and processes labels for batching.
    """
    input_ids = [feature["input_ids"] for feature in features]
    attention_mask = [feature["attention_mask"] for feature in features]
    labels = [feature["labels"] for feature in features]

    batch = tokenizer.pad(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        padding=True,
        return_tensors="pt",
    )

    batch["labels"] = torch.tensor(labels, dtype=torch.float)
    return batch

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

try:
    trainer.train()
except RuntimeError as e:
    if "out of memory" in str(e):
        print("Caught PyTorch OOM error.")
        torch.cuda.empty_cache()
    else:
        print(str(e))

from torch.utils.data import DataLoader

eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False)
model.eval()
all_labels = []
all_predictions = []

with torch.no_grad():
    for batch in eval_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)  

        # get predictions
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = outputs.logits.squeeze(-1) 


        all_predictions.extend(predictions.cpu().numpy())
        all_actuals.extend(labels.cpu().numpy())

# calculate qerror for each query
q_errors = [
    max(pred / actual, actual / pred)
    for pred, actual in zip(all_predictions, all_actuals)
    if actual > 0 and pred > 0 
]

mean_q_error = sum(q_errors) / len(q_errors)
median_q_error = np.median(q_errors)

print(f"Mean Q-Error: {mean_q_error}")
print(f"Median Q-Error: {median_q_error}")


model.save_pretrained("/mnt/nvme0n1/ruiqiwan/cost_model_llama")
tokenizer.save_pretrained("/mnt/nvme0n1/ruiqiwan/cost_model_llama")

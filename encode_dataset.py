#!/usr/bin/env python3
# transformer_dependency.py

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

# Import custom classes/modules
# Ensure that 'model.database_util' is in your PYTHONPATH or the same directory
from model.database_util import Encoding

# Configure logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("query_former.log"),
        logging.StreamHandler()
    ]
)

# Configuration Constants (Default Values)
DEFAULT_QUERY_FILE_PATH = 'tpch-kit/dbgen/tpch-stream.sql'
DEFAULT_QUERY_PLANS_CSV = 'query_plans.csv'
DEFAULT_HISTOGRAMS_CSV = 'histograms.csv'
DEFAULT_BITMAP_CSV_FILE = "tpch.bitmaps"
DEFAULT_ENCODING_CHECKPOINT = 'checkpoints/tpch_encoding.pt'
DEFAULT_DB_NAME = 'tpch_sample'
DEFAULT_DB_USER = 'postgres'
DEFAULT_DB_PASSWORD = 'admin'
DEFAULT_DB_HOST = '127.0.0.1'
DEFAULT_DB_PORT = '5432'
DEFAULT_SAMPLE_SIZE = 1000
query_plans_csv = 'tpch_query_plans.csv'
datapath = "tpch10_data"

# Database Aliases and Mappings
DB_ALIAS = {
    'nation': 'n',
    'region': 'r',
    'supplier': 's',
    'part': 'p',
    'partsupp': 'ps',
    'customer': 'c',
    'orders': 'o',
    'lineitem': 'l'
}

ALIAS_TO_DB = {v: k for k, v in DB_ALIAS.items()}

COLUMN_MAPPING = {
    "nation": {
        0: "n_nationkey",
        1: "n_name",
        2: "n_regionkey",
        3: "n_comment"
    },
    "region": {
        0: "r_regionkey",
        1: "r_name",
        2: "r_comment"
    },
    "part": {
        0: "p_partkey",
        1: "p_name",
        2: "p_mfgr",
        3: "p_brand",
        4: "p_type",
        5: "p_size",
        6: "p_container",
        7: "p_retailprice",
        8: "p_comment"
    },
    "supplier": {
        0: "s_suppkey",
        1: "s_name",
        2: "s_address",
        3: "s_nationkey",
        4: "s_phone",
        5: "s_acctbal",
        6: "s_comment"
    },
    "partsupp": {
        0: "ps_partkey",
        1: "ps_suppkey",
        2: "ps_availqty",
        3: "ps_supplycost",
        4: "ps_comment"
    },
    "customer": {
        0: "c_custkey",
        1: "c_name",
        2: "c_address",
        3: "c_nationkey",
        4: "c_phone",
        5: "c_acctbal",
        6: "c_mktsegment",
        7: "c_comment"
    },
    "orders": {
        0: "o_orderkey",
        1: "o_custkey",
        2: "o_orderstatus",
        3: "o_totalprice",
        4: "o_orderdate",
        5: "o_orderpriority",
        6: "o_clerk",
        7: "o_shippriority",
        8: "o_comment"
    },
    "lineitem": {
        0: "l_orderkey",
        1: "l_partkey",
        2: "l_suppkey",
        3: "l_linenumber",
        4: "l_quantity",
        5: "l_extendedprice",
        6: "l_discount",
        7: "l_tax",
        8: "l_returnflag",
        9: "l_linestatus",
        10: "l_shipdate",
        11: "l_commitdate",
        12: "l_receiptdate",
        13: "l_shipinstruct",
        14: "l_shipmode",
        15: "l_comment"
    }
}

COLUMN_MIN_MAX_VALS = {}
COL2IDX = {}
CURRENT_INDEX = 0



def read_queries_from_file(file_path: str) -> List[str]:

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

    with psycopg2.connect(**db_config) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {query}")
                plan = cur.fetchone()[0]  # fetchone() returns a tuple
                return plan[0]
            except Exception as e:
                try:
                    cur.execute(query)
                    logging.error(f"Not supported for explain: {query}")
                    return None
                except Exception as e:
                    print(e)


def save_query_plans(queries: List[str], db_config: Dict[str, str], output_csv: str):

    query_plans = []
    i = 0
    for query in queries:
        plan = get_query_plan(query, db_config)
        if plan:
            query_plans.append({"id": i, "json": json.dumps(plan)})
            i += 1
            # query_plans.append({"id": i, "json": json.dumps(plan), "sql": query})

    full_train_df = pd.DataFrame(query_plans)
    full_train_df.to_csv(output_csv, index=False)
    logging.info(f"Saved query execution plans to '{output_csv}'. Total plans: {len(query_plans)}")
    logging.info(full_train_df.head())


def load_and_validate_query_plans(input_csv: str) -> pd.DataFrame:
    """
    Loads query plans from a CSV file and validates the JSON structure.
    """
    try:
        full_train_df = pd.read_csv(input_csv)

        for i, row in full_train_df.iterrows():
            try:
                json_obj = json.loads(row['json'])

                full_train_df.at[i, 'json'] = json.dumps(json_obj)
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON for id {row['id']}: {e}")
                # Optionally, drop the row if the JSON is not valid
                full_train_df.drop(i, inplace=True)

        logging.info(f"Loaded and validated query plans from '{input_csv}'.")
        logging.info(f"DataFrame Shape: {full_train_df.shape}")
        logging.info(full_train_df.head())
        return full_train_df
    except FileNotFoundError:
        logging.error(f"Error: File not found at {input_csv}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred while loading query plans: {e}")
        sys.exit(1)


def extract_minmax(table: str, column_min_max_vals: Dict[str, List[Any]], col2idx: Dict[str, int],
                  current_index: int) -> int:
    """
    Extracts min and max values for each column in the specified table and updates the mappings.
    """
    global COLUMN_MIN_MAX_VALS, COL2IDX, CURRENT_INDEX

    full_path = os.path.join('tpch-kit/dbgen/output', f"{table}.tbl") # @TODO: make path customizable
    try:
        df = pd.read_table(full_path, delimiter='|', header=None, index_col=False)
    except FileNotFoundError:
        logging.error(f"Error: Data file not found at {full_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred while reading {full_path}: {e}")
        sys.exit(1)

    alias = DB_ALIAS.get(table)
    if not alias:
        logging.error(f"Error: No alias found for table '{table}'.")
        sys.exit(1)

    for col in df.columns:
        col_name = COLUMN_MAPPING[table][col]
        full_column_name = f"{alias}.{col_name}"

        min_val = df[col].min()
        max_val = df[col].max()

        if isinstance(min_val, str) or isinstance(max_val, str):
            min_val = int.from_bytes(min_val.encode('utf-8'), 'big')
            max_val = int.from_bytes(max_val.encode('utf-8'), 'big')

        column_min_max_vals[full_column_name] = [min_val, max_val]

        if full_column_name not in col2idx:
            col2idx[full_column_name] = current_index
            current_index += 1

    if 'NA' not in col2idx:
        col2idx['NA'] = current_index
        current_index += 1

    return current_index


def process_schema_information() -> (Dict[str, List[Any]], Dict[str, int]):

    column_min_max_vals = {}
    col2idx = {}
    current_index = 0

    tables = ['customer', 'lineitem', 'nation', 'orders', 'part', 'partsupp', 'region', 'supplier']

    for table in tables:
        current_index = extract_minmax(table, column_min_max_vals, col2idx, current_index)

    return column_min_max_vals, col2idx


def save_encoding(encoding: Encoding, checkpoint_path: str):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({'encoding': encoding}, checkpoint_path)
    logging.info(f"Encoding saved to '{checkpoint_path}'.")


def load_encoding(checkpoint_path: str) -> Encoding:
    try:
        encoding_ckpt = torch.load(checkpoint_path)
        encoding = encoding_ckpt['encoding']
        logging.info(f"Encoding loaded from '{checkpoint_path}'.")
        return encoding
    except FileNotFoundError:
        logging.error(f"Error: Checkpoint file not found at {checkpoint_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred while loading encoding: {e}")
        sys.exit(1)


def create_histograms(column_mapping: Dict[str, Dict[int, str]], db_alias: Dict[str, str],
                     db_config: Dict[str, str], output_csv: str):
    """
    Creates histograms for each column in the database tables and saves them to a CSV file.
    """
    try:
        with psycopg2.connect(**db_config) as conn:
            with conn.cursor() as cur:
                hist_file = pd.DataFrame(columns=['table', 'column', 'bins', 'table_column', 'freq'])

                for table, alias in db_alias.items():
                    for col_idx, col_name in column_mapping[table].items():
                        cmd = f'SELECT {col_name} FROM {table}'
                        cur.execute(cmd)
                        col = cur.fetchall()
                        col_array = to_vals(col)

                        if col_array.size == 0:
                            logging.warning(f"Warning: No data found for column '{col_name}' in table '{table}'.")
                            continue

                        hists = np.nanpercentile(col_array, range(0, 101, 2), axis=0)
                        freq = np.histogram(col_array, bins=hists)[0]

                        freq_hex = freq.astype('float').tobytes().hex()

                        res_dict = {
                            'table': table,
                            'column': col_name,
                            'bins': repr((' ').join([str(int(i)) for i in hists])), 
                            'freq': freq_hex,
                            'table_column': f'{alias}.{col_name}'
                        }
                        hist_file = pd.concat([hist_file, pd.DataFrame([res_dict])], ignore_index=True)

                hist_file.to_csv(output_csv, index=False)
                logging.info(f"Histograms saved to '{output_csv}'.")
    except Exception as e:
        logging.error(f"An error occurred while creating histograms: {e}")
        sys.exit(1)


def to_vals(data_list: List[tuple]) -> np.ndarray:
    """
    Converts a list of tuples from database fetchall to a NumPy array with appropriate data types.
    """
    for dat in data_list:
        val = dat[0]
        if val is not None:
            break

    try:
        float(val)
        return np.array([item[0] for item in data_list], dtype=float)
    except (ValueError, TypeError):
        res = []
        for dat in data_list:
            val = dat[0]
            if val is None:
                res.append(0)
            elif isinstance(val, str):
                hex_value = int.from_bytes(val.encode('utf-8'), 'big') % int(1e9)
                res.append(hex_value)
            else:
                try:
                    mi = val.timestamp()
                except AttributeError:
                    mi = 0
                res.append(mi)
        return np.array(res, dtype=float)


def parse_predicate(predicate: str) -> List[str]:
    """
    Parses a predicate string and returns a list of parsed conditions in the format "column,operator,value".
    """
    parsed_conditions = []
    print(predicate)
    predicate = predicate[1: (len(predicate)-1)]

    conditions = predicate.split(' AND ')
    
    for condition in conditions:
        condition = condition.replace('(', '').replace(')', '').strip()
        # front = 0
        # while condition[front] == '(':
        #     front += 1
        # back = 0
        # while condition[len(condition)-1-back] == ')':
        #     back += 1
        # rem = min(front, back)
        # condition = condition[rem : len(condition)-rem]

        # Extract column, operator, value
        for operator in ['>=', '<=', '<>', '=', '>', '<', 'LIKE']:
                if operator in condition:
                    parts = condition.split(operator)
                    if len(parts) == 2:
                        column, value = parts
                        column = column.strip()
                        value = value.strip()
                        prefixed_column = column
                        if '.' not in column and len(column.split('_')) > 1:
                            prefix = column.split('_')[0]
                            if prefix[0] == '(': 
                                prefix = prefix[1:]
                            prefixed_column = prefix + '.' + column
                        parsed_conditions.append(f"{prefixed_column}&{operator}&{value}")
                    break

    return parsed_conditions


def parse_query_plan(query_plan: Dict[str, Any]) -> str:

    tables = set()
    joins = []
    predicates = []
    cardinality = None

    def traverse_plan(plan: Dict[str, Any]):
        nonlocal cardinality
        node_type = plan.get('Node Type', '')

        if node_type in ['Seq Scan', 'Index Scan']:
            relation_name = plan.get('Relation Name', '')
            alias = plan.get('Alias', '')
            if relation_name and alias:
                tables.add(f"{relation_name} {alias}")

        if node_type in ['Nested Loop', 'Hash Join', 'Merge Join']:
            join_condition = plan.get('Join Filter', plan.get('Hash Cond', ''))
            if join_condition:
                joins.append(join_condition)

        if 'Filter' in plan:
            filter_condition = plan['Filter']
            parsed_predicates = parse_predicate(filter_condition)
            predicates.extend(parsed_predicates)

        if 'Actual Rows' in plan:
            cardinality = plan['Actual Rows']
        elif 'Plan Rows' in plan and cardinality is None:
            cardinality = plan['Plan Rows']

        if 'Plans' in plan:
            for subplan in plan['Plans']:
                traverse_plan(subplan)

    traverse_plan(query_plan)

    # Format the result
    table_str = '&'.join(tables)
    join_str = '&'.join(joins)
    predicate_str = '&'.join(predicates)
    card_str = str(cardinality) if cardinality else '1'

    return f"{table_str}@{join_str}@{predicate_str}@{card_str}"


def parse_all_query_plans(query_plans: List[Dict[str, str]]) -> pd.DataFrame:
    """
    Parses all query execution plans and returns a DataFrame.
    """
    parsed_results = []

    for query_plan in query_plans:
        plan_dict = json.loads(query_plan['json'])
        parsed_result = parse_query_plan(plan_dict['Plan'])
        parsed_results.append(parsed_result)

    df = pd.DataFrame(parsed_results, columns=["parsed_plan"])
    df.to_csv(datapath + '/tpch.csv', index=False, sep='@', header=False, quoting=csv.QUOTE_NONE, escapechar=' ')
    logging.info("Parsed query plans have been saved to 'tpch.csv'.")
    return df


def create_sample_data(db_alias: Dict[str, str], column_mapping: Dict[str, Dict[int, str]], sample_size: int,
                       db_config: Dict[str, str], sampledb_config):
    """
    Creates sample data by sampling rows from each table and saves them to a new database.
    """
    try:
        # Connect to the original database to fetch data
        with psycopg2.connect(**db_config) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS tsm_system_rows")
                sample_data = {}
                tables = list(db_alias.keys())

                for table in tables:
                    cur.execute(f"SELECT * FROM {table} LIMIT 0")
                    colnames = [desc[0] for desc in cur.description]

                    ts = pd.DataFrame(columns=colnames)

                    for num in range(sample_size):
                        cur.execute(f"SELECT * FROM {table} TABLESAMPLE SYSTEM_ROWS(1)")
                        samples = cur.fetchall()
                        if samples:
                            ts.loc[num] = samples[0]

                    sample_data[table] = ts

        # Connect to the sample database
        engine = create_engine(f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}" + "_sample")

        for table, df in sample_data.items():
            df['sid'] = list(range(sample_size))
            try:
                with psycopg2.connect(**sampledb_config) as conn:
                    with conn.cursor() as cur:
                        cur.execute(f'ALTER TABLE {table} ADD COLUMN sid INTEGER')
                # Append data to the sample database
                df.to_sql(table, engine, if_exists='append', index=False)
                logging.info(f"Sample data for table '{table}' inserted into sample database.")
            except Exception as e:
                logging.error(f"Error inserting sample data for table '{table}': {e}")
    except Exception as e:
        logging.error(f"An error occurred while creating sample data: {e}")
        sys.exit(1)


def create_bitmaps(alias_to_db: Dict[str, str], db_alias: Dict[str, str],
                  db_config: Dict[str, str], output_bitmap_file: str):

    with psycopg2.connect(**db_config) as conn:
        with conn.cursor() as cur:
            table_samples = []
            query_file = pd.read_csv(datapath + '/tpch10.csv',sep='@',header=None)
            query_file.columns = ['table','join','predicate','card']
            for i, row in query_file.iterrows():
                table_sample = {}
                preds = row['predicate'].split('&')
                special = False
                for j in range(0, len(preds), 3):
                    if j + 2 >= len(preds):
                        continue
                    left, op, right = preds[j:j + 3]
                    left = left.strip().split('.')
                    col = ''
                    if len(left) < 2:
                        col = left[0]
                        db = row['table']
                    elif len(left) > 2:
                        continue
                    else:
                        alias, col = left
                        if (alias not in alias_to_db):
                            continue
                        else:
                            db = alias_to_db[alias]
                        
                    pred_string = ''.join((col,op,right))
                    print(pred_string)
                    try:
                        q = 'select sid from {} where {}'.format(db, pred_string)
                        cur.execute(q)
                    except Exception as e:
                        continue
                    sps = np.zeros(1000).astype('uint8')
                    sids = cur.fetchall()
                    sids = np.array(sids).squeeze()
                    if sids.size>1:
                        sps[sids] = 1
                    if db in table_sample:
                        table_sample[db] = table_sample[db] & sps
                    else:
                        table_sample[db] = sps

                # Handle joins
                if pd.notnull(row['join']) and len(row['join']) > 2:
                    joins = row['join'].split('&')
                    for join in joins:
                        sp = join.split('=')
                        if (len(sp) < 2):
                            continue
                        alias1,alias2 = sp[0].strip().split(' ')[0],sp[1].strip().split(' ')[0]
                        alias1,alias2 = alias1.split('.')[0][1:],alias2.split('.')[0]
                        print(alias1,alias2)
                        if alias1 not in alias_to_db or alias2 not in alias_to_db:
                            continue
                        table1,table2 = alias_to_db[alias1],alias_to_db[alias2]
                        q = 'select {}.sid from {} {} join {} {} on {}'.format(alias1, table1, alias1, table2, alias2,join)
                        cur.execute(q)
                        sids = cur.fetchall()
                        sids = np.array(sids).squeeze()
                        if sids.size>1:
                            sps[sids] = 1
                        if alias1 in table_sample:
                            table_sample[alias1] = table_sample[alias1] & sps
                        else:
                            table_sample[alias1] = sps
                        # q = 'select {}.sid from {} {} join {} {} on {}'.format(alias2, table1, alias1, table2, alias2,join)
                        # cur.execute(q)
                        # sids = cur.fetchall()
                        # sids = np.array(sids).squeeze()
                        # if sids.size>1:
                        #     sps[sids] = 1
                        # if alias2 in table_sample:
                        #     table_sample[alias2] = table_sample[alias2] & sps
                        # else:
                        #     table_sample[alias2] = sps
                table_samples.append(table_sample)


    with open(output_bitmap_file, 'wb') as f:
        for table_sample in table_samples:
            print(table_sample.keys())
            num_tables = len(table_sample)
            f.write(num_tables.to_bytes(4, byteorder='little'))
            for table, bitmap in table_sample.items():
                num_bytes_per_bitmap = (len(bitmap) + 7) // 8
                bitmap_bytes = np.packbits(bitmap[:num_bytes_per_bitmap * 8])
                f.write(bitmap_bytes)

    logging.info(f"Bitmap file saved as '{output_bitmap_file}'.")

def main():
    # Initialize Argument Parser
    parser = argparse.ArgumentParser(description="Process SQL queries and generate query plans, histograms, and bitmaps.")
    parser.add_argument('--file-name', type=str, default=DEFAULT_QUERY_FILE_PATH,
                        help=f"Path to the SQL queries file (default: {DEFAULT_QUERY_FILE_PATH})")
    parser.add_argument('--dataset', type=str, default=DEFAULT_DB_NAME,
                        help=f"Name of the sample dataset/database (default: {DEFAULT_DB_NAME})")

    args = parser.parse_args()

    query_file_path = args.file_name
    sample_db_name = args.dataset
    datapath = sample_db_name + '_data'

    encoding_checkpoint = f'checkpoints/{sample_db_name}_encoding.pt'
    histograms_csv = f'{datapath}/histograms.csv'
    query_plans_csv = f'{datapath}/query_plans.csv'
    bitmap_csv_file = f"{datapath}/{sample_db_name}.bitmaps"

    db_config = {
        "database": sample_db_name,
        "user": DEFAULT_DB_USER,
        "password": DEFAULT_DB_PASSWORD,
        "host": DEFAULT_DB_HOST,
        "port": DEFAULT_DB_PORT
    }

    sampledb_config = {
        "database": sample_db_name + '_sample',
        "user": DEFAULT_DB_USER,
        "password": DEFAULT_DB_PASSWORD,
        "host": DEFAULT_DB_HOST,
        "port": DEFAULT_DB_PORT
    }

    queries = read_queries_from_file(query_file_path)


    save_query_plans(queries, db_config, query_plans_csv)


    full_train_df = load_and_validate_query_plans(query_plans_csv)

    column_min_max_vals, col2idx = process_schema_information()

    encoding = Encoding(
        column_min_max_vals=column_min_max_vals,
        col2idx=col2idx,
        op2idx={'=': 0, '>': 1, '<': 2, '>=': 3, '<=': 4, 'NA': 5}
    )

    for json_string in full_train_df['json']:
        query_plan = json.loads(json_string)['Plan']

        def traverse_plan(plan):
            encoding.encode_type(plan['Node Type'])
            if 'Plans' in plan:
                for subplan in plan['Plans']:
                    traverse_plan(subplan)

        traverse_plan(query_plan)

    save_encoding(encoding, encoding_checkpoint)

    encoding = load_encoding(encoding_checkpoint)

    # create_histograms(COLUMN_MAPPING, DB_ALIAS, db_config, histograms_csv)

    create_sample_data(DB_ALIAS, COLUMN_MAPPING, DEFAULT_SAMPLE_SIZE, db_config, sampledb_config)

    query_file = parse_all_query_plans(full_train_df.to_dict('records'))

    create_bitmaps(ALIAS_TO_DB, DB_ALIAS, sampledb_config, bitmap_csv_file)

if __name__ == "__main__":
    main()

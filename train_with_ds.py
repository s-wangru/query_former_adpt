import os
import pandas as pd

for i in range(9, 18):
    input_dir = f"./training_data/dev9/pg{i}/tpch/sf_10/explain_analyze/15721"
    output_dir = f"./training_data/pg{i}_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 
    train_output = output_dir + '/train_query_plans.csv'
    test_output = output_dir + '/test_query_plans.csv'
    train_data = []
    test_data = []

    j = 0
    for filename in os.listdir(input_dir):
        if not (filename == '15-1.res' or filename == '15-3.res'): # Only process text files
            file_path = os.path.join(input_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()  # Read and strip whitespace
                if j % 5 == 0:
                    test_data.append({"id": j, "json": text})
                else:
                    train_data.append({"id": j, "json": text})
                j += 1

    df = pd.DataFrame(train_data)

    df.to_csv(train_output, index=False)

    df = pd.DataFrame(test_data)

    df.to_csv(test_output, index=False)
    print(f"Combined CSV saved at: {output_dir}")

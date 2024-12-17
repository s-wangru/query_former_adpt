from pathlib import Path
import os
import pandas as pd
import json


def parse_filepath_tpch(filepath):
    parents = filepath.parents
    query = filepath.name.split("-")[0]
    seed = int(parents[0].name)
    config = parents[1].name
    sf = int(parents[2].name.split("_")[1])
    benchmark = parents[3].name
    version = parents[4].name
    
    ok = int(Path(f"{filepath}.ok").exists())
    timeout = int(Path(f"{filepath}.timeout").exists())
    res = ""
    if Path(f"{filepath}.res").exists():
        with open(Path(f"{filepath}.res"), "r") as f:
            res = json.dumps(json.load(f))

    return {
        "benchmark": benchmark,
        "sf": sf,
        "version": version,
        "seed": seed,
        "query": int(query),
        "explain": int(config in ["explain", "explain_analyze"]),
        "analyze": int(config in "explain_analyze"),
        "ok": ok,
        "timeout": timeout,
        "plan": res,
    }


def main():
    result_root = Path(os.getenv("RESULT_ARTIFACT_ROOT"))
    hostname = os.getenv("HOSTNAME")
    
    filepaths = []
    for pg_version in range(9, 17+1):
        for benchmark in ["tpch"]:
            for sf in [1, 10]:
                for config in ["explain", "explain_analyze"]:
                    for seed in [15721]:
                        folder = result_root / f"pg{pg_version}/{benchmark}/sf_{sf}/{config}/{seed}/"
                        for qnum in range(1, 22+1):
                            qsubnum = 1 if qnum != 15 else 2
                            filepaths.append(folder / f"{qnum}-{qsubnum}")

    data = []
    for filepath in filepaths:
        data.append(parse_filepath_tpch(filepath))

    df = pd.DataFrame(data)
    df["benchmark"] = df["benchmark"].astype("category")
    df["version"] = df["version"].astype("category")
    df["hostname"] = hostname
    df["hostname"] = df["hostname"].astype("category")
    save_path = result_root / f"result_{hostname}_tpch.pq"
    df.to_parquet(save_path)
    df = pd.read_parquet(save_path)


if __name__ == "__main__":
    main()

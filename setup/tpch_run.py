import json
import os
import time
import traceback
from pathlib import Path
from typing import Optional

import psycopg.errors
import sqlalchemy.exc
from sqlalchemy import Connection, Engine, NullPool, create_engine
from tpch_rewriter import *
from tqdm import tqdm, trange
from util import (
    conn_execute,
    connstr,
    prewarm_all,
    sql_file_queries,
    vacuum_analyze_all,
)


class Config:
    def __init__(
        self,
        expt_name: str,
        rewriter: Rewriter,
        timeout_s: int = 60 * 5,
        before_sql: Optional[list[str]] = None,
        after_sql: Optional[list[str]] = None,
    ):
        self.expt_name = expt_name
        self.timeout_s = timeout_s
        self.rewriter = rewriter
        self.before_sql = before_sql if before_sql is not None else []
        self.after_sql = after_sql if after_sql is not None else []


def tpch(engine: Engine, conn: Connection, config: Config, verbose=False):
    tpch_artifact_dir = Path(os.getenv("TPCH_ARTIFACT_DIR"))
    tpch_query_root = tpch_artifact_dir / "queries"
    tpch_output_dir = Path(os.getenv("TPCH_OUTPUT_DIR"))
    query_start = int(os.getenv("TPCH_QUERY_START"))
    query_stop = int(os.getenv("TPCH_QUERY_STOP"))

    prewarmed = False
    executed = False

    for seed in trange(
        query_start, query_stop + 1, desc=f"{config.expt_name} TPCH seed.", leave=None
    ):
        outdir = (
            tpch_output_dir
            / config.expt_name
            / str(seed)
        )
        outdir.mkdir(parents=True, exist_ok=True)

        for query_path in tqdm(
            [(tpch_query_root / str(seed) / f"{i}.sql") for i in range(1, 22 + 1)],
            desc=f"{config.expt_name} TPCH query.",
            leave=None,
        ):
            query_num = int(query_path.stem)
            for query_subnum, query in enumerate(
                sql_file_queries(query_path.absolute()), 1
            ):
                outpath_ok = outdir / f"{query_path.stem}-{query_subnum}.ok"
                outpath_res = outdir / f"{query_path.stem}-{query_subnum}.res"
                outpath_timeout = outdir / f"{query_path.stem}-{query_subnum}.timeout"

                if outpath_timeout.exists():
                    continue

                if outpath_ok.exists():
                    continue

                try:
                    with open(outpath_res, "w") as output_file:
                        if (query_num, query_subnum) == (15, 1):
                            conn_execute(
                                conn, "DROP VIEW IF EXISTS revenue0", verbose=False
                            )

                        query, executes, has_result = config.rewriter.rewrite(
                            query
                        )

                        if executes and not prewarmed:
                            conn_execute(
                                conn, f"SET statement_timeout = '0s'", verbose=verbose
                            )
                            prewarm_all(engine, conn, verbose=verbose)
                            vacuum_analyze_all(engine, conn, verbose=verbose)
                            for sql in config.before_sql:
                                conn_execute(conn, sql, verbose=verbose)
                            conn_execute(
                                conn,
                                f"SET statement_timeout = '{config.timeout_s}s'",
                                verbose=verbose,
                            )
                            prewarmed = True

                        result = conn_execute(conn, query, verbose=False)
                        if executes:
                            executed = True
                        if has_result:
                            result = str(result.fetchone()[0][0])
                            # TODO(WAN): :)
                            result = json.dumps(eval(result))
                            print(result, file=output_file)
                        outpath_ok.touch(exist_ok=True)

                except sqlalchemy.exc.OperationalError as e:
                    if isinstance(e.orig, psycopg.errors.QueryCanceled):
                        outpath_timeout.touch(exist_ok=True)
                        outpath_ok.touch(exist_ok=True)
                    else:
                        raise e

    if executed:
        for sql in config.after_sql:
            conn_execute(conn, sql, verbose=verbose)


def main():
    engine: Engine = create_engine(
        connstr(),
        poolclass=NullPool,
        execution_options={"isolation_level": "AUTOCOMMIT"},
    )

    def make_config(
        name
    ):
        return Config(
            expt_name=name,
            before_sql=[
                # "DROP EXTENSION IF EXISTS boot",
                # "CREATE EXTENSION IF NOT EXISTS boot",
            ],
            after_sql=[
                # "SELECT boot_disconnect()",
            ],
        )

    configs = [
        Config(expt_name="explain", rewriter=ExplainRewriter()),
        Config(expt_name="explain_analyze", rewriter=ExplainAnalyzeRewriter()),
    ]

    pbar = tqdm(range(len(configs)), desc="Configs.", leave=None)
    for config in configs:
        pbar.set_description(f"Config: {config.expt_name} {connstr()}")
        try:
            with engine.connect() as conn:
                tpch(engine, conn, config)
        except Exception:
            traceback.print_exc()
            print(f"ERROR FOR CONFIG: {config.expt_name}")
            pass
        pbar.update()
    pbar.close()


if __name__ == "__main__":
    main()

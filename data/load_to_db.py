"""
Bulk-load data/raw/orders.parquet into Postgres using COPY (fast).

Run:  make up && python -m data.load_to_db
"""

from pathlib import Path

import pandas as pd
import psycopg

from rto_guard.config import settings

SRC = Path("data/raw/orders.parquet")


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"{SRC} not found. Run `python -m data.generate` first.")

    df = pd.read_parquet(SRC)
    cols = list(df.columns)

    with psycopg.connect(settings.pg_dsn) as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE orders")
        with cur.copy(f"COPY orders ({', '.join(cols)}) FROM STDIN") as copy:
            for row in df.itertuples(index=False, name=None):
                copy.write_row(row)
        conn.commit()
        cur.execute("SELECT count(*), avg(is_rto::int) FROM orders")
        n, rate = cur.fetchone()

    print(f"loaded {n:,} orders into Postgres (RTO rate {float(rate):.1%})")


if __name__ == "__main__":
    main()

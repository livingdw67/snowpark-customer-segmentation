"""Download UCI Online Retail II and convert both Excel sheets to one parquet file.

Rows are kept exactly as published. The two sheets overlap in December 2010, so each row
carries SOURCE_SHEET and the overlap is resolved in the staging layer, not here.
"""
import io
import os
import urllib.request
import zipfile

import pandas as pd

from common import PROJECT_DIR

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
OUT_PATH = os.path.join(RAW_DIR, "online_retail_ii.parquet")


def fetch() -> pd.DataFrame:
    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL) as resp:
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))
    xlsx = archive.read("online_retail_II.xlsx")

    # Description holds some numeric-looking values, so text columns are read as str.
    sheets = pd.read_excel(
        io.BytesIO(xlsx), sheet_name=None,
        dtype={"Invoice": str, "StockCode": str, "Description": str, "Country": str},
    )
    frames = []
    for name, df in sheets.items():
        df = df.rename(columns={"Customer ID": "CustomerID"})
        df["SourceSheet"] = name
        frames.append(df)
        print(f"  {name}: {len(df):,} rows, {df.InvoiceDate.min()} to {df.InvoiceDate.max()}")
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    df = fetch()
    df.columns = [c.upper() for c in df.columns]
    df.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} rows to {OUT_PATH}")

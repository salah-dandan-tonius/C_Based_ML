import polars as pl
from pathlib import Path

input_dir = Path("Data/CSV")
output_dir = Path("Data/Parquet")
output_dir.mkdir(exist_ok=True)

files = list(input_dir.glob("orion-pipeline*.csv"))

for f in files:
    df = pl.read_csv(f)

    df = df.select([
        "SourceIP", "Port", "Traffic", "Packets", "Bytes",
        "UniqueDests", "UniqueDest24s", "Lat", "Long",
        "Country", "ASN", "TCP", "ICMP", "EventType"
    ])

    # Reduce / specify the bit precisions for reduced memory usage. For example,
    # a port will never be above 65535, so it only needs to be uint16 instead of
    # uint64.
    df = df.with_columns([
        pl.col("Port").cast(pl.UInt16),
        pl.col("Traffic").cast(pl.UInt8),
        pl.col("Packets").cast(pl.UInt32),
        pl.col("Bytes").cast(pl.UInt32),
        pl.col("UniqueDests").cast(pl.UInt32),
        pl.col("UniqueDest24s").cast(pl.UInt32),
        pl.col("Lat").cast(pl.Float32),
        pl.col("Long").cast(pl.Float32),
        pl.col("ASN").cast(pl.Int32),
        pl.col("Country").cast(pl.Utf8),
        pl.col("TCP").cast(pl.Utf8),
        pl.col("ICMP").cast(pl.Utf8),
        pl.col("EventType").cast(pl.Utf8)
    ])
    output_file = output_dir / (f.stem + ".parquet")

    df.write_parquet(output_file, compression="zstd")
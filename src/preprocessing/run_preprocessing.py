"""Create cleaned sample or full source files for downstream teammates."""

from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.preprocessing.normalize import (
    ascii_fold,
    core_business_name,
    extract_emails,
    extract_house_number,
    extract_numbers,
    extract_phones,
    extract_postal_code,
    extract_region_hint,
    extract_urls,
    important_tokens,
    normalize_address,
    normalize_business_name,
    pipe_join,
    prefix_key,
    sorted_token_key,
    soundex_key,
    stable_hash,
    tokenize,
)

SOURCE_FILES = {
    "train_source1": "dataset/train/train_source1.tsv",
    "train_source2": "dataset/train/train_source2.tsv",
    "train_source3": "dataset/train/train_source3.tsv",
    "test_source1": "dataset/test/test_source1.tsv",
    "test_source2": "dataset/test/test_source2.tsv",
    "test_source3": "dataset/test/test_source3.tsv",
}


def resolve_input(input_path: Path) -> tuple[Path, TemporaryDirectory[str] | None]:
    """Return an extracted student_resource directory and optional temp owner."""
    input_path = input_path.expanduser().resolve()
    if input_path.is_dir():
        if input_path.name == "student_resource":
            return input_path, None
        nested = input_path / "student_resource"
        if nested.exists():
            return nested, None
        raise FileNotFoundError(f"Could not find student_resource under {input_path}")

    if input_path.suffix.lower() != ".zip":
        raise ValueError("--input must be a .zip file or an extracted student_resource folder")

    temp = TemporaryDirectory()
    with zipfile.ZipFile(input_path) as zf:
        names = [
            name
            for name in zf.namelist()
            if name.startswith("student_resource/dataset/")
            and not name.endswith("/")
            and "/._" not in name
        ]
        for name in names:
            zf.extract(name, temp.name)
    return Path(temp.name) / "student_resource", temp


def source_from_entity_id(entity_id: object) -> str:
    text = str(entity_id)
    return text.split("-", 1)[0] if "-" in text else ""


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add advanced cleaned columns required by downstream teams."""
    out = df.copy()
    for col in ["entity_id", "business_name", "business_address", "country"]:
        if col not in out.columns:
            raise ValueError(f"Missing required column: {col}")
        out[col] = out[col].fillna("").astype(str)

    out["source"] = out["entity_id"].map(source_from_entity_id)
    out["country_clean"] = out["country"].map(normalize_address)
    out["name_clean"] = out["business_name"].map(normalize_business_name)
    out["name_core"] = out["business_name"].map(core_business_name)
    out["name_ascii"] = out["name_core"].map(ascii_fold)
    out["address_clean"] = out["business_address"].map(normalize_address)
    out["address_ascii"] = out["address_clean"].map(ascii_fold)

    name_tokens = out["name_core"].map(tokenize)
    address_tokens = out["address_clean"].map(tokenize)
    important_name = name_tokens.map(important_tokens)
    important_address = address_tokens.map(important_tokens)
    numbers = out["business_address"].map(extract_numbers)
    combined_text = out["business_name"].str.cat(out["business_address"], sep=" ")

    out["name_tokens"] = name_tokens.map(pipe_join)
    out["address_tokens"] = address_tokens.map(pipe_join)
    out["important_name_tokens"] = important_name.map(pipe_join)
    out["important_address_tokens"] = important_address.map(pipe_join)
    out["numbers"] = numbers.map(pipe_join)
    out["house_number"] = out["business_address"].map(extract_house_number)
    out["postal_code"] = [
        extract_postal_code(address, country)
        for address, country in zip(out["business_address"], out["country"])
    ]
    out["region_hint"] = [
        extract_region_hint(address, country)
        for address, country in zip(out["business_address"], out["country"])
    ]
    out["emails"] = [
        pipe_join(extract_emails(name, address))
        for name, address in zip(out["business_name"], out["business_address"])
    ]
    out["phones"] = [
        pipe_join(extract_phones(name, address))
        for name, address in zip(out["business_name"], out["business_address"])
    ]
    out["urls"] = [
        pipe_join(extract_urls(name, address))
        for name, address in zip(out["business_name"], out["business_address"])
    ]

    out["name_sorted_key"] = name_tokens.map(sorted_token_key)
    out["address_sorted_key"] = address_tokens.map(sorted_token_key)
    out["name_prefix_key"] = important_name.map(prefix_key)
    out["address_prefix_key"] = important_address.map(prefix_key)
    out["name_soundex_key"] = important_name.map(soundex_key)
    out["postal_block_key"] = [
        pipe_join(part for part in [country, postal] if part)
        for country, postal in zip(out["country_clean"], out["postal_code"])
    ]
    out["name_postal_block_key"] = [
        pipe_join(part for part in [country, name_prefix, postal] if part)
        for country, name_prefix, postal in zip(
            out["country_clean"], out["name_prefix_key"], out["postal_code"]
        )
    ]
    out["house_postal_block_key"] = [
        pipe_join(part for part in [country, house, postal] if part)
        for country, house, postal in zip(
            out["country_clean"], out["house_number"], out["postal_code"]
        )
    ]
    out["record_fingerprint"] = [
        stable_hash(country, name, address)
        for country, name, address in zip(
            out["country_clean"], out["name_clean"], out["address_clean"]
        )
    ]

    out["name_len"] = out["name_clean"].str.len()
    out["address_len"] = out["address_clean"].str.len()
    out["combined_text_len"] = combined_text.str.len()
    out["name_token_count"] = name_tokens.map(len)
    out["address_token_count"] = address_tokens.map(len)
    out["number_count"] = numbers.map(len)
    out["has_name"] = out["name_clean"].ne("")
    out["has_address"] = out["address_clean"].ne("")
    out["has_postal_code"] = out["postal_code"].ne("")
    out["has_phone"] = out["phones"].ne("")
    out["has_email"] = out["emails"].ne("")
    out["has_url"] = out["urls"].ne("")
    out["is_low_information_name"] = (out["name_token_count"] <= 1) | (out["name_len"] <= 2)
    out["is_low_information_address"] = (out["address_token_count"] <= 2) | (out["address_len"] <= 8)

    return out[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country",
            "source",
            "country_clean",
            "name_clean",
            "name_core",
            "name_ascii",
            "address_clean",
            "address_ascii",
            "name_tokens",
            "address_tokens",
            "important_name_tokens",
            "important_address_tokens",
            "numbers",
            "house_number",
            "postal_code",
            "region_hint",
            "emails",
            "phones",
            "urls",
            "name_sorted_key",
            "address_sorted_key",
            "name_prefix_key",
            "address_prefix_key",
            "name_soundex_key",
            "postal_block_key",
            "name_postal_block_key",
            "house_postal_block_key",
            "record_fingerprint",
            "name_len",
            "address_len",
            "combined_text_len",
            "name_token_count",
            "address_token_count",
            "number_count",
            "has_name",
            "has_address",
            "has_postal_code",
            "has_phone",
            "has_email",
            "has_url",
            "is_low_information_name",
            "is_low_information_address",
        ]
    ]


def summarize_frame(df: pd.DataFrame, dataset_name: str) -> dict[str, object]:
    """Create compact quality stats for manifests."""
    bool_stats = {}
    for col in [
        "has_name",
        "has_address",
        "has_postal_code",
        "has_phone",
        "has_email",
        "has_url",
        "is_low_information_name",
        "is_low_information_address",
    ]:
        bool_stats[col] = int(df[col].sum())

    return {
        "dataset": dataset_name,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "source_counts": df["source"].value_counts(dropna=False).to_dict(),
        "country_counts": df["country_clean"].value_counts(dropna=False).head(25).to_dict(),
        "boolean_counts": bool_stats,
    }


def merge_manifest_stats(chunks: list[dict[str, object]]) -> dict[str, object]:
    """Merge per-chunk stats without loading full processed data again."""
    if not chunks:
        return {}

    merged = {
        "dataset": chunks[0]["dataset"],
        "rows": 0,
        "columns": chunks[0]["columns"],
        "source_counts": {},
        "country_counts": {},
        "boolean_counts": {},
    }

    for chunk in chunks:
        merged["rows"] += int(chunk["rows"])
        for group in ["source_counts", "country_counts", "boolean_counts"]:
            target = merged[group]
            for key, value in chunk[group].items():
                target[key] = target.get(key, 0) + int(value)

    merged["country_counts"] = dict(
        sorted(merged["country_counts"].items(), key=lambda item: item[1], reverse=True)[:25]
    )
    return merged


def write_run_manifest(
    output_dir: Path,
    mode: str,
    input_path: str,
    dataset_stats: list[dict[str, object]],
) -> None:
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "input": input_path,
        "datasets": dataset_stats,
    }
    path = output_dir / f"preprocessing_{mode}_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {path}")


def sample_size_for(dataset_name: str, args: argparse.Namespace) -> int:
    if dataset_name.endswith("source1"):
        return args.sample_size_s1
    return args.sample_size_other


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def preprocess_sample(student_dir: Path, output_dir: Path, args: argparse.Namespace) -> None:
    sample_dir = output_dir / "samples"
    stats = []
    for dataset_name, rel_path in SOURCE_FILES.items():
        path = student_dir / rel_path
        nrows = sample_size_for(dataset_name, args)
        df = pd.read_csv(path, sep="\t", dtype=str, nrows=nrows, keep_default_na=False)
        clean = clean_frame(df)
        write_parquet(clean, sample_dir / f"{dataset_name}_clean.parquet")
        stats.append(summarize_frame(clean, dataset_name))
        print(f"sample {dataset_name}: {len(clean):,} rows")
    write_run_manifest(output_dir, "sample", args.input, stats)


def preprocess_full(student_dir: Path, output_dir: Path, args: argparse.Namespace) -> None:
    processed_dir = output_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    stats = []
    for dataset_name, rel_path in SOURCE_FILES.items():
        source_path = student_dir / rel_path
        final_path = processed_dir / f"{dataset_name}_clean.parquet"
        writer = None
        total_rows = 0
        chunk_stats = []

        try:
            reader = pd.read_csv(
                source_path,
                sep="\t",
                dtype=str,
                chunksize=args.chunk_size,
                keep_default_na=False,
            )
            for chunk in reader:
                clean = clean_frame(chunk)
                chunk_stats.append(summarize_frame(clean, dataset_name))
                table = pa.Table.from_pandas(clean, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(final_path, table.schema, compression="snappy")
                writer.write_table(table)
                total_rows += len(clean)
                print(f"{dataset_name}: {total_rows:,} rows cleaned")
        finally:
            if writer is not None:
                writer.close()

        stats.append(merge_manifest_stats(chunk_stats))
        print(f"wrote {final_path}")
    write_run_manifest(output_dir, "full", args.input, stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess challenge source TSV files.")
    parser.add_argument("--input", required=True, help="Path to challenge ZIP or student_resource folder.")
    parser.add_argument("--output-dir", default="data", help="Directory containing samples/processed outputs.")
    parser.add_argument("--mode", choices=["sample", "full"], required=True)
    parser.add_argument("--sample-size-s1", type=int, default=50_000)
    parser.add_argument("--sample-size-other", type=int, default=200_000)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    student_dir, temp = resolve_input(Path(args.input))
    try:
        output_dir = Path(args.output_dir)
        if args.mode == "sample":
            preprocess_sample(student_dir, output_dir, args)
        else:
            preprocess_full(student_dir, output_dir, args)
    finally:
        if temp is not None:
            temp.cleanup()


if __name__ == "__main__":
    main()

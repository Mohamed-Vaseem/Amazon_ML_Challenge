"""Inspect row counts and schemas without loading whole files into memory."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

SOURCE_FILES = [
    "student_resource/dataset/train/train_source1.tsv",
    "student_resource/dataset/train/train_source2.tsv",
    "student_resource/dataset/train/train_source3.tsv",
    "student_resource/dataset/train/train_ground_truth.tsv",
    "student_resource/dataset/test/test_source1.tsv",
    "student_resource/dataset/test/test_source2.tsv",
    "student_resource/dataset/test/test_source3.tsv",
]


def inspect_zip(path: Path) -> None:
    with zipfile.ZipFile(path) as zf:
        for name in SOURCE_FILES:
            with zf.open(name) as handle:
                header = handle.readline().decode("utf-8-sig").rstrip("\r\n").split("\t")
                rows = sum(1 for _ in handle)
            print(f"{name}\trows={rows:,}\tcolumns={len(header)}\theader={header}")


def inspect_folder(path: Path) -> None:
    root = path if path.name == "student_resource" else path / "student_resource"
    for name in SOURCE_FILES:
        rel = name.removeprefix("student_resource/")
        file_path = root / rel
        with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = handle.readline().rstrip("\r\n").split("\t")
            rows = sum(1 for _ in handle)
        print(f"{file_path}\trows={rows:,}\tcolumns={len(header)}\theader={header}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect raw challenge dataset files.")
    parser.add_argument("--input", required=True, help="Path to challenge ZIP or extracted folder.")
    args = parser.parse_args()

    path = Path(args.input).expanduser().resolve()
    if path.suffix.lower() == ".zip":
        inspect_zip(path)
    else:
        inspect_folder(path)


if __name__ == "__main__":
    main()

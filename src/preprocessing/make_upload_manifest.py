"""Create a manifest for files uploaded to shared cloud storage."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*") if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build upload manifest with size and SHA256.")
    parser.add_argument("--input-dir", required=True, help="Directory containing files to upload.")
    parser.add_argument("--output", required=True, help="CSV manifest path to write.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in iter_files(input_dir):
        rows.append(
            {
                "relative_path": path.relative_to(input_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {output} with {len(rows)} files")


if __name__ == "__main__":
    main()

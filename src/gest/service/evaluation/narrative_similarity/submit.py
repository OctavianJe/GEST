from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create submission.zip with track_a.jsonl at root."
    )
    parser.add_argument(
        "--input", default="results/Narrative Similarity Task/submissions/track_a.jsonl"
    )
    parser.add_argument(
        "--output", default="results/Narrative Similarity Task/submissions/submission.zip"
    )
    args = parser.parse_args()

    zip_path = Path(args.output)
    file_path = Path(args.input)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(file_path, arcname="track_a.jsonl")
        names = zf.namelist()

    print("Created", zip_path, "with", names)


if __name__ == "__main__":
    main()

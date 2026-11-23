"""Download the raw datasets into data/raw/.

    python data/scripts/download.py            # everything
    python data/scripts/download.py --only scifact

- SciFact       : official release (corpus + claims) from the AllenAI bucket.
                  If the bucket is not reachable, a GitHub copy is used instead, and
                  every file is checked against the SHA-256 of the official release
                  (I checked that two independent copies give exactly these hashes and
                  the official sizes: 5,183 abstracts, 809 train / 300 dev / 300 test claims).
- Climate-FEVER : the jsonl file from the authors' GitHub repository.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path

SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
SCIFACT_MIRROR = "https://raw.githubusercontent.com/adrianfeijoo/search-bias-detection/main/data/raw/scifact/"
SCIFACT_SHA256 = {
    "corpus.jsonl": "b8d6c89624cb2ed74dee8938effc4f5d8bd2086887880af8110d64be4ceade62",
    "claims_train.jsonl": "f4c8fa82d8bd0653a9cc8d61a6ea48c25eacea64e90af5dbf390ebb1b74372f0",
    "claims_dev.jsonl": "86f0435d08fdb65d1aa41d1472684f57e6e71930626497bdf4d7a9ec1a632217",
    "claims_test.jsonl": "558930d75215c73f84a28fe538307d6d397c9de1ec7239514cf45f80d75d2ca3",
}
CLIMATE_URL = (
    "https://raw.githubusercontent.com/tdiggelm/climate-fever-dataset/main/dataset/climate-fever.jsonl"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> None:
    """Download url to dest (skips the download if the file is already there)."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"already there: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, dest)


def check_scifact(folder: Path) -> bool:
    """True if the 4 files are there and identical to the official release."""
    for name, expected in SCIFACT_SHA256.items():
        path = folder / name
        if not path.exists() or sha256(path) != expected:
            print(f"  {name}: missing or different from the official release")
            return False
    return True


def download_scifact(raw_dir: Path) -> None:
    out = raw_dir / "scifact"
    if out.exists() and check_scifact(out):
        print("SciFact already downloaded and verified")
        return
    out.mkdir(parents=True, exist_ok=True)
    try:
        archive = raw_dir / "scifact.tar.gz"
        fetch(SCIFACT_URL, archive)
        with tarfile.open(archive) as tar:
            # filter="data" refuses weird paths inside the archive (safer)
            tar.extractall(raw_dir / "_scifact_tmp", filter="data")
        # the archive has a data/ folder inside, move its files to data/raw/scifact/
        for f in (raw_dir / "_scifact_tmp" / "data").glob("*.jsonl"):
            f.replace(out / f.name)
        archive.unlink()
        shutil.rmtree(raw_dir / "_scifact_tmp", ignore_errors=True)
    except Exception as err:  # bucket down or blocked: use the verified copy
        print(f"official bucket failed ({type(err).__name__}), using the GitHub copy")
        for name in SCIFACT_SHA256:
            (out / name).unlink(missing_ok=True)
            fetch(SCIFACT_MIRROR + name, out / name)
    if not check_scifact(out):
        raise SystemExit("SciFact files do not match the official release, stop here.")
    print("SciFact verified:", sorted(p.name for p in out.glob("*.jsonl")))


def download_climate_fever(raw_dir: Path) -> None:
    fetch(CLIMATE_URL, raw_dir / "climate_fever" / "climate-fever.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--only", choices=["scifact", "climate_fever"], default=None)
    args = parser.parse_args()
    raw_dir = Path(args.raw_dir)
    if args.only in (None, "scifact"):
        download_scifact(raw_dir)
    if args.only in (None, "climate_fever"):
        download_climate_fever(raw_dir)


if __name__ == "__main__":
    main()

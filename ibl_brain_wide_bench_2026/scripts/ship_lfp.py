"""Split a merged lfpack BWM LFP archive and ship it to S3, one recording at a time.

Each archive version goes under its own prefix, ``<S3_PREFIX>/<version>/<pid>.h5`` (v03: the
lfpack mild tier; v05: the lfpack 1.2.0 fine tier, alpha=2.5).

Extracts a single recording (`lfpack.subset_h5`), uploads it, then deletes the local
copy before moving to the next one, so disk usage stays bounded to the source archive
plus one temporary file regardless of how many recordings are shipped.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import time
from pathlib import Path

import h5py
from lfpack import subset_h5
from one.api import ONE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

S3_PREFIX = "s3://brain-wide-bench/brainsets/lfp/ibl_brain_wide_bench_2026"
AWS_PROFILE = "ibl"


def bwm_eids(pipeline_dir: Path) -> set[str]:
    """Union of pretrain and eval session ids that define the BWM cohort.

    Parameters
    ----------
    pipeline_dir : Path
        Directory containing ``data/pretrain_eids.txt`` and ``data/eval_eids.txt``.

    Returns:
    -------
    set of str
        All session ids in either file.
    """
    eids: set[str] = set()
    for name in ("pretrain_eids.txt", "eval_eids.txt"):
        eids.update(pipeline_dir.joinpath("data", name).read_text().split())
    return eids


def resolve_pid_eid(one: ONE, pid: str, retries: int = 2) -> str | None:
    """Resolve a probe insertion id to its session id via Alyx.

    Retries a couple of times first, since a transient Alyx query hiccup is far more
    likely than a genuinely stale insertion.

    Returns:
    -------
    str or None
        The session id, or None if `pid` cannot be resolved after retrying.
    """
    for attempt in range(retries + 1):
        try:
            eid, _ = one.pid2eid(pid)
            return str(eid)
        except Exception as exc:
            if attempt == retries:
                logger.warning(
                    f"could not resolve pid {pid} to an eid after {retries + 1} attempts: {exc}"
                )
    return None


def shipped_pids(version: str) -> set[str]:
    """Pids already present under the version prefix, so an interrupted run can resume."""
    result = subprocess.run(
        ["aws", "s3", "ls", "--profile", AWS_PROFILE, f"{S3_PREFIX}/{version}/"],
        capture_output=True,
        text=True,
    )
    # a missing prefix makes `aws s3 ls` exit 1 with no output: nothing shipped yet
    return {
        line.split()[-1].removesuffix(".h5")
        for line in result.stdout.splitlines()
        if line.endswith(".h5")
    }


def upload_and_remove(local_path: Path, s3_key: str, retries: int = 3) -> None:
    """Upload a file to S3 with the ibl profile, retrying transient failures, then delete it."""
    for attempt in range(retries + 1):
        try:
            subprocess.run(
                [
                    "aws",
                    "s3",
                    "cp",
                    "--only-show-errors",
                    "--profile",
                    AWS_PROFILE,
                    str(local_path),
                    s3_key,
                ],
                check=True,
            )
            break
        except subprocess.CalledProcessError:
            if attempt == retries:
                raise
            logger.warning(
                f"upload of {local_path.name} failed, retrying ({attempt + 1}/{retries})"
            )
            time.sleep(10 * (attempt + 1))
    local_path.unlink()


def main(
    archive: Path,
    version: str,
    pipeline_dir: Path,
    tmp_dir: Path,
    dry_run: bool,
    limit: int | None,
) -> None:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    wanted_eids = bwm_eids(pipeline_dir)

    with h5py.File(archive, "r") as f:
        pids = list(f.keys())
    logger.info(
        f"{len(pids)} recordings in {archive.name}, {len(wanted_eids)} eids in the BWM cohort"
    )

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        username="intbrainlab",
        password="international",
    )

    done = shipped_pids(version)
    logger.info(f"{len(done)} recordings already under {S3_PREFIX}/{version}/, skipping them")

    shipped, skipped = 0, 0
    for pid in pids:
        if limit is not None and shipped >= limit:
            break
        if pid in done:
            continue

        eid = resolve_pid_eid(one, pid)
        if eid is None or eid not in wanted_eids:
            skipped += 1
            continue

        dst = tmp_dir.joinpath(f"{pid}.h5")
        subset_h5(archive, dst, [pid])
        s3_key = f"{S3_PREFIX}/{version}/{pid}.h5"

        if dry_run:
            logger.info(
                f"[dry-run] would upload {dst} ({dst.stat().st_size / 1e6:.1f} MB) -> {s3_key}"
            )
            dst.unlink()
        else:
            upload_and_remove(dst, s3_key)
            logger.info(f"shipped {pid} (eid={eid}) -> {s3_key}")
        shipped += 1

    logger.info(f"done: shipped={shipped} skipped={skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path.home().joinpath(
            "Documents",
            "datadisk",
            "lfp-processing",
            "lfpack",
            "v05_bwm",
            "lf_compressed_v05_a2p5_fine_all_bwm.h5",
        ),
    )
    parser.add_argument("--version", default="v05", help="S3 sub-prefix, e.g. v05")
    parser.add_argument("--pipeline-dir", type=Path, default=Path(__file__).parent.parent)
    parser.add_argument(
        "--tmp-dir", type=Path, default=None, help="defaults to a folder next to --archive"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--limit", type=int, default=None, help="ship at most N recordings, for testing"
    )
    args = parser.parse_args()

    tmp_dir = args.tmp_dir or args.archive.parent.joinpath("_ship_lfp_tmp")
    main(args.archive, args.version, args.pipeline_dir, tmp_dir, args.dry_run, args.limit)

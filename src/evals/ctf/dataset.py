"""
Dataset handling for CTF tasks.
"""

from pathlib import Path
from typing import Optional

from inspect_ai.dataset import Dataset, Sample, json_dataset


def read_dataset(shuffle: bool = True, limit: Optional[int] = None) -> Dataset:
    """Read the CTF dataset.

    Args:
        shuffle: Whether to shuffle the dataset
        limit: Optional limit on number of samples to return

    Returns:
        A Dataset instance containing CTF tasks
    """
    data_dir = Path.cwd() / "data" / "ctf"
    dataset = json_dataset(
        json_file=data_dir / "tasks.json",
        sample_fields=_record_to_sample,
        shuffle=shuffle,
    )

    if limit:
        dataset = dataset.limit(limit)

    return dataset


def _record_to_sample(record: dict) -> Sample:
    """Convert a dataset record to a Sample.

    Args:
        record: The dataset record to convert

    Returns:
        A Sample instance
    """
    return Sample(
        id=record["id"],
        input=record["query"],
        target=record["flag"],
        metadata={
            "source": record.get("source", "unknown"),
            "tags": record.get("tags", []),
            "solution": record.get("solution", ""),
        },
        files=record.get("files", {}),
        setup=record.get("setup", None),
    )

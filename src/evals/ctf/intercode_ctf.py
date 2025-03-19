"""
CTF benchmark implementation for the CyberAgentBreeder project.
"""

from pathlib import Path
from textwrap import dedent
from typing import Any, Literal, Union

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import includes

from evals.benchmark import Benchmark, register_benchmark
from .dataset import read_dataset

COMPOSE_FILE = Path.cwd() / "compose.yaml"
DEFAULT_TOOL_CONFIGS = {
    "bash": {"timeout": 180},
    "python": {"timeout": 180},
}


@register_benchmark("intercode_ctf")
class IntercodeCTFBenchmark(Benchmark):
    """CTF benchmark implementation."""

    def __init__(
        self,
        args=None,
        split: Union[Literal["validation"], Literal["test"]] = "validation",
        shuffle: bool = True,
        limit: int = 100,
    ) -> Dataset:
        """Initialize the CTF benchmark.

        Args:
            args: Arguments object containing configurations
            split: Dataset split to use
            shuffle: Whether to shuffle the dataset
            limit: Maximum number of samples to use
        """
        self.split = split
        self.args = args
        self.dataset = read_dataset(shuffle=shuffle, limit=limit)

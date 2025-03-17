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
from .solvers import CTFSolver, DEFAULT_TOOL_CONFIGS

COMPOSE_FILE = Path.cwd() / "compose.yaml"


@register_benchmark("ctf")
class CTFBenchmark(Benchmark):
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

    @task
    def match_task(self):
        """Create the CTF task."""
        return Task(
            time_limit=self.args.task_timeout if self.args else 180,
            name=self.__class__.__name__,
            dataset=self.dataset,
            solver=CTFSolver(),
            scorer=includes(),
            config=GenerateConfig(temperature=0.5),
            sandbox=("docker", COMPOSE_FILE.as_posix()),
        )

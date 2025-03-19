from pathlib import Path
from typing import Any, Literal, Union

from inspect_ai.dataset import Dataset

from ..benchmark import Benchmark, register_benchmark
from .dataset import read_dataset
from .docker import generate_dockerfile, DEFAULT_APT_GET_INSTALLS, DEFAULT_PIP3_INSTALLS

COMPOSE_FILE = Path.cwd() / "src/evals/ctf/compose.yaml"

print(COMPOSE_FILE)
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
        generate_dockerfile(DEFAULT_APT_GET_INSTALLS, DEFAULT_PIP3_INSTALLS)
        self.dataset = read_dataset(shuffle=shuffle)
        self.sandbox = ("docker", COMPOSE_FILE.as_posix())

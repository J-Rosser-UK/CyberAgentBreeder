from pathlib import Path
from typing import Any, Literal, Union

from inspect_ai.dataset import Dataset
from inspect_ai import Task
from inspect_ai.scorer import includes, accuracy, scorer, Scorer
from inspect_ai.scorer._common import str_match_scorer
from ..benchmark import Benchmark, register_benchmark
from .dataset import read_dataset
from .docker import generate_dockerfile, DEFAULT_APT_GET_INSTALLS, DEFAULT_PIP3_INSTALLS
from .tools import DEFAULT_TOOL_CONFIGS
from ..metrics import ci_lower, ci_upper, median

COMPOSE_FILE = Path.cwd() / "src/evals/intercode_ctf/compose.yaml"


@register_benchmark("intercode_ctf")
class IntercodeCTFBenchmark(Benchmark):
    """CTF benchmark implementation."""

    def __init__(
        self,
        args=None,
        shuffle: bool = True,
    ) -> Dataset:
        """Initialize the CTF benchmark.

        Args:
            args: Arguments object containing configurations
            split: Dataset split to use
            shuffle: Whether to shuffle the dataset
            limit: Maximum number of samples to use
        """

        self.args = args
        self.shuffle = shuffle
        generate_dockerfile(DEFAULT_APT_GET_INSTALLS, DEFAULT_PIP3_INSTALLS)

    def tasks(self, solvers) -> list[Task]:

        tasks = []
        for solver in solvers:
            try:
                tasks.append(
                    Task(
                        dataset=read_dataset(shuffle=self.shuffle),
                        name=solver[0],
                        solver=solver[1](DEFAULT_TOOL_CONFIGS),
                        scorer=self.includes(),
                        sandbox=("docker", COMPOSE_FILE.as_posix()),
                    )
                )
            except Exception as e:
                print(f"Error creating task for {solver[0]}: {e}")
                continue
        return tasks

    @staticmethod
    @scorer(metrics=[accuracy(), ci_lower(), ci_upper(), median()])
    def includes(ignore_case: bool = True) -> Scorer:
        """Check whether the specified text is included in the model output.

        Args:
        ignore_case: Use a case insensitive comparison.

        """

        def check(value: str, target: str) -> tuple[str, bool]:
            if ignore_case:
                value = value.casefold()
                target = target.casefold()
            return value, target in value

        return str_match_scorer(check)

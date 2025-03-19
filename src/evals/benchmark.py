from inspect_ai import Task, task
from inspect_ai.dataset import Sample, Dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import solver, Solver, TaskState, Generate
from inspect_ai._eval.eval import eval
from inspect_ai.scorer import Score, scorer, accuracy, includes

from inspect_ai._util.appdirs import inspect_cache_dir
from inspect_ai._util.error import pip_dependency_error
from inspect_ai._util.file import safe_filename
from inspect_ai._util.hash import mm3_hash
from inspect_ai._util.version import verify_required_version

from inspect_ai.dataset._dataset import (
    Dataset,
    FieldSpec,
    MemoryDataset,
    RecordToSample,
)
from inspect_ai.dataset._util import data_to_samples, record_to_sample_fn

from pathlib import Path
import hashlib
from abc import ABC, abstractmethod
import os
import importlib.util
import uuid
import json
from typing import Any, Union
import re
import time
import random
from .model import CustomModel, CustomModelAPI
from .metrics import ci_lower, ci_upper, median

from ctf.dataset import read_dataset


benchmark_registry = {}
COMPOSE_FILE = Path.cwd() / "compose.yaml"


def register_benchmark(name):
    """
    Decorator that registers a benchmark class in the global benchmark_registry.

    Args:
        name (str): The key to use for the registry.
    """

    def decorator(cls):
        benchmark_registry[name] = cls
        return cls

    return decorator


class AgentScaffoldException(Exception):
    """Custom exception for errors in the agent scaffold."""

    pass


class Benchmark(ABC):

    def evaluate(self, scaffolds, limit=10, log_d="logs"):

        temp_files = []
        solvers = []
        for scaffold in scaffolds:
            solver_callable, temp_file = Benchmark.get_callable(
                scaffold.scaffold_id, scaffold.scaffold_name, scaffold.scaffold_code
            )
            solvers.append([scaffold.scaffold_name, solver_callable])
            temp_files.append(temp_file)

        tasks = [
            Task(
                dataset=read_dataset(
                    # shuffle=shuffle,
                ),
                name=solver[0],
                solver=solver[1],
                scorer=includes(),
                sandbox=("docker", COMPOSE_FILE.as_posix()),
            )
            for solver in solvers
        ]

        results = eval(
            tasks,
            model=self.args.model,
            limit=limit,
            log_dir=f"./src/{log_d}/{self.split}/{self.args.log_timestamp}/{self.__class__.__name__}-{str(scaffolds[0].population_id)}/logs",  # specify where logs are stored
            log_format="json",  # choose log format ("eval" or "json")
            score=True,  # ensure scoring is enable
            max_tasks=500,
        )

        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except Exception as e:
                print("Error removing temp file:", e)

        # 'results' is a list of EvalLog objects (usually one per task)
        # Each EvalLog contains metrics for the entire task/dataset.
        model_metrics = {}  # dictionary to hold info for each model

        for res in results:

            # 1) Get the model name and task name
            model_name = str(getattr(res.eval, "model", ""))
            task_name = res.eval.task

            # 2) Initialize defaults (or None) for each metric
            accuracy = None
            ci_lower = None
            ci_upper = None
            median = None

            # 3) Check if results and scores exist
            if res.results and res.results.scores:
                for score in res.results.scores:
                    if score.metrics:
                        # 4) For each metric, check if it exists and store its value
                        if "accuracy" in score.metrics:
                            accuracy = score.metrics["accuracy"].value
                        if "ci_lower" in score.metrics:
                            ci_lower = score.metrics["ci_lower"].value
                        if "ci_upper" in score.metrics:
                            ci_upper = score.metrics["ci_upper"].value
                        if "median" in score.metrics:
                            median = score.metrics["median"].value

            # 5) Save the metrics in a dictionary, keyed by the model name
            if not model_metrics.get(model_name):
                model_metrics[model_name] = {task_name: {}}

            if not model_metrics[model_name].get(task_name):
                model_metrics[model_name][task_name] = {}

            model_metrics[model_name][task_name] = {
                "accuracy": accuracy,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "median": median,
            }

        return model_metrics

    @abstractmethod
    def _record_to_sample(self, record: dict[str, Any]) -> Sample:
        pass

    @staticmethod
    def get_callable(scaffold_id, scaffold_name, scaffold_code) -> tuple:

        try:
            forward_function = scaffold_code
            # Create the agent scaffold in temporary code
            current_directory = os.path.dirname(os.path.abspath(__file__))
            parent_directory = os.path.dirname(current_directory)
            cleaned_name = re.sub(r"[^A-Za-z0-9 ]+", "", scaffold_name)
            temp_file = (
                f"""{parent_directory}/temp/agent_scaffold_temp_"""
                + f"""
                {cleaned_name}_{scaffold_id}_{uuid.uuid4()}.py""".strip()
            )

            # Write the complete solver script to the file, including the forward function
            with open(temp_file, "w") as f:
                f.write(forward_function)

            # Import the solver_callable class from the temp file
            spec = importlib.util.spec_from_file_location(
                "agent_scaffold_temp", temp_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find any function decorated with @solver
            solver_callable = None
            for name, obj in module.__dict__.items():
                if hasattr(obj, "__solver__") and callable(obj):
                    solver_callable = obj
                    break

            if solver_callable is None:
                raise ValueError(
                    "No function decorated with @solver found in the module"
                )

        except Exception as e:
            print("Error during benchmark evaluation:", e)

            # get traceback
            import traceback

            print(traceback.format_exc())

            return None, temp_file

        return solver_callable, temp_file

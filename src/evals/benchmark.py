from abc import ABC, abstractmethod
from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import includes
from inspect_ai._eval.eval import eval
from inspect_ai.tool import Tool


import os
from typing import Any

from pathlib import Path


benchmark_registry = {}


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
            solver_callable, temp_file = Benchmark.get_callable(scaffold.scaffold_code)
            solvers.append([scaffold.scaffold_name, solver_callable])
            temp_files.append(temp_file)

        tasks = [
            Task(
                dataset=self.dataset,
                name=solver[0],
                solver=solver[1],
                scorer=includes(),
                sandbox=self.sandbox,
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

    @staticmethod
    def get_callable(scaffold_code: str) -> tuple[callable, str]:
        """
        Convert the scaffold code into a callable function.

        Args:
            scaffold_code (str): The scaffold code to convert.

        Returns:
            tuple: A tuple containing the callable function and the module name.
        """
        import importlib.util
        import sys
        from types import ModuleType

        module_name = "scaffold_module"
        spec = importlib.util.spec_from_loader(module_name, loader=None)
        scaffold_module = importlib.util.module_from_spec(spec)
        exec(scaffold_code, scaffold_module.__dict__)
        sys.modules[module_name] = scaffold_module

        # Find the function decorated with @solver
        solver_function = None
        for attr_name in dir(scaffold_module):
            attr = getattr(scaffold_module, attr_name)

            if callable(attr) and attr.__name__ == "solver_wrapper":
                print(attr)
                solver_function = attr
                break

        if solver_function is None:
            raise ValueError(
                "No function decorated with @solver found in the scaffold code."
            )

        return solver_function, module_name

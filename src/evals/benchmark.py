from abc import ABC

from inspect_ai import Task
from inspect_ai.scorer import includes
from inspect_ai._eval.eval import eval

import os
import ast
import importlib.util
import inspect
import sys
from typing import Any, Callable, Dict, List, Tuple, Union
from types import ModuleType

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

    def evaluate(self, scaffolds):

        solvers = []

        for scaffold in scaffolds:
            try:
                name = scaffold.scaffold_name
                solver = self.extract_solver_functions(str(scaffold.scaffold_code))
                solvers.append((name, solver))
            except Exception as e:
                print(
                    f"Warning: Error extracting solver functions for {scaffold.scaffold_name}:"
                )
                import traceback

                traceback.print_exc()

        results = eval(
            self.tasks(solvers),
            model=self.args.model,
            limit=self.args.n_evals,
            token_limit=self.args.token_limit,
            log_dir=f"./src/logs/{self.args.log_timestamp}/{self.__class__.__name__}-{str(scaffolds[0].population_id)}/logs",  # specify where logs are stored
            log_format="json",  # choose log format ("eval" or "json")
            score=True,  # ensure scoring is enable
            max_tasks=500,
        )

        # 'results' is a list of EvalLog objects (usually one per task)
        # Each EvalLog contains metrics for the entire task/dataset.
        model_metrics = {}  # dictionary to hold info for each model

        for res in results:

            # 1) Get the model name and task name
            model_name = str(getattr(res.eval, "model", ""))
            task_name = res.eval.task

            print(model_name, task_name)

            # 2) Initialize defaults (or None) for each metric
            accuracy, ci_lower, ci_upper, median = None, None, None, None

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
    def extract_solver_functions(code_string, module_name="dynamic_module"):
        """
        Extract callable solver functions from a Python code string.

        Args:
            code_string (str): A string containing Python code
            module_name (str): Name to give the dynamically created module

        Returns:
            list: A list of callable solver functions that are decorated with @solver
        """
        # Create a temporary module to execute the code
        module = ModuleType(module_name)
        sys.modules[module_name] = module

        # Execute the code string in the context of the module
        exec(code_string, module.__dict__)

        # Parse the code string to find solver-decorated function names
        tree = ast.parse(code_string)
        solver_function_names = []

        # Find all solver-decorated functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.decorator_list:
                    for decorator in node.decorator_list:
                        if (
                            isinstance(decorator, ast.Name) and decorator.id == "solver"
                        ) or (
                            isinstance(decorator, ast.Attribute)
                            and decorator.attr == "solver"
                        ):
                            solver_function_names.append(node.name)

        # Get the actual function objects from the module
        solver_functions = []
        for name in solver_function_names:
            if hasattr(module, name):
                solver_functions.append(getattr(module, name))

        output = solver_functions[-1]

        assert isinstance(output, Callable)

        return output

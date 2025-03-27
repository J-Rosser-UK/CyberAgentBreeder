from base.tables import Scaffold
from .benchmark import benchmark_registry


class Validator:

    def __init__(self, args, split="validation"):
        """
        Initializes the Validator class.

        Args:
            args: Arguments object containing configurations for the evaluator, including
            dataset file paths and model settings.
        """
        self.args = args
        self.benchmarks = benchmark_registry
        self.benchmark = self.benchmarks[args.benchmark](
            args=self.args,
            shuffle=False,
        )

    def validate(self, scaffolds_for_validation: list[Scaffold]):
        if len(scaffolds_for_validation) == 0:
            return
        model_metrics = self.benchmark.evaluate(scaffolds_for_validation)

        print(model_metrics)

        for scaffold in scaffolds_for_validation:
            scaffold.update(
                scaffold_capability_ci_median=0.0,
                scaffold_capability_ci_sample_size=self.args.n_evals,
                scaffold_capability_ci_lower=0.0,
                scaffold_capability_ci_upper=0.0,
                scaffold_capability_ci_confidence_level=0.95,
            )

            for model, task_metrics in model_metrics.items():
                for task, metrics in task_metrics.items():

                    print(f"Model: {model}")
                    print(f"Task: {task}")
                    print(f"  accuracy: {metrics['accuracy']}")
                    print(f"  ci_lower: {metrics['ci_lower']}")
                    print(f"  ci_upper: {metrics['ci_upper']}")
                    print(f"  median:   {metrics['median']}")

                    if task == scaffold.scaffold_name:
                        if metrics["median"] is not None:
                            scaffold.update(
                                scaffold_capability_ci_median=metrics["median"],
                                scaffold_capability_ci_lower=metrics["ci_lower"],
                                scaffold_capability_ci_upper=metrics["ci_upper"],
                            )

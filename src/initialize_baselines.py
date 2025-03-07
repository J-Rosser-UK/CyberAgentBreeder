import argparse
from base import initialize_session, Population, Scaffold, initialize_population_id
from evals import Validator
import os
import warnings
from sqlalchemy.exc import SAWarning
from discover.seed_scaffolds import COT
import logging
import datetime
import time

# Disable logging for httpx
logging.getLogger("httpx").disabled = True

# Suppress all SAWarnings
warnings.filterwarnings("ignore", category=SAWarning)


def initialize_population_id(args) -> str:
    """
    Initializes the first generation of scaffolds for a given population.

    Args:
        args: Arguments object containing configurations for the population initialization.

    Returns:
        str: The unique ID of the initialized population.
    """
    for session in initialize_session():

        archive = [COT]

        population = Population(session=session, population_benchmark=args.benchmark)

        validator = Validator(args)

        generation_timestamp = datetime.datetime.utcnow()
        scaffolds_for_validation = []
        for scaffold in archive:
            scaffold = Scaffold(
                session=session,
                scaffold_name=scaffold["name"],
                scaffold_code=scaffold["code"],
                scaffold_thought_process=scaffold["thought"],
                population=population,
                generation_timestamp=generation_timestamp,
            )
            scaffolds_for_validation.append(scaffold)

        validator.validate(scaffolds_for_validation, log_d="baselines")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    current_directory = os.path.dirname(os.path.abspath(__file__))
    log_timestamp_str = str(time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--current_dir", type=str, default=current_directory)
    parser.add_argument("--log_timestamp", type=str, default=log_timestamp_str)
    parser.add_argument("--random_seed", type=int, default=40)
    parser.add_argument("--n_generation", type=int, default=10)
    parser.add_argument("--n_mutations", type=int, default=10)
    parser.add_argument("--n_evals", type=int, default=1000)
    parser.add_argument("--debug_max", type=int, default=3)
    parser.add_argument("--mode", type=str, default="ablation")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("-p", "--population_id", type=str, default="None")
    parser.add_argument("--benchmark", type=str, default="mmlu")
    parser.add_argument("--task_timeout", type=int, default=60 * 60)

    args = parser.parse_args()

    initialize_population_id(args)

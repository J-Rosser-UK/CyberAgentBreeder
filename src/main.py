import argparse
import random
import logging
import os
import time
import warnings
import asyncio
from tqdm import tqdm
from sqlalchemy.exc import SAWarning
from discover import Discover
from discover import Discover
from descriptor import Clusterer
from base import initialize_session, Scaffold, initialize_population_id
from evals import Validator

# Disable logging for httpx
logging.getLogger("httpx").disabled = True

# Suppress all SAWarnings
warnings.filterwarnings("ignore", category=SAWarning)


def main(args):
    random.seed(args.random_seed)

    # Initialize args.population_id only if it doesn't exist
    if not args.population_id:
        args.population_id = initialize_population_id(args)
        print(f"Population ID: {args.population_id}")

    for session in initialize_session():
        validator = Validator(args)
        clusterer = Clusterer(args)

        scaffolds = (
            session.query(Scaffold).filter_by(population_id=args.population_id).all()
        )

        # Recluster the population
        clusterer.cluster(scaffolds)

        # Only choose scaffolds which haven't been validated yet
        scaffolds_for_validation = (
            session.query(Scaffold)
            .filter_by(
                population_id=args.population_id, scaffold_capability_ci_median=None
            )
            .order_by(Scaffold.scaffold_timestamp.desc())
            .all()[:10]
        )
        validator.validate(scaffolds_for_validation)

        print(f"Reloaded population ID: {args.population_id}")

        # Begin Bayesian Illumination...
        for _ in tqdm(range(args.n_generation), desc="Generations"):
            discoverer = Discover(args)

            discoverer.discover(session)

            print("Generation complete")

            scaffolds = (
                session.query(Scaffold)
                .filter_by(population_id=args.population_id)
                .all()
            )

            print([scaffold.scaffold_name for scaffold in scaffolds])

            # Recluster the population
            clusterer.cluster(scaffolds)

            print("Clustering complete")

            # Only choose scaffolds which haven't been validated yet
            scaffolds_for_validation = (
                session.query(Scaffold)
                .filter_by(
                    population_id=args.population_id, scaffold_capability_ci_median=None
                )
                .all()
            )
            print("Scaffolds for validation:")
            print([scaffold.scaffold_name for scaffold in scaffolds_for_validation])

            validator.validate(scaffolds_for_validation)

            print("Validation complete")
            session.commit()

            print("Generation complete")

    return args.population_id  # Return the population ID for restarts


if __name__ == "__main__":
    # logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser()
    current_directory = os.path.dirname(os.path.abspath(__file__))
    log_timestamp_str = time.strftime("%Y%m%d-%H%M%S")
    parser.add_argument("--current_dir", type=str, default=current_directory)
    parser.add_argument("--log_timestamp", type=str, default=log_timestamp_str)
    parser.add_argument("--random_seed", type=int, default=40)
    parser.add_argument("--n_generation", type=int, default=10)    # number of generations (there are inbuilt restarts so no probs there if we want to keep running longer)
    parser.add_argument("--n_mutations", type=int, default=20)    # number of new scaffolds created each generation (there is no debugging so i would expect lots to fail, this should give us 10 successes!
    parser.add_argument("--n_evals", type=int, default=100)    # number of inspect samples for evals
    parser.add_argument("--token_limit", type=int, default=250_000)
    parser.add_argument("--debug_max", type=int, default=3)    # doesn't do anything, here for when i reimplement debugging
    parser.add_argument("--scaffold_model", type=str, default="openai/gpt-4o-mini")
    parser.add_argument(
        "--meta_agent_model", type=str, default="anthropic/claude-3-7-sonnet-20250219"
    )
    parser.add_argument("-p", "--population_id", type=str, default="None")    # set this if you want to pick backup where you left off
    parser.add_argument("--benchmark", type=str, default="intercode_ctf")	# replace with "cybench" for those runs
    parser.add_argument("--task_timeout", type=int, default=30 * 60)

    # For k8s
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--max_tasks", type=int, default=100)
    parser.add_argument("--max_subprocesses", type=int, default=100)
    parser.add_argument("--max_sandboxes", type=int, default=100)

    args = parser.parse_args()

    if args.population_id == "None":
        args.population_id = None

    if args.population_id == "last":
        for session in initialize_session():
            scaffold = (
                session.query(Scaffold)
                .filter(Scaffold.population_benchmark == args.benchmark)
                .order_by(Scaffold.population_timestamp.desc())
                .limit(1)
                .one()
            )
            args.population_id = scaffold.population_id
            args.benchmark = scaffold.population_benchmark

    args.population_id = main(args)

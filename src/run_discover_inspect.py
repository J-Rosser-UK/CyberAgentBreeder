#!/usr/bin/env python3
"""
Script to run the Discover Inspect module for framework discovery.
"""

import argparse
import asyncio
import logging
from pathlib import Path
import datetime
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from discover.discover import Discover
from src.base import create_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            f"logs/discover_inspect_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Discover Inspect to discover new frameworks"
    )

    parser.add_argument(
        "--population_id",
        type=str,
        default="default",
        help="Population ID to use for the discovery process",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use for insights and scaffold generation",
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        default="intercode_ctf",
        help="Benchmark to use for the generated scaffolds",
    )

    parser.add_argument(
        "--n_mutations", type=int, default=3, help="Number of scaffolds to generate"
    )

    parser.add_argument(
        "--code_samples_file",
        type=str,
        default=None,
        help="Path to file containing code samples to analyze (optional)",
    )

    return parser.parse_args()


async def main():
    """Run the discover inspect process."""
    args = parse_args()
    logger.info(f"Starting Discover Inspect with args: {args}")

    # Create database session
    session = create_session()

    # Initialize Discover Inspect
    discover = Discover(args)

    # Load code samples if specified
    code_samples = None
    if args.code_samples_file:
        try:
            with open(args.code_samples_file, "r") as f:
                code_samples = f.read()
            logger.info(f"Loaded code samples from {args.code_samples_file}")
        except Exception as e:
            logger.error(f"Error loading code samples: {e}")

    # Run generation
    try:
        logger.info("Running generation...")
        results = await discover.run_generation(session, code_samples)
        logger.info(f"Generation complete. Generated {len(results)} scaffolds.")

        # Print scaffold names
        for i, scaffold in enumerate(results):
            logger.info(f"Scaffold {i+1}: {scaffold.scaffold_name}")

    except Exception as e:
        logger.error(f"Error during generation: {e}")

    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())

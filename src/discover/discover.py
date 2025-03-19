"""
Discover module for the CyberAgentBreeder project.
"""

import random
import asyncio
import logging
import datetime
from tqdm import tqdm

from base import Scaffold
from descriptor import Descriptor
from evals import Validator
from evals.intercode_ctf.intercode_ctf import IntercodeCTFBenchmark

from discover.mutation_prompts import multi_agent_scaffold_mutation_prompts
from .evolve import Evolve, load_prompt_with_examples
from base import elites


class Discover:
    """Main discovery class for evolving CTF agents."""

    def __init__(self, args) -> None:
        """Initialize the Discover class.

        Args:
            args: Arguments object containing configurations
            population: Population instance to operate on

        """
        self.args = args

        self.population_id = self.args.population_id

        self.mutation_operators = multi_agent_scaffold_mutation_prompts

        self.batch_size = 1
        self.descriptor = Descriptor()

        self.validator = Validator(args)
        self.base_prompt = None
        self.base_prompt_response_format = None

        # Initialize CTF benchmark
        self.ctf_benchmark = IntercodeCTFBenchmark(args=args)

    async def generate_offspring(
        self,
        parents,
        session,
    ):
        """Generate offspring from parents.

        Args:
            parents: List of parent scaffolds
            session: Database session

        Returns:
            Generated offspring scaffold
        """
        try:
            evolver = Evolve(
                self.args,
                self.mutation_operators,
                self.validator,
                self.base_prompt,
                self.base_prompt_response_format,
                session,
            )
            offspring_scaffold = await evolver.evolve(parents)

        except Exception as e:
            logging.error(f"Error generating offspring: {e}")
            offspring_scaffold = None

        return offspring_scaffold

    async def run_generation(self, session):
        """Run a generation of evolution.

        Args:
            session: Database session

        Returns:
            List of generated offspring
        """
        parents = []

        for _ in range(self.args.n_mutations):
            scaffold_1 = random.choice(elites(session, self.population_id)).to_dict()
            scaffold_2 = random.choice(elites(session, self.population_id)).to_dict()

            parents.append((scaffold_1, scaffold_2))

        # Load the prompt with examples from database
        self.base_prompt = load_prompt_with_examples(session)
        self.base_prompt_response_format = {
            "thought": "Your explanation of the design choices, structure, and any important considerations for the scaffold.",
            "name": "The snake-case name of the scaffold. E.g. react_and_plan",
            "code": "The complete Python script for the new scaffold.",
        }

        generation_timestamp = datetime.datetime.utcnow()

        # Create tasks for all mutations
        tasks = [
            asyncio.create_task(self.generate_offspring(parents[i], session))
            for i in range(self.args.n_mutations)
        ]

        results = []

        # Use tqdm + asyncio.as_completed to update the bar after each task finishes
        with tqdm(total=len(tasks), desc="Mutations in progress") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update(1)

        for scaffold in results:
            if scaffold and scaffold["scaffold_code"]:
                scaffold = Scaffold(
                    session=session,
                    scaffold_name=scaffold["scaffold_name"],
                    scaffold_code=scaffold["scaffold_code"],
                    scaffold_reasoning=scaffold["scaffold_reasoning"],
                    scaffold_first_parent_id=scaffold["scaffold_first_parent_id"],
                    scaffold_second_parent_id=scaffold["scaffold_second_parent_id"],
                    population=self.population_id,
                    generation_timestamp=generation_timestamp,
                    scaffold_benchmark=self.args.benchmark,
                )
                self.population_id.scaffolds.append(scaffold)

                scaffold.update(scaffold_descriptor=self.descriptor.generate(scaffold))

        return results

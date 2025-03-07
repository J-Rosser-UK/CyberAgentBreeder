import random
from base import Scaffold
from descriptor import Descriptor
from evals import Validator
from discover.mutation_prompts import (
    multi_agent_scaffold_mutation_prompts,
    multi_agent_scaffold_safety_mutation_prompts,
)

from .evolve import Evolve
import asyncio
import logging
import datetime
from tqdm import tqdm
from discover.utils import get_base_prompt_with_archive


class Discover:

    def __init__(self, args, population, debug_sample) -> None:
        """
        Initializes the Discover class.

        Args:
            args: Arguments object containing configurations for the generator.

            population_id: The ID of the population to operate on.
        """
        self.args = args
        self.debug_sample = debug_sample
        self.population = population

        self.mutation_operators = multi_agent_scaffold_mutation_prompts
        if self.args.mode in ["blue"]:
            self.mutation_operators = (
                multi_agent_scaffold_mutation_prompts
                + multi_agent_scaffold_safety_mutation_prompts
            )
        self.batch_size = 1
        self.descriptor = Descriptor()

        self.validator = Validator(args)
        self.base_prompt = None
        self.base_prompt_response_format = None

    async def generate_offspring(
        self,
        parents,
    ):

        try:
            evolver = Evolve(
                self.args,
                self.mutation_operators,
                self.validator,
                self.base_prompt,
                self.base_prompt_response_format,
                self.debug_sample,
            )
            # Create a new Discover instance per task
            offspring_scaffold = await evolver.evolve(parents)

        except Exception as e:
            logging.error(f"Error generating offspring: {e}")
            offspring_scaffold = None

        return offspring_scaffold

    async def run_generation(self, session):

        parents = []

        for _ in range(self.args.n_mutations):

            if self.args.mode in ["blue", "red"]:

                scaffold_1 = random.choice(self.population.pareto_elites).to_dict()
                scaffold_2 = random.choice(self.population.pareto_elites).to_dict()

            elif self.args.mode in ["ablation"]:

                scaffold_1 = random.choice(self.population.elites).to_dict()
                scaffold_2 = random.choice(self.population.elites).to_dict()

            parents.append((scaffold_1, scaffold_2))

        self.base_prompt, self.base_prompt_response_format = (
            get_base_prompt_with_archive(self.args, session)
        )

        generation_timestamp = datetime.datetime.utcnow()

        # Create tasks for all mutations
        tasks = [
            asyncio.create_task(self.generate_offspring(parents[i]))
            for i in range(self.args.n_mutations)
        ]

        results = []

        # Use tqdm + asyncio.as_completed to update the bar after each task finishes
        with tqdm(total=len(tasks), desc="Mutations in progress") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update(1)  # Update the bar once a task finishes
        # print(results)

        for scaffold in results:
            if scaffold and scaffold["scaffold_code"]:

                scaffold = Scaffold(
                    session=session,
                    scaffold_name=scaffold["scaffold_name"],
                    scaffold_code=scaffold["scaffold_code"],
                    scaffold_thought_process=scaffold["scaffold_thought_process"],
                    scaffold_first_parent_id=scaffold["scaffold_first_parent_id"],
                    scaffold_second_parent_id=scaffold["scaffold_second_parent_id"],
                    population=self.population,
                    generation_timestamp=generation_timestamp,
                )
                self.population.scaffolds.append(scaffold)

                scaffold.update(scaffold_descriptor=self.descriptor.generate(scaffold))

        return results

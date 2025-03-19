import random
import os
import uuid
from evals import AgentScaffoldException
import logging
import json
import re
import asyncio
from evals import Benchmark
from pathlib import Path
import glob
from base import elites
import openai

client = openai.OpenAI()


def load_prompt_with_examples(session):
    """Load the prompt.md file and replace example scaffolds with actual examples from database."""
    prompt_path = Path(__file__).parent / "prompt.md"

    with open(prompt_path, "r") as f:
        prompt_content = f.read()

    # Get elite scaffolds from the database
    elite_scaffolds = elites(
        session, None
    )  # None for population_id means get all elites

    # Format each example scaffold
    example_scaffolds = []
    for scaffold in elite_scaffolds:
        if scaffold.scaffold_reasoning:  # Only include scaffolds with thought process
            example_scaffolds.append(
                f"<{scaffold.scaffold_name}>\n{scaffold.scaffold_reasoning}\n</{scaffold.scaffold_name}>"
            )

    # Replace the placeholder with actual examples
    example_section = "\n\n".join(example_scaffolds)
    prompt_content = prompt_content.replace("{{EXAMPLE_SCAFFOLDS}}", example_section)

    return prompt_content


class Evolve:

    def __init__(
        self,
        args,
        mutation_operators,
        evaluator,
        base_prompt,
        base_prompt_response_format,
        debug_sample,
        session,
    ) -> None:
        """
        Initializes the Evolve class.

        Args:
            args: Arguments object containing configurations for the mutator, such
            as debugging limits and model settings.

            population: The population of scaffolds for mutation.
            mutation_operators: A list of mutation operator strings to apply.
            evaluator: An evaluator object for validating and testing evolved scaffolds.
            session: Database session for loading example scaffolds.
        """

        self.mutation_operators = mutation_operators
        self.args = args
        self.evaluator = evaluator
        self.session = session

        # Load the prompt with examples if base_prompt is None
        self.base_prompt = base_prompt or load_prompt_with_examples(session)
        self.base_prompt_response_format = base_prompt_response_format

        self.debug_sample = debug_sample

    async def evolve(self, parents: list[dict]) -> dict:
        """
        Applies a mutation to a given scaffold.

        Args:
            scaffold (Scaffold): The scaffold object to mutate.

        Returns:
            Scaffold: The mutated scaffold object. Returns None if the mutation fails.
        """

        mutated_scaffold = None
        scaffold_response = {"code": None}
        i = 0
        while (not mutated_scaffold or not scaffold_response.get("code")) and i < 3:
            i += 1
            try:

                mutation_operator = random.choice([self._mutate, self._crossover])

                (
                    scaffold_name,
                    scaffold_code,
                    scaffold_reasoning,
                    sampled_mutation,
                ) = await mutation_operator(parents)

                mutated_scaffold = {
                    "scaffold_name": scaffold_name,
                    "scaffold_code": scaffold_code,
                    "scaffold_first_parent_id": str(parents[0]["scaffold_id"]),
                    "scaffold_second_parent_id": (
                        str(parents[1]["scaffold_id"])
                        if mutation_operator == self._crossover
                        else None
                    ),
                    "scaffold_reasoning": scaffold_reasoning,
                    "scaffold_mutation_prompt": (
                        sampled_mutation if sampled_mutation else ""
                    ),
                }
            except Exception as e:

                print(f"Error evolving scaffold: {e}")
                mutated_scaffold = None

        return mutated_scaffold

    async def _mutate(self, parents: list[dict]):
        """
        Applies a sampled mutation to a scaffold and refines it using reflexion-based prompts.

        Args:
            None

        Returns:
            tuple: A tuple containing the next_response (dict), the updated messages (list),
                and the reflexion_response_format (str).
        """
        scaffold = parents[0]
        logging.info(f"Mutating {scaffold.get('scaffold_name')} scaffold...")
        print(f"Mutating {scaffold.get('scaffold_name')} scaffold...")

        sampled_mutation = random.choice(self.mutation_operators)

        messages = [
            {
                "role": "user",
                "content": """You are a helpful assistant. Make sure to return in a WELL-FORMED JSON object.""",
            },
            {
                "role": "user",
                "content": f"""
                {self.base_prompt}
             
                Here is the multi-agent scaffold I would like you to mutate:

                ---------------
                Scaffold: {scaffold.get('scaffold_name')}
                {scaffold.get("scaffold_reasoning")}
                ---------------
                {scaffold.get("scaffold_code")}

                The mutation I would like to apply is:
                {sampled_mutation}

                
                IMPORTANT:
                In general, the new scaffold will perform better with more detailed prompts for the agents, more planning steps,
                encouringing them to think longer and harder. It may be worth adding a final agent to the scaffold to help
                transform the output of the final agent into the desired output format for the task as the scaffold will be scored
                very lowley if the output is not in the correct format, even if the thinking was sound.

                CRITICAL REQUIREMENT:
                The scaffold code MUST include the @solver decorator on the main function. This is required for the scaffold to work.
                The function signature should be: @solver def scaffold_name(...parameters...): -> Solver
                Inside this function, you must define an async solve(state: TaskState, generate: Generate) -> TaskState function.

                Ensure that the new forward functions outputs a response as a
                STRING in the exact format as specified in the required_answer_format. This could be
                either a single letter (e.g. A, B, C, D) or a word or phrase, or a 
                short piece of code.
                
                """.strip(),
            },
        ]

        return await self._evolve(messages, sampled_mutation)

    async def _crossover(self, parents: list[dict]):
        """
        Applies crossover to two scaffolds and refines the result using reflexion-based prompts.

        Args:
            None

        Returns:
            tuple: A tuple containing the next_response (dict), the updated messages (list),
                and the reflexion_response_format (str).
        """
        scaffold_1 = parents[0]
        scaffold_2 = parents[1]
        logging.info(
            f"Crossing over {scaffold_1.get('scaffold_name')} and {scaffold_2.get('scaffold_name')} scaffolds..."
        )
        print(
            f"Crossing over {scaffold_1.get('scaffold_name')} and {scaffold_2.get('scaffold_name')} scaffolds..."
        )

        messages = [
            {
                "role": "user",
                "content": """You are a helpful assistant. Make sure to return in a WELL-FORMED JSON object.""",
            },
            {
                "role": "user",
                "content": f"""
                {self.base_prompt}
             
                Here are the two scaffolds I'd like you to crossover/combine into a novel new scaffold:

                ---------------
                Scaffold 1: {scaffold_1.get('scaffold_name')}
                {scaffold_1.get("scaffold_reasoning")}
                ---------------
                {scaffold_1.get("scaffold_code")}

                ---------------
                Scaffold 2: {scaffold_2.get('scaffold_name')}
                {scaffold_2.get("scaffold_reasoning")}
                ---------------
                {scaffold_2.get("scaffold_code")}   

                Ensure that the new forward functions outputs a response as a
                STRING in the exact format as specified in the required_answer_format. This could be
                either a single letter (e.g. A, B, C, D) or a word or phrase, or a 
                short piece of code.

                CRITICAL REQUIREMENT:
                The scaffold code MUST include the @solver decorator on the main function. This is required for the scaffold to work.
                The function signature should be: @solver def scaffold_name(...parameters...): -> Solver
                Inside this function, you must define an async solve(state: TaskState, generate: Generate) -> TaskState function.
                """.strip(),
            },
        ]

        return await self._evolve(messages, None)

    async def _evolve(self, messages, sampled_mutation):

        # Generate new solution and do reflection
        try:

            output = await client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
            )

            response = output.choices[0].message.content

            # Parse the response to extract scaffold_name, scaffold_code, and scaffold_reasoning
            scaffold_name = (
                re.search(r"<name>(.*?)</name>", response, re.DOTALL).group(1).strip()
            )
            # Clean up the scaffold to only allow numbers, letters, hyphens and underscores
            scaffold_name = re.sub(r"[^A-Za-z0-9 \-\u2013\u2014]+", "", scaffold_name)

            scaffold_code = (
                re.search(r"<code>(.*?)</code>", response, re.DOTALL).group(1).strip()
            )
            scaffold_reasoning = (
                re.search(r"<reasoning>(.*?)</reasoning>", response, re.DOTALL)
                .group(1)
                .strip()
            )

            # Validate that the scaffold code includes the @solver decorator
            if "@solver" not in scaffold_code:
                print("Error: Generated scaffold code missing @solver decorator")
                return None

        except Exception as e:
            print("During LLM generate new solution:")
            print(e)

            return None

        return scaffold_name, scaffold_code, scaffold_reasoning, sampled_mutation

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
from evals.intercode_ctf.dataset import read_dataset

from evals.intercode_ctf.tools import DEFAULT_TOOL_CONFIGS

from inspect_ai.model import (
    CachePolicy,
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentText,
    call_tools,
    get_model,
)
from inspect_ai.scorer import ValueToFloat, score, value_to_float, includes
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, ToolCall, ToolResult, bash, tool, python
from inspect_ai import Task


class DiscoverInspect:
    """Main discovery class for evolving CTF agents."""

    def __init__(self, args) -> None:
        """Initialize the Discover class.

        Args:
            args: Arguments object containing configurations
            population: Population instance to operate on

        """
        self.args = args

        self.mutation_operators = multi_agent_scaffold_mutation_prompts

        self.batch_size = 1
        self.descriptor = Descriptor()

        self.validator = Validator(args)

        # Initialize CTF benchmark
        self.ctf_benchmark = IntercodeCTFBenchmark(args=args)

    async def discover(self, session):

        self.base_prompt = load_prompt_with_examples(self.args, session)

        parents = []

        for _ in range(self.args.n_mutations):
            scaffold_1 = random.choice(
                elites(session, self.args.population_id)
            ).to_dict()
            scaffold_2 = random.choice(
                [e for e in elites(session, self.args.population_id) if e != scaffold_1]
            ).to_dict()

            mutation_prompt = random.choice(
                ["crossover", random.choice(self.mutation_operators)]
            )

            parents.append((scaffold_1, scaffold_2, mutation_prompt))

        generation_timestamp = datetime.datetime.utcnow()

        results = eval(
            self.tasks(parents),
            model="anthropic/claude-3-5-sonnet-20240620",
            limit=self.args.n_evals,
            token_limit=self.args.token_limit,
            log_dir=f"./src/logs/{self.args.log_timestamp}/discover/{self.__class__.__name__}-{str(self.args.population_id)}/logs",  # specify where logs are stored
            log_format="json",  # choose log format ("eval" or "json")
            score=True,  # ensure scoring is enable
            max_tasks=500,
        )

        for scaffold in results:
            if scaffold and scaffold["scaffold_code"]:
                scaffold = Scaffold(
                    session=session,
                    scaffold_name=scaffold["scaffold_name"],
                    scaffold_code=scaffold["scaffold_code"],
                    scaffold_reasoning=scaffold["scaffold_reasoning"],
                    scaffold_first_parent_id=scaffold["scaffold_first_parent_id"],
                    scaffold_second_parent_id=scaffold["scaffold_second_parent_id"],
                    population_id=self.args.population_id,
                    generation_timestamp=generation_timestamp,
                    scaffold_benchmark=self.args.benchmark,
                )

                scaffold.update(scaffold_descriptor=self.descriptor.generate(scaffold))

        return results

    def tasks(self, parents):
        """Generate tasks for the given parents.

        Args:
            parents: List of parent scaffolds
        """
        solvers = []
        for parent_1, parent_2, mutation_prompt in parents:
            solvers.append(
                self.discover(
                    DEFAULT_TOOL_CONFIGS,
                    (parent_1, parent_2, mutation_prompt),
                    self.base_prompt,
                )
            )

        return [
            Task(
                dataset=read_dataset(shuffle=self.shuffle),
                name=f"discover-{i}",
                solver=solver,
                scorer=includes(),
            )
            for i, solver in enumerate(solvers)
        ]

    @staticmethod
    @solver
    def discover(tools, parents, base_prompt):

        scaffold_1, scaffold_2, mutation_prompt = parents

        print(scaffold_1.scaffold_name, scaffold_2.scaffold_name, mutation_prompt)

        if mutation_prompt == "crossover":

            messages = [
                {
                    "role": "user",
                    "content": f"""
                        {base_prompt}
                    
                        Here are the two scaffolds I'd like you to crossover/combine into a novel new scaffold:

                        <scaffold_for_crossover_1>
                            <name>{scaffold_1.get('scaffold_name')}</name>
                            <reasoning>{scaffold_1.get("scaffold_reasoning")}</reasoning>
                            <code>{scaffold_1.get("scaffold_code")}</code>
                        </scaffold_for_crossover_1>

                        <scaffold_for_crossover_2>
                            <name>{scaffold_2.get('scaffold_name')}</name>
                            <reasoning>{scaffold_2.get("scaffold_reasoning")}</reasoning>
                            <code>{scaffold_2.get("scaffold_code")}</code>
                        </scaffold_for_crossover_2>

                        
                        """.strip(),
                },
            ]
        else:

            messages = [
                {
                    "role": "user",
                    "content": f"""
                {base_prompt}
             
                Here is the multi-agent scaffold I would like you to mutate:

                <scaffold_for_mutation>
                <name>{scaffold_1.get('scaffold_name')}</name>
                <reasoning>{scaffold_1.get("scaffold_reasoning")}</reasoning>
                <code>{scaffold_1.get("scaffold_code")}</code>
                </scaffold_for_mutation>

               
                The mutation I would like to apply is:
                <mutation_operator>
                {mutation_prompt}
                </mutation_operator>

                
                """.strip(),
                },
            ]

        async def solve(state: TaskState, generate: Generate) -> TaskState:

            state.messages = []

            state.messages.append(ChatMessageUser(content=messages[0]["content"]))

            model = get_model()

            output = await model.generate(input=state.messages, cache=False)

            state.messages.append(ChatMessageAssistant(content=output.message.content))

            state.output.completion = output.message.content

            print("COMPLETION", state.output.completion)

            return state

        return solve

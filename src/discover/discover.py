"""
Discover module for the CyberAgentBreeder project.
"""

import random
import asyncio
import logging
import datetime
from tqdm import tqdm
import re
from pathlib import Path
from textwrap import dedent
from base import Scaffold
from descriptor import Descriptor
from evals import Validator
from evals.intercode_ctf.intercode_ctf import IntercodeCTFBenchmark
from evals.benchmark import Benchmark
from discover.mutation_operators import multi_agent_scaffold_mutation_operators

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
from inspect_ai import eval


def load_prompt_with_examples(args, session):
    """Load the prompt.md file and replace example scaffolds with actual examples from database."""
    prompt_path = Path(__file__).parent / "prompt.md"

    with open(prompt_path, "r") as f:
        prompt_content = f.read()

    # Get elite scaffolds from the database
    elite_scaffolds = elites(session, args.population_id)[0:5][
        ::-1
    ]  # Only use the top 5 elites and reverse them so that the highest median is last

    # Format each example scaffold
    example_scaffolds = []
    for i, scaffold in enumerate(elite_scaffolds):
        if scaffold.scaffold_reasoning:  # Only include scaffolds with reasoning
            example_scaffolds.append(
                f"<example_scaffold_{i+1}>\n"
                + "<reasoning>"
                + f"{scaffold.scaffold_reasoning}"
                + "</reasoning>\n"
                + "<name>"
                + f"{scaffold.scaffold_name}"
                + "</name>\n"
                + "<code>"
                + f"{scaffold.scaffold_code}"
                + "</code>\n"
                + "</example_scaffold_{i+1}>"
            )

    # Replace the placeholder with actual examples
    example_section = "\n\n".join(example_scaffolds)
    prompt_content = prompt_content.replace("{{EXAMPLE_SCAFFOLDS}}", example_section)

    # write to file
    # with open("prompt_with_examples.md", "w") as f:
    #     f.write(prompt_content)

    return prompt_content


class Discover:
    """Main discovery class for evolving CTF agents."""

    def __init__(self, args) -> None:
        """Initialize the Discover class.

        Args:
            args: Arguments object containing configurations
            population: Population instance to operate on

        """
        self.args = args

        self.mutation_operators = multi_agent_scaffold_mutation_operators

        self.batch_size = 1
        self.descriptor = Descriptor()

        self.validator = Validator(args)

        # Initialize CTF benchmark
        self.ctf_benchmark = IntercodeCTFBenchmark(args=args)

    def discover(self, session):

        self.base_prompt = load_prompt_with_examples(self.args, session)

        parents = []

        for _ in range(self.args.n_mutations):
            mutation_operator = random.choice(
                ["crossover", random.choice(self.mutation_operators)]
            )

            scaffold_1 = random.choice(
                elites(session, self.args.population_id)
            ).to_dict()

            if mutation_operator == "crossover":
                scaffold_2 = random.choice(
                    [
                        e
                        for e in elites(session, self.args.population_id)
                        if str(e.scaffold_id) != str(scaffold_1.get("scaffold_id"))
                    ]
                ).to_dict()
            else:
                scaffold_2 = None

            parents.append((scaffold_1, scaffold_2, mutation_operator))

        generation_timestamp = datetime.datetime.utcnow()

        results = eval(
            self.tasks(parents),
            model=self.args.meta_agent_model,
            limit=1,
            log_dir=f"./src/logs/{self.args.log_timestamp}/discover/{self.__class__.__name__}-{str(self.args.population_id)}/logs",  # specify where logs are stored
            log_format="json",  # choose log format ("eval" or "json")
            score=False,  # ensure scoring is enable
            max_samples=self.args.max_samples,
            max_tasks=self.args.max_tasks,
            max_subprocesses=self.args.max_subprocesses,
            max_sandboxes=self.args.max_sandboxes,
            max_connections=self.args.max_anthropic_connections,
            max_tokens=self.args.max_tokens,
        )

        for result in results:

            sample = result.samples[0]
            response = sample.output.completion
            metadata = sample.metadata

            # Generate new solution and do reflection
            try:

                # Parse the response to extract scaffold_name, scaffold_code, and scaffold_reasoning
                scaffold_name = (
                    re.search(r"<name>(.*?)</name>", response, re.DOTALL)
                    .group(1)
                    .strip()
                )
                # Clean up the scaffold to only allow numbers, letters, hyphens and underscores
                scaffold_name = re.sub(
                    r"[^A-Za-z0-9_\-\u2013\u2014]+", "", scaffold_name
                )

                scaffold_code = (
                    re.search(r"<code>(.*?)</code>", response, re.DOTALL)
                    .group(1)
                    .strip()
                )
                if scaffold_code is None:
                    continue

                scaffold_reasoning = (
                    re.search(r"<reasoning>(.*?)</reasoning>", response, re.DOTALL)
                    .group(1)
                    .strip()
                )

                # Validate that the scaffold code includes the @solver decorator
                if "@solver" not in scaffold_code:
                    raise Exception(
                        "Error: Generated scaffold code missing @solver decorator"
                    )

                scaffold = Scaffold(
                    session=session,
                    scaffold_name=scaffold_name,
                    scaffold_code=scaffold_code,
                    scaffold_reasoning=scaffold_reasoning,
                    scaffold_first_parent_id=metadata["parent_1"],
                    scaffold_second_parent_id=metadata["parent_2"],
                    scaffold_mutation_operator=metadata["mutation_operator"],
                    population_id=self.args.population_id,
                    generation_timestamp=generation_timestamp,
                    scaffold_benchmark=self.args.benchmark,
                )

                scaffold.update(scaffold_descriptor=self.descriptor.generate(scaffold))

            except Exception as e:
                print(f"During LLM generate new solution: {e}")

                import traceback

                traceback.print_exc()

        return results

    def tasks(self, parents):
        """Generate tasks for the given parents.

        Args:
            parents: List of parent scaffolds
        """
        solvers = []
        for parent_1, parent_2, mutation_operator in parents:
            solvers.append(
                self.discover_solver(
                    DEFAULT_TOOL_CONFIGS,
                    (parent_1, parent_2, mutation_operator),
                    self.base_prompt,
                )
            )

        return [
            Task(
                dataset=read_dataset(shuffle=False),
                name=f"discover-{i}",
                solver=solver,
                scorer=includes(),
                time_limit=self.args.time_limit,
            )
            for i, solver in enumerate(solvers)
        ]

    @staticmethod
    @solver
    def discover_solver(tools, parents, base_prompt):

        scaffold_1, scaffold_2, mutation_operator = parents

        print(
            scaffold_1.get("scaffold_name"),
            scaffold_2.get("scaffold_name") if scaffold_2 else None,
            mutation_operator,
        )

        if mutation_operator == "crossover":

            messages = [
                {
                    "role": "user",
                    "content": f"""
                        {base_prompt}
                    
                        Here are the two scaffolds I'd like you to crossover/combine into a novel new scaffold:

                        <scaffold_for_crossover_1>
                            <name>{scaffold_1.get('scaffold_name')}</name>
                            <reasoning>{scaffold_1.get('scaffold_reasoning')}</reasoning>
                            <code>{scaffold_1.get('scaffold_code')}</code>
                        </scaffold_for_crossover_1>

                        <scaffold_for_crossover_2>
                            <name>{scaffold_2.get('scaffold_name')}</name>
                            <reasoning>{scaffold_2.get('scaffold_reasoning')}</reasoning>
                            <code>{scaffold_2.get('scaffold_code')}</code>
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
                <reasoning>{scaffold_1.get('scaffold_reasoning')}</reasoning>
                <code>{scaffold_1.get('scaffold_code')}</code>
                </scaffold_for_mutation>

               
                The mutation I would like to apply is:
                <mutation_operator>
                {mutation_operator}
                </mutation_operator>

                
                """.strip(),
                },
            ]

        max_continues = 3

        async def solve(state: TaskState, generate: Generate) -> TaskState:

            state.metadata["parent_1"] = scaffold_1.get("scaffold_id")
            state.metadata["parent_2"] = (
                scaffold_2.get("scaffold_id") if scaffold_2 else None
            )
            state.metadata["mutation_operator"] = mutation_operator

            state.messages = []

            state.messages.append(ChatMessageUser(content=messages[0]["content"]))

            model = get_model()
            m = 0
            complete_output: str = ""
            for m in range(max_continues):
                print("CONTINUE", m)

                output = await model.generate(input=state.messages, cache=False)

                if isinstance(output.message.content, list):
                    output_message_content = str(output.message.content[0].text)
                else:
                    output_message_content = str(output.message.content)

                print(
                    "OUTPUT",
                    output_message_content[:100],
                    "...",
                    output_message_content[-100:],
                )

                complete_output += output_message_content

                state.messages.append(output.message)

                if all(
                    tag in complete_output
                    for tag in ["</code>", "</reasoning>", "</name>"]
                ):
                    print("COMPLETE", m)
                    break

                state.messages.append(
                    ChatMessageUser(
                        content=dedent(
                            f"""
Please continue the response where it left off.

If the partial response ended in the middle of a section (e.g., <reasoning>, <name>, or <code>), continue that section and include the closing tag (e.g., </reasoning>, </name>, or </code>). If a section was completed (indicated by the closing tag), move on to the next logical section based on the original instructions.

Output your continuation without any preamble or explanation. Begin writing as if you were the original author, picking up mid-sentence, mid-paragraph, or mid-line of code if necessary.

It is crucial that your continuation flows seamlessly from the partial response, allowing the two parts to be joined together without any visible break or repetition. It is also crucial that all sections are wrapped in the appropriate tags. Begin your continuation now:"""
                        ).strip()
                    )
                )

            for _ in range(3):

                try:
                    scaffold_code = (
                        re.search(r"<code>(.*?)</code>", complete_output, re.DOTALL)
                        .group(1)
                        .strip()
                    )
                    solver_fn = Benchmark.extract_solver_functions(scaffold_code)
                    if solver_fn:
                        output = solver_fn(state, generate)
                        break
                except Exception as e:
                    import traceback

                    complete_traceback = traceback.format_exc()

                    if complete_traceback:
                        state.messages.append(
                            ChatMessageUser(
                                content=dedent(
                                    f"""
        The scaffold code has errors. Please fix the following issues and provide a new version between <code> tags:

        {complete_traceback}

        Ensure your response maintains the same overall structure but fixes these errors."""
                                ).strip()
                            )
                        )
                        # Reset complete_output to only keep non-code sections
                        complete_output = re.sub(
                            r"<code>.*?</code>", "", complete_output, flags=re.DOTALL
                        )

            state.output.completion = complete_output

            print("COMPLETION", state.output.completion)

            return state

        return solve

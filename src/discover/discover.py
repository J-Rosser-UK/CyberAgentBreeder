"""
Discover module for the CyberAgentBreeder project.
"""

import random
import asyncio
import logging
import datetime
from tqdm import tqdm
import re
import uuid
import os
from pathlib import Path
from textwrap import dedent
from base import Scaffold
from descriptor import Descriptor
from evals import Validator
from evals.intercode_ctf.intercode_ctf import IntercodeCTFBenchmark
from evals.benchmark import Benchmark
from discover.mutation_operators import multi_agent_scaffold_mutation_operators
from typing import Callable
from base import elites
from evals.intercode_ctf.dataset import read_dataset
from evals.intercode_ctf.intercode_ctf import COMPOSE_FILE

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
            model=self.args.scaffold_model,
            limit=1,
            log_dir=f"./src/logs/{self.args.log_timestamp}/discover/{self.__class__.__name__}-{str(self.args.population_id)}/logs",  # specify where logs are stored
            log_format="json",  # choose log format ("eval" or "json")
            score=False,  # ensure scoring is enable
            log_buffer=1,
            fail_on_error=False,
            max_samples=self.args.max_samples,
            max_tasks=self.args.max_tasks,
            max_subprocesses=self.args.max_subprocesses,
            max_sandboxes=self.args.max_sandboxes,
            max_connections=(
                self.args.max_anthropic_connections
                if self.args.scaffold_model.startswith("anthropic")
                else self.args.max_openai_connections
            ),
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
                sandbox=("docker", COMPOSE_FILE.as_posix()),
            )
            for i, solver in enumerate(solvers)
        ]

    @solver
    def discover_solver(self, tools, parents, base_prompt):

        scaffold_1, scaffold_2, mutation_operator = parents

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

            initial_state_messages = state.messages.copy()

            meta_agent_state_messages = []

            meta_agent_state_messages.append(
                ChatMessageUser(content=messages[0]["content"])
            )

            model = get_model(self.args.meta_agent_model)
            m = 0
            complete_output: str = ""
            for m in range(max_continues):
                print(f"CONTINUE {m}")

                output = await model.generate(
                    input=meta_agent_state_messages, cache=False
                )

                if isinstance(output.message.content, list):
                    output_message_content = str(output.message.content[0].text)
                else:
                    output_message_content = str(output.message.content)

                complete_output += output_message_content

                print(str(output.message)[:100])

                meta_agent_state_messages.append(output.message)

                if all(
                    tag in complete_output
                    for tag in ["</code>", "</reasoning>", "</name>"]
                ):
                    print("All tags found!")
                    break

                meta_agent_state_messages.append(
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

            reasoning = (
                re.search(r"<reasoning>(.*?)</reasoning>", complete_output, re.DOTALL)
                .group(1)
                .strip()
            )

            scaffold_name = (
                re.search(r"<name>(.*?)</name>", complete_output, re.DOTALL)
                .group(1)
                .strip()
            )

            scaffold_code = (
                re.search(r"<code>(.*?)</code>", complete_output, re.DOTALL)
                .group(1)
                .strip()
            )

            for d in range(self.args.debug_max):
                print(f"debugging {d}")

                try:
                    state.messages = initial_state_messages
                    current_directory = os.path.dirname(os.path.abspath(__file__))
                    parent_directory = os.path.dirname(current_directory)
                    cleaned_name = re.sub(r"[^A-Za-z0-9 ]+", "", scaffold_name)
                    temp_file = (
                        f"""{parent_directory}/temp/agent_system_temp_"""
                        + f"""
                        {cleaned_name}_{uuid.uuid4()}.py""".strip()
                    )
                    solver_fn = Benchmark.extract_solver_functions(
                        scaffold_code, temp_file
                    )
                    if solver_fn:
                        solver_solver = solver_fn(tools)
                        assert isinstance(solver_solver, Callable)
                        try:
                            output = await asyncio.wait_for(
                                solver_solver(state, generate), timeout=60
                            )
                        except asyncio.TimeoutError:
                            meta_agent_state_messages.append(
                                ChatMessageUser(content=dedent("""Scaffold runs!"""))
                            )

                            print("Code ran! Timeout error reached :)")
                            break

                    else:
                        raise Exception("No solver function found in scaffold code")
                except Exception as e:
                    import traceback

                    traceback.print_exc()

                    complete_traceback = traceback.format_exc()

                    if complete_traceback:
                        meta_agent_state_messages.append(
                            ChatMessageUser(
                                content=dedent(
                                    f"""
                                    You are an expert programmer tasked with identifying and fixing bugs in scaffold code. You will be provided with the original scaffold code and a traceback of the error(s). Your goal is to analyze the code, identify the bug(s), and provide a corrected version of the code.

                                    Here is the scaffold code:

                                    <scaffold_code>
                                    {scaffold_code}
                                    </scaffold_code>

                                    Here is the traceback of the error(s):

                                    <traceback>
                                    {complete_traceback}
                                    </traceback>

                                    Please follow these steps to complete the task:

                                    1. Analyze the scaffold code and the traceback carefully.
                                    2. Identify the bug(s) present in the code.
                                    3. Fix the identified bug(s) and create a corrected version of the code.
                                    4. Present your findings and the corrected code in the specified format.

                                    Before providing your final output, wrap your debugging process in <debug_process> tags. In this section:

                                    1. Quote the specific lines from the traceback that indicate where the error occurred.
                                    2. Explain what each quoted line means in plain English.
                                    3. List out the variables and functions involved in the error.
                                    4. Describe how these elements interact to cause the bug.
                                    5. Outline your plan for fixing the bug, step by step.

                                    In your final output, provide two sections:

                                    1. A <bug_identified> section where you briefly describe the bug(s) you found.
                                    2. A <code> section containing the full corrected code. Do not use any markdown formatting (such as ``` or ```python) within this section. The code should start immediately after the opening <code> tag.

                                    Remember to maintain the overall structure of the original code while fixing the errors.

                                    Here's an example of the expected output format:

                                    <debug_process>
                                    [Your detailed analysis of the code and traceback, following the steps outlined above]
                                    </debug_process>

                                    <bug_identified>
                                    [Brief description of the identified bug(s)]
                                    </bug_identified>

                                    <code>
                                    # Full corrected code without any markdown formatting
                                    # This code will be run verbatim, so ensure it is syntactically correct
                                    # Do not include any markdown formatting (such as ``` or ```python) within this section. The code should start immediately after the opening <code> tag.
                                    # Do not include any ... or leave any bits of the code out.
                                    </code>

                                    Please proceed with your analysis and provide the corrected code.
                                    """
                                ).strip()
                            )
                        )
                        debug_output = await model.generate(
                            input=meta_agent_state_messages, cache=False
                        )

                        meta_agent_state_messages.append(debug_output.message)

                        try:
                            scaffold_code = (
                                re.search(
                                    r"<code>(.*?)</code>",
                                    debug_output.message.content,
                                    re.DOTALL,
                                )
                                .group(1)
                                .strip()
                            )
                        except Exception as e:
                            print("Error extracting scaffold code", e)
                            continue
                finally:
                    os.remove(temp_file)

            state.output.completion = f"""
            <name>{scaffold_name}</name>
            <reasoning>{reasoning}</reasoning>
            <code>{scaffold_code}</code>
            """

            # print("COMPLETION", state.output.completion)

            state.metadata["meta_agent_state_messages"] = meta_agent_state_messages

            return state

        return solve

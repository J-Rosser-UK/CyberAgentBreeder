from inspect_ai import Task, task
from inspect_ai.scorer import includes
from inspect_ai._eval.eval import eval

from seed.react_and_plan_agent import react_and_plan_agent
from seed.cyber_recon_scaffold import cyber_recon_scaffold
from seed.meta_gpt_scaffold import meta_gpt_scaffold
from seed.agent_verse_scaffold import agent_verse_scaffold
from seed.dylan_scaffold import dylan_scaffold


from evals import (
    read_dataset,
    COMPOSE_FILE,
    DEFAULT_TOOL_CONFIGS,
    generate_dockerfile,
)


generate_dockerfile()

solvers = [
    # react_and_plan_agent(DEFAULT_TOOL_CONFIGS),
    # cyber_recon_scaffold(DEFAULT_TOOL_CONFIGS),
    # meta_gpt_scaffold(DEFAULT_TOOL_CONFIGS),
    agent_verse_scaffold(DEFAULT_TOOL_CONFIGS),
    # dylan_scaffold(DEFAULT_TOOL_CONFIGS),
]


tasks = [
    Task(
        dataset=read_dataset(
            # shuffle=shuffle,
        ),
        name=solver.__name__,
        solver=solver,
        scorer=includes(),
        sandbox=("docker", COMPOSE_FILE.as_posix()),
    )
    for solver in solvers
]


output = eval(
    tasks=tasks, model="openai/gpt-4o-mini", limit=10, max_tasks=500, log_format="json"
)

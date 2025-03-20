from inspect_ai import Task, task
from inspect_ai.scorer import includes
from inspect_ai._eval.eval import eval

from seed.react_and_plan_agent import react_and_plan_agent
from seed.cyber_recon_scaffold import cyber_recon_scaffold


from evals import (
    read_dataset,
    COMPOSE_FILE,
    DEFAULT_TOOL_CONFIGS,
    generate_dockerfile,
)


generate_dockerfile()

solvers = [
    react_and_plan_agent(DEFAULT_TOOL_CONFIGS),
    cyber_recon_scaffold(DEFAULT_TOOL_CONFIGS),
]

print(react_and_plan_agent.__name__, str(solvers[0]))

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
    tasks=tasks,
    model="openai/gpt-4o-mini",
    limit=5,
    max_tasks=500,
)

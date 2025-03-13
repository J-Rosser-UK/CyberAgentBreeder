from typing import Callable

from inspect_ai.log import transcript
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.solver import (
    Generate,
    SampleLimitExceededError,
    Solver,
    TaskState,
    fork,
    solver,
)
from inspect_ai.tool import Tool

from agents.basic_cyber_agent import SYSTEM_MESSAGE, TOOLS, basic_cyber_agent
from agents.utils.agent_store import AgentStore


def count_attempts(state: TaskState) -> int:
    count = 0
    for message in state.messages:
        if isinstance(message, ChatMessageTool) and message.function in ["submit"]:
            count += 1
    return count


@solver(name="clean_reset_transition")
def clean_reset_transition(
    *,
    init_system_message: str,
    init_task_message: str,
) -> Solver:
    """Transition to reset the agent state to the initial state."""

    async def clean_reset_transition(state: TaskState, generate: Generate) -> TaskState:
        state.tools = []
        state.messages = [
            ChatMessageSystem(content=init_system_message),
            ChatMessageUser(content=init_task_message),
        ]
        return state

    return clean_reset_transition


def create_agent_initializer_by_token_limit(
    tools: list[Tool] = TOOLS,
) -> Callable[..., Solver]:
    """Returns a function that is used to initialize a sub agent.

    Args:
        tools (list[Tool]): The tools available to the agent.
    """

    def agent_initializer_by_token_limit(
        attempts_remaining: int, token_limit: int, tools: list[Tool] = tools
    ) -> Solver:
        """Function that initializes a sub agent.

        Args:
            attempts_remaining (int): The number of attempts remaining.
            token_limit (int): The token limit for the agent.
            tools (list[Tool]): The tools available to the agent.
        """
        return basic_cyber_agent(
            tools=tools,
            max_attempts=attempts_remaining,
            token_limit=token_limit,
            message_limit=1_000_000,  # inf
            system_message=None,  # system message will be set by the transition
        )

    return agent_initializer_by_token_limit


@solver(name="long_horizon_agent")
def long_horizon_agent(
    *,
    system_message: str = SYSTEM_MESSAGE,
    max_attempts: int = 30,
    token_limit: int = 100_000,
    token_limit_sub_agent: int = 50_000,
    sub_agent_initializer: Callable[
        ..., Solver
    ] = create_agent_initializer_by_token_limit(),
) -> Solver:
    """Agent that dispatches sub agents until the overall token limit is reached.

    Order of operations:
    1. Dispatches a sub agent
        - Sub agent executes until a successful submission or a resource limit is reached
          (i.e. token_limit_sub_agent or max attempts)
    2. Resets the agent state: removes all messages except for the system message and task
      message; the environment state is not reset
    3. Dispatches another sub agent

    1-3 loop until the task is completed or a resource limit is reached (i.e.
        token_limit_overall, max_attempts).

    Args:
        system_message (str): The system message to use for the sub agents.
        max_attempts (int): Maximum number of submissions to accept before
            terminating. Attempts are shared across all sub agents.
        token_limit (int): The overall token limit for the long horizon agent including
            all dispatched sub agents.
        token_limit_sub_agent (int): The token limit for each sub agent.
        sub_agent_initializer (Callable[..., Solver]): A function that returns a sub
            agent solver.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # extract task
        if isinstance(state.input, str):
            task = state.input
        else:
            raise ValueError("Input must be a string")

        transition = clean_reset_transition(
            init_system_message=system_message, init_task_message=task
        )

        # get next token limit
        def set_next_token_limit(state: TaskState) -> None:
            state.token_limit = min(
                token_limit, state.token_usage + token_limit_sub_agent
            )

        # log to transcript
        dispatch_count = 0

        def log(
            dispatch_count: int,
            token_usage: int,
            next_transition_at_tokens: int | None,
            submission_attempts_remaining: int,
        ) -> int:
            dispatch_count += 1
            transcript().info(
                f"Long Horizon Agent dispatching sub agent {dispatch_count}"
                f"\n - overall token limit: {token_limit}"
                f"\n - current token usage: {token_usage}"
                f"\n - next agent will execute until token usage reaches: {next_transition_at_tokens}"
                f"\n - submission attempts remaining: {submission_attempts_remaining}"
            )
            return dispatch_count

        # dispatch first agent
        state = await fork(state, transition)
        set_next_token_limit(state)
        dispatch_count = log(
            dispatch_count, state.token_usage, state.token_limit, max_attempts
        )
        try:
            state = await fork(
                state, sub_agent_initializer(max_attempts, state.token_limit)
            )
        except SampleLimitExceededError as e:
            if e.state is None:
                raise e
            state = e.state

        # reset token limit
        set_next_token_limit(state)

        attempts_remaining = max_attempts - count_attempts(state)

        while not state.completed and not state.store_as(AgentStore).terminate:
            # execution transition
            state = await fork(state, transition)

            set_next_token_limit(state)

            # dispatch sub agent
            dispatch_count = log(
                dispatch_count,
                state.token_usage,
                state.token_limit,
                attempts_remaining,
            )
            try:
                state = await fork(
                    state,
                    sub_agent_initializer(
                        attempts_remaining=attempts_remaining,
                        token_limit=state.token_limit,
                    ),
                )
            except SampleLimitExceededError as e:
                if e.state is None:
                    raise e
                state = e.state

            set_next_token_limit(state)

            attempts_remaining = max_attempts - count_attempts(state)

        return state

    return solve
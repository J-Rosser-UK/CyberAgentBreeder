from inspect_ai._eval.eval import eval
from inspect_ai.tool import Tool, ToolCall, ToolResult, bash, tool, python
from dataclasses import dataclass
from typing import Callable


@dataclass
class ToolConfig:
    """Additional configuration required for the react_and_plan_agent to make use of a tool

    Args:
        name (str): Name of the tool
        tool (Tool): Tool to be used
        formatter (Callable[[ToolCall], str]): Function to produce a string
            representation of the tool call
    """

    name: str
    tool: Tool
    formatter: Callable[[ToolCall], str]


@tool
def submit() -> Tool:
    async def execute(answer: str) -> ToolResult:
        """Submit an answer for evaluation.

        Args:
            answer (str): Submitted answer
        """
        return answer

    return execute


DEFAULT_TOOL_CONFIGS = [
    ToolConfig(
        name="bash",
        tool=bash(timeout=180),
        formatter=lambda tool_call: tool_call.arguments["cmd"],
    ),
    ToolConfig(
        name="python",
        tool=python(timeout=180),
        formatter=lambda tool_call: tool_call.arguments["code"],
    ),
    ToolConfig(
        name="submit",
        tool=submit(),
        formatter=lambda tool_call: f"submit: {tool_call.arguments['answer']}",
    ),
]

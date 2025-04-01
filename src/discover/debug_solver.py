"""
Debug solver module for the CyberAgentBreeder project.

This module provides a simple debugger solver that runs scaffold code from the input
on a sample in a simplified manner, catching any exceptions and returning the traceback
to the agent.
"""

import traceback
from textwrap import dedent
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    call_tools,
    get_model,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, ToolCall, ToolResult, bash, python


@tool
def submit() -> Tool:
    async def execute(answer: str) -> ToolResult:
        """Submit an answer for evaluation.

        Args:
            answer (str): Submitted answer
        """
        return answer

    return execute


@solver
def debug_solver(tools) -> Solver:
    """
    A debug solver that runs scaffold code from the input on a sample and catches any exceptions.

    This solver executes the scaffold code provided in the task input against a sample challenge
    and catches any exceptions, returning the traceback to the agent. It's a simpler alternative
    to the react_and_plan agent for debugging purposes.

    Args:
        tools: List of tool configurations to be used by the solver

    Returns:
        Solver function that executes the scaffold and returns debugging information
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Setup tools
        state.tools = [tool.tool for tool in tools]
        tool_names = {tool.name for tool in tools}
        tool_formatters = {tool.name: tool.formatter for tool in tools}

        # Extract scaffold code from input
        task = ""
        if len(state.messages) < 1:
            raise ValueError("Expected at least one message. Got none.")

        for message in state.messages:
            if isinstance(message, ChatMessageUser):
                if isinstance(message.content, str):
                    task += message.content

        if not task:
            raise ValueError("Could not extract task/scaffold from messages.")

        # Start the debugging process
        debug_intro = "Starting scaffold debugging process. Will execute the scaffold and catch any exceptions."
        state.messages.append(ChatMessageAssistant(content=debug_intro))

        try:
            # Create a wrapper to run the scaffold code in a try-except block
            debug_code = f"""
try:
    # Original scaffold code
{task.strip()}

    # Test execution with minimal sample
    import inspect
    from inspect_ai.solver import TaskState, Generate
    
    # Create mock objects for testing
    class MockState:
        def __init__(self):
            self.input = "Sample CTF task for debugging"
            self.messages = []
            self.tools = []
            self.output = type('obj', (object,), {'completion': None})
            self.completed = False
            self.store = {{}}
    
    class MockGenerate:
        async def __call__(self, *args, **kwargs):
            return None
    
    # Find all solver functions in the code
    solvers = []
    for name, obj in list(locals().items()):
        if callable(obj) and hasattr(obj, '__name__') and 'solver' in getattr(obj, '__annotations__', {{}}).get('return', ''):
            solvers.append(obj)
    
    if not solvers:
        print("No solver functions found in the code.")
    else:
        print(f"Found {{len(solvers)}} solver function(s): {{[s.__name__ for s in solvers]}}")
        
        # Create mock tool configs
        class MockToolConfig:
            def __init__(self, name):
                self.name = name
                self.tool = None
                self.formatter = lambda x: str(x)
        
        mock_tools = [MockToolConfig("bash"), MockToolConfig("python"), MockToolConfig("submit")]
        
        # Try to initialize the main solver
        solver_func = solvers[-1](mock_tools)
        print(f"Successfully initialized the solver function: {{solvers[-1].__name__}}")
        
        # Create state and generate mocks
        state = MockState()
        generate = MockGenerate()
        
        # Call the solver function with mock objects
        print("Executing solver with mock objects...")
        import asyncio
        result = asyncio.run(solver_func(state, generate))
        print("Solver executed successfully!")
    
except Exception as e:
    import traceback
    print("\\n\\nERROR TRACEBACK:")
    traceback.print_exc()
"""

            # Execute the debug code
            tool_call = ToolCall(
                id="python",
                function="python",
                arguments={"code": debug_code},
                type="function",
            )

            # Add the debug attempt to messages
            state.messages.append(
                ChatMessageAssistant(
                    content="Executing scaffold code in debug mode",
                    tool_calls=[tool_call],
                )
            )

            # Execute the tool call
            tool_results = await call_tools(state.messages[-1], state.tools)

            # Process the result
            tool_result_text = next(
                (result.text for result in tool_results if result.function == "python"),
                None,
            )

            if tool_result_text:
                # Set the debug output as the completion
                state.output.completion = tool_result_text

                # Format the output as a helpful debugging message
                debug_summary = "## Scaffold Debug Results\n\n"

                if "ERROR TRACEBACK:" in tool_result_text:
                    debug_summary += "❌ **Errors detected in scaffold code**\n\n"
                    debug_summary += "```python\n"
                    error_part = tool_result_text.split("ERROR TRACEBACK:")[1].strip()
                    debug_summary += error_part
                    debug_summary += "\n```\n\n"
                    debug_summary += (
                        "Please fix the errors in your scaffold code and try again."
                    )
                else:
                    debug_summary += (
                        "✅ **No errors detected in scaffold initialization**\n\n"
                    )
                    debug_summary += "```\n"
                    debug_summary += tool_result_text
                    debug_summary += "\n```\n\n"
                    debug_summary += "The scaffold initialized successfully. Further testing with actual challenges may reveal runtime issues."

                state.messages.append(ChatMessageAssistant(content=debug_summary))

        except Exception as e:
            error_traceback = traceback.format_exc()
            error_message = (
                f"Error during debugging process:\n```\n{error_traceback}\n```"
            )
            state.messages.append(ChatMessageAssistant(content=error_message))
            state.output.completion = error_message

        return state

    return solve

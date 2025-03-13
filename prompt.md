## Introduction

`inspect_ai` is an open-source framework designed to facilitate complex agentic scaffolding and evaluation for large language models (LLMs). Its primary purpose is to help build and evaluate AI agents by providing components for prompt engineering, multi-turn dialogues, and tool usage​. In `inspect_ai`, solvers and tools are central concepts that enable AI agents to solve tasks in structured ways. A solver is essentially a function (or chain of functions) that directs how an AI should process a task, possibly over multiple steps. Solvers can implement strategies like prompt formatting, multi-turn reasoning, or decision-making logic, and they often call the LLM to generate answers at the appropriate times. Tools, on the other hand, are external functions or APIs that the AI can invoke via the framework to extend its capabilities beyond text generation. By using tools, an AI agent can perform actions such as calculations, web searches, or database queries as part of its reasoning process​. Together, solvers and tools allow the creation of agentic scaffolds – structured processes where an AI model plans, acts (via tools), and observes results in a loop to tackle complex tasks. This document will guide you through writing custom solvers, integrating tools, and designing multi-agent scaffolds using the `inspect_ai` framework, with a focus on clarity and specificity for automated generation.


## Writing a Solver
A solver in `inspect_ai` defines the logic an AI agentic scaffold follows to solve a given task, one could even use the terms scaffold and solver interchangeably. Technically, a solver is a callable that transforms a task’s state into a result​. Solvers are defined using the @solver decorator, which registers your solver function with the framework for logging and configuration purposes​. The typical signature of a solver function is:

```python
@solver  
def my_solver(...parameters...):  
    async def solve(state: TaskState, generate: Generate) -> TaskState:  
        # solver logic here
        return state  
    return solve  
```

### TaskState
Here, state is a `TaskState` object representing the current state of the task (including the conversation history, user input, and any intermediate or final outputs), and generate is a function provided by `inspect_ai` to call the language model for a new completion.
```python
class TaskState:
    ...
    @property
    def input(self) -> str | list[ChatMessage]:
        """Input from the `Sample`, should be considered immutable."""
        return self._input

    @property
    def metadata(self) -> dict[str, Any]:
        """Metadata from the `Sample` for this `TaskState`"""
        return self._metadata

    @property
    def messages(self) -> list[ChatMessage]:
        """
        Chat conversation history for sample.

        This will generally get appended to every time a `generate` call is made
        to the model. Useful for both debug and for solvers/scorers to assess
        model performance or choose the next step.
        """
        return self._messages

    @property
    def output(self) -> ModelOutput:
        """
        The 'final' model output once we've completed all solving.

        For simple evals this may just be the last `message` from the
        conversation history, but more complex solvers may set this directly.
        """
        return self._output

    @property
    def tools(self) -> list[Tool]:
        """Tools available to the model."""
        return self._tools

    @property
    def target(self) -> Target:
        """The scoring target for this `Sample`."""
        return self._target

    @property
    def completed(self) -> bool:
        """Is the task completed.

        Additionally, checks message and token limits and raises if they are exceeded, and also checks for an operator interrupt of the sample.
        """
        from inspect_ai.log._samples import set_active_sample_total_messages

        from ._limit import SampleLimitExceededError

        # update messages
        set_active_sample_total_messages(len(self.messages))

        if self._completed:
            return True
        elif self.message_limit and len(self.messages) >= self.message_limit:
            raise SampleLimitExceededError(
                "message",
                value=len(self.messages),
                limit=self.message_limit,
                state=self,
            )
        elif self.token_limit and self.token_usage >= self.token_limit:
            raise SampleLimitExceededError(
                "token", value=self.token_usage, limit=self.token_limit, state=self
            )
        else:
            check_sample_interrupt()
            return self._completed

    @property
    def store(self) -> Store:
        """Store for shared data"""
        return self._store

    ...

```

The TaskState holds important context: for example, state.messages is the list of chat messages exchanged so far (the conversation history), state.user_prompt is a convenience reference to the first user message, and state.output will hold the model’s most recent answer​. The solver’s job is to update this state (e.g. by adding messages or calling the LLM) to eventually produce a final answer in state.output. The @solver decorator ensures the solver is properly registered (making it appear in logs and usable in eval configurations) and typically wraps an inner async solve() function where the actual logic resides​. Basic Solver Example:
Below is an example of a simple solver that takes a user question from the TaskState, calls an LLM to get an answer, and returns the updated state:
```python
@solver  
def answer_direct():  
    async def solve(state: TaskState, generate: Generate) -> TaskState:  
        # Directly pass the current state (with user's prompt) to the model  
        state = await generate(state)  
        return state  # state.output now contains the model's answer  
    return solve  
```
### Generate
This few_shot_solver solver does minimal processing: it simply uses generate(state) to prompt the model with whatever is currently in the state (typically the user’s question) and waits for the model’s completion. The generate function is asynchronous, so we await it to get the model’s response. After calling generate, state.output will contain the model’s reply, and the full conversation (in state.messages) will include the assistant’s answer appended. The solver then returns the state, ending the solving process with the model’s answer ready.

Whilst you can use the "generate" parameter provided to return a model response, instead to enable more complex scaffolds, it is better practice to define an agent based off of the state.model and call its generate method.

Because solvers are just Python functions, they can maintain memory and multiple turns by preserving and updating the TaskState across calls. For instance, you can structure a solver to be composed of multiple few shot examples before calling the generate method:
```python
from inspect_ai.model import (
    get_model,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
)
import random

few_shot_dataset:list = ...
SYSTEM_PROMPT:str = ...
NUM_FEW_SHOT = 5


def _select_random_samples(dataset, num_samples):
    if len(dataset) < num_samples:
        return dataset
    random_samples = random.sample(dataset, num_samples)
    return random_samples


@solver  
def few_shot_solver():  
    async def solve(state: TaskState, generate: Generate) -> TaskState:

        # It's best practice to clear the state.messages at the start of every solver
        state.messages = []

        # Instead of using the "generate" argument, best practice is to define your own agents.
        agent_1 = get_model(state.model)

        while not state.completed:

            state.messages.append(ChatMessageSystem(content=SYSTEM_PROMPT))

            for sample in _select_random_samples(
                few_shot_dataset, NUM_FEW_SHOT
            ):

                few_shot_user_message = ChatMessageUser(
                    content=dedent(
                        f"""<user_problem>{sample["problem"]}</user_problem>

                    Please proceed with analyzing and solving the given problem and enclose your final answer in <answer></answer> tags."""
                    ).strip()
                )
                state.messages.append(few_shot_user_message)

                few_shot_assistant_message = ChatMessageAssistant(
                    content=self._remove_boxed_tags(sample["solution"])
                    + "\n <answer>"
                    + sample["answer"]
                    + "</answer>"
                )
                state.messages.append(few_shot_assistant_message)

            input_message = ChatMessageUser(content=state.input)

            state.messages.append(input_message)

            output = await agent_1.generate(state.messages, state.tools)
            
            state.messages.append(output.message)

            # Store the final output in state
            state.output = output

            state.completed = True

        return state  # state.output now contains the model's answer  
    return solve  
```

## Using Tools in Solvers
Tools in `inspect_ai` are functions that an AI agent can invoke to perform actions beyond text generation, such as calculations, web browsing, or database queries​. Integrating tools into your solver allows the agent to interact with the outside world or perform internal computations as part of its reasoning process. Tools are defined in Python and registered with the framework using the @tool decorator. This decorator makes a Python function available for the model to call during an agent’s operation. When defining a tool, you typically write a function that returns an inner execute coroutine. The inner function performs the actual action and should be annotated with parameter types and a docstring describing what it does. For example, here is a simple tool that adds two numbers:
```python
from inspect_ai import tool

@tool  
def add():  
    async def execute(x: int, y: int):  
        """Add two numbers.  
        Args:  
            x: First number to add.  
            y: Second number to add.  
        Returns:  
            The sum of the two numbers.  
        """  
        return x + y  
    return execute
```
In this example, @tool registers the add tool with the system. The execute function inside has type annotations for its arguments (x: int, y: int) and a docstring explaining each parameter and the return value. These annotations and docstring are important because they inform the language model how to format a call to this tool and what the tool does​. When the agent is running, the model might decide it needs to use this tool – for instance, if asked “What is 1 + 1?”, the model could choose to call the add tool with x=1 and y=1. To enable tool usage in a solver, you need to provide the tool to the model. One common way is to use the built-in use_tools() solver that comes with `inspect_ai`, which prepares the TaskState to allow tool calls. For example:
```python
from inspect_ai.solver import use_tools, generate

# Provide the 'add' tool to the model and then generate an answer
@task  
def addition_task():  
    return Task(  
        dataset=[Sample(input="What is 1 + 1?", target="2")],  
        solver=[  
            use_tools(add()),   # make the add tool available  
            generate()          # then call the model to solve the task  
        ]  
    )  

```
In this task definition, the solver is a list of two solvers: use_tools(add()) followed by generate(). The use_tools solver modifies the TaskState to include the add tool in the model’s tool list (and possibly a directive so the model knows it can use it)​. The subsequent generate() call will prompt the model with the user’s question plus information about the available tool. If the model decides to use the tool, it will output a structured tool invocation instead of a direct answer. Inspect’s runtime will detect this and execute the add function accordingly. Notably, the model doesn’t call add directly; it must produce a request (often a JSON or special format) indicating the tool name and arguments, and then the framework executes the function and returns the result back to the model​. For example, the model might output something like {"action": "add", "args": {"x": 1, "y": 1}}, and `inspect_ai` will run add(1,1) and supply the result (2) to the model as if it were a new message. The conversation then continues, allowing the model to use that result to formulate a final answer. You can define and register multiple tools and provide all of them to the solver, giving the agent a choice of actions. For instance, if you have a search() tool for web search and a calculate() tool for math, you can do use_tools(search(), calculate()) before calling generate(). The model will be aware of both tools and can dynamically choose which one to invoke based on the task at hand. When the model generates a tool invocation, inspect will match the tool name to the registered Python function, execute it, and return the output to the model, all seamlessly. As a solver writer, you can also handle tool results or errors if needed – by default, inspect will catch common errors (like timeouts or permission errors) and inform the model of the error in its next input​​, but you could wrap tool calls in try/except if you want custom behavior. The key point is that tools greatly extend what your AI agent can do, and the `inspect_ai` framework makes it straightforward to plug these tools into your solver’s logic. Calling Tools with Arguments: You don’t manually call execute in your solver; instead, you let the model decide when to use the tool. However, if you ever want to trigger a tool from your solver code (for example, as a forced step), you can call the Python function directly. For instance, result = await add()(x=1, y=1) would execute the add tool and give you the result, which you could then insert into the state (perhaps as a message from the assistant). In general, though, the typical pattern is to use use_tools to provide tools and let the agent invoke them. The framework handles passing arguments (based on the model’s output) and returning the result as part of the next model prompt. This design allows the AI to incorporate tool outputs into its reasoning process fluidly, just like an agent planning an action and seeing the outcome before continuing.


## Multi-Agent Scaffolds and Multiple Turns
An agentic scaffold refers to a structured arrangement where multiple solvers (or multiple steps of reasoning) work together, often emulating a multi-agent or multi-step interaction. In `inspect_ai`, you can compose solvers sequentially or even nest calls to create complex behavior. One way to build a scaffold is by listing several solvers in the Task’s solver list, which causes them to run one after the other on the same TaskState​. Each solver in the sequence can add to or transform the state, enabling a stepwise reasoning process. For example, you might use one solver to generate an initial answer and another to critique and improve that answer. Example Scaffold (Critique and Refine): Suppose we want an agent that answers a question, then evaluates its own answer and tries to improve it. We can achieve this by combining the basic generate() solver with a second solver that performs self-critique. Inspect provides a built-in solver self_critique() for this purpose, but for illustration, we’ll show the concept:
```python
@task  
def critique_scaffold():  
    return Task(  
        dataset=[Sample(input="How do airplanes fly?", target=None)],  
        solver=[  
            generate(),        # 1. Model generates an initial answer  
            self_critique()    # 2. Model critiques that answer and improves it  
        ]  
    )  
```
Here, the first solver simply calls the model to get an answer to the user’s question. The second solver (self_critique) takes the resulting answer in state.output and prompts the model again to analyze and refine the answer. Under the hood, self_critique() might add a new user message like “Please critique the above answer and correct any mistakes.” along with the original question and answer​​. The model then produces a critique and a revised answer in a follow-up turn. The final output is an improved answer, and the conversation (TaskState messages) now contains the full dialogue: the initial question, initial answer, critique, and revised answer. This scaffold effectively uses the model as both a responder and a critic in two roles. More generally, multiple solvers can interact by passing the TaskState along a pipeline of logic. One solver’s output becomes the input for the next. This is how you can implement multi-agent interactions: for instance, one solver could act as a “planner” that formulates a strategy or series of tool calls, and another solver could act as an “executor” that uses those plans to query tools and get results, before perhaps a final solver produces the answer to the user. Because all solvers share the same TaskState, any messages or data added by one will be visible to subsequent steps, providing continuity (memory) across turns. Multiple Turns and Memory: Agents often need to carry on a conversation for several turns, remembering what happened in prior turns. In `inspect_ai`, the memory of an agent is primarily the TaskState.messages list. Every time the model generates a message (via generate()), that message is appended to this list. Solvers can also append messages – for example, adding a system instruction or inserting an observation from a tool. By preserving the TaskState across turns, the agent naturally “remembers” prior interactions. If you want an agent to take multiple turns until a goal is achieved, you can implement a loop within a solver (as shown in the multi_turn_solver example earlier) or recursively call solvers. For instance, the built-in basic_agent() solver runs a loop where the model can alternate between proposing an action (like a tool use) and observing the result, continuing until it decides to stop​​. In your own custom solver, you could do something similar: repeatedly call generate() and handle the output (check if it’s a tool request, an answer, or a request to end the conversation) until some termination condition is met. Always ensure to include a safety stop (e.g. a max number of turns or a time limit) to prevent infinite loops if the model gets stuck​. By combining planning logic, tool usage, and multiple generations, you can create rich multi-agent scaffolds where an AI systematically works through complex tasks with memory and self-direction.
5. Best Practices for Writing Solvers
Designing effective solvers requires balancing flexibility for the AI and control to ensure reliability. Here are some best practices to consider:
Keep Solvers Focused and Modular: Each solver should have a clear purpose. If a task is complex, break the solution into multiple solvers or steps (as a scaffold). For example, one solver for formulating the query, one for using a tool, and one for finalizing the answer. This modular approach makes it easier to debug and reuse components. Use the pipeline capability (multiple solvers in a list) to your advantage to separate concerns, rather than one giant monolithic solver.
Leverage System Messages and Templates: Guide the model’s behavior by providing instructions via system messages or prompt templates. For instance, if you want the model to follow a certain format or strategy, a solver can prepend a system instruction to state.messages before calling generate(). This ensures the model knows how to use tools or when to stop, etc. Structured prompts lead to more reliable model behavior.
Control Token Usage: Multi-turn interactions and long conversations can consume many tokens. To keep the process efficient, limit unnecessary verbosity. You can instruct the model (via prompts) to be concise, or truncate irrelevant history if it grows too large. Inspect allows setting limits like message_limit or token_limit on tasks​ – make use of these to prevent runaway outputs or infinite reasoning loops. If your solver enters a loop, always have a breaking condition after a reasonable number of turns. Also consider using shorter prompts or summaries of previous turns if the conversation history gets very long.
Limit Tool Calls and Plan Their Use: While tools are powerful, uncontrolled tool use can lead to inefficient or error-prone behavior. Encourage the model (through instructions) to only call a tool when it’s needed. In some cases, you might want to intercept tool usage – for example, if the model keeps calling the same tool without making progress, your solver could step in and halt further calls or adjust the strategy. You could implement a counter for tool usage in the TaskState metadata and stop after a certain number. It’s also wise to handle exceptions from tools: by default, common errors are caught and returned to the model as error messages​, but if a specific tool is critical, you might catch errors and have the solver take a different action (like try an alternative approach or give a fallback answer).
Use Async and Concurrency Wisely: Define your solver’s inner solve function as async (as required by @solver) and always await calls to generate() or tool executions. This allows `inspect_ai` to manage concurrency (possibly running multiple generations in parallel if evaluating many samples). If your solver needs to do heavy computation, consider offloading it (e.g., a tool that runs in a subprocess) so as not to block the event loop. For network calls, use asynchronous HTTP libraries inside tools to avoid slowing down the solver loop​.
Logging and Debugging: Take advantage of Inspect’s logging and evaluation reports. Each solver’s actions (like prompt modifications, model outputs, tool calls) can be logged. Using the `inspect_ai` eval logs or a debug mode, you can get a transcript of what happened at each step. If a solver isn’t behaving as expected, insert temporary print statements or use the Inspect log viewer to inspect the TaskState after each solver step. Debugging multi-turn agents can be tricky, so build your scaffold incrementally: test each part (e.g., tool call in isolation, critique step alone) before combining them. This helps isolate issues.
Anticipate Failure Cases: Think about how your solver should handle model failures or unexpected outputs. For example, if the model’s answer is irrelevant or malformed, should the solver try again with a different prompt? If the model refuses to use a tool when it’s needed, you might add a gentle reminder in the prompt. Conversely, if the model overuses tools, maybe add a rule to proceed to an answer after a certain point. Common failure cases include the model misunderstanding the task, producing an invalid tool call format, or getting stuck in a loop without arriving at an answer. Plan for these by adding checks in your solver. Even a simple check like “if we’ve looped N times, stop and output the best attempt so far” can save time and avoid infinite loops. Use the state.scores or custom flags in state.metadata if you want to mark certain conditions (like partial success) for later analysis​.
By following these practices, you’ll write solvers that are robust, efficient, and easier to maintain, which in turn helps the LLM agent perform better and more predictably within the scaffold you’ve created.
6. Few-Shot Examples
To solidify these concepts, here are several example solver patterns with variations in structure, reasoning steps, and tool use. These serve as “few-shot” illustrations that you (or an LLM) can draw upon when generating new solvers:
Example: Chain-of-Thought Planning – Purpose: Insert a reasoning step before answering.
Description: This solver asks the model to think step-by-step first, then provide a final answer. It does this in two turns: first, it prompts the model for a plan or chain-of-thought, and on the second turn it asks for the answer.
Implementation Sketch: The solver could add an instruction like “Think aloud step by step, then conclude with the answer.” and call generate() to get the reasoning. The model’s reasoning (chain-of-thought) might come back as an assistant message. Then the solver appends another user message like “Based on the above reasoning, what is the final answer?” and calls generate() again for the final answer.
Code Snippet:
```python
@solver
def reasoning_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # 1. Ask for reasoning
        state.messages.append(ChatMessageUser(content="Please think step by step before answering."))
        state = await generate(state)  # model's chain-of-thought as assistant message
        # 2. Ask for final answer using the reasoning
        state.messages.append(ChatMessageUser(content="Now given your reasoning, provide a concise answer."))
        state = await generate(state)  # model's final answer
        return state
    return solve
```
Outcome: The model first produces a chain-of-thought (which is captured in the conversation), then it produces a final answer. This approach helps the model break down the problem, potentially leading to more accurate results for complex questions.
Example: Tool-Using Solver – Purpose: Utilize external tools (like APIs or a calculator) as part of answering a query.
Description: This solver lets the model fetch external information or compute something before answering. For instance, if asked a math word problem or a trivia question, the model can call a search tool or a calculation tool.
Implementation Sketch: Provide the relevant tools via use_tools and then prompt the model. Optionally, add a system message that encourages tool use (e.g., “You can use the calculator or search the web if needed to find the answer.”). The solver might run a loop where it alternates between the model’s tool use and providing the tool results until the model signals it has enough information to answer.
Code Snippet:
```python
@solver
def tool_using_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Make sure tools are available (could also be done outside this solver)
        state.tools = [ calculator(), web_search() ]  # assume these tools are defined
        state.tool_choice = None  # (framework will handle tool decision)
        # Initial prompt already in state.user_prompt
        # Call the model; it may return either an answer or a tool invocation
        state = await generate(state)
        # If the model called a tool, `inspect_ai` will handle execution and add the result to messages.
        # You might loop here to allow multiple tool uses:
        for _ in range(3):
            if state.output.tool_used:  # pseudo-flag indicating a tool was just used
                state = await generate(state)  # model responds after seeing tool result
            else:
                break  # model gave a final answer
        return state
    return solve
```
Outcome: The solver enables the model to use a calculator or web search as needed. For example, the model might first output an action to use calculator() with certain inputs; the framework executes it and appends the result. The solver then calls generate() again, now the model has the result and might either use another tool or give a final answer. This pattern can retrieve up-to-date info or perform calculations that the model alone might struggle with.
Example: Self-Reflection and Refinement – Purpose: Have the model review its answer and improve it (a simple Reflexion strategy).
Description: After an initial answer, the solver asks the model to reflect on whether the answer is correct or could be improved, and then produce a revised answer if necessary​​. This is similar to the earlier critique scaffold but can be done within one solver or a short sequence of solvers.
Implementation Sketch: The solver first gets an initial answer via generate(). Then it adds a new prompt like “Analyze the above answer. Is it fully correct and complete? If not, please correct it.” and calls generate() again. The model will either indicate the answer was fine or provide a better answer. You might even compare the two answers and decide which to keep.
Code Snippet:
```python
@solver
def self_refine_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # 1. Get initial answer
        state = await generate(state)
        initial_answer = state.output.completion
        # 2. Ask model to reflect on its answer
        state.messages.append(ChatMessageUser(content=
            "Please check your answer for correctness and completeness. Improve it if necessary."))
        state = await generate(state)
        refined_answer = state.output.completion
        # (Optional) Decide which answer to keep or combine
        # Here we assume the refined_answer is final.
        return state
    return solve
```
Outcome: The model effectively gets a second chance to correct itself. This often yields a more accurate or well-rounded answer. If the initial answer was already good, the prompt asks the model to confirm that, which the model can do (perhaps responding that the answer was correct). If there were issues, the model will produce a new answer. This pattern leverages the model’s own evaluative capabilities to improve results.
Example: Verification Step with External Data – Purpose: Verify the model’s answer using a reliable source or criteria, and have the model try again if it was wrong.
Description: In scenarios where you have a way to check the answer (e.g., a known correct result or an external oracle), you can incorporate a verification step. For instance, for arithmetic or factual questions, the solver could compute the true answer via a tool or compare against a known target. If the model’s answer is incorrect, the solver can prompt the model to try again or explain the mistake.
Implementation Sketch: After the model produces an answer, the solver compares state.output to state.target (if available) or uses a tool (like a calculator or database lookup) to verify. If the verification fails, the solver might append a message like “That doesn’t seem correct, please reconsider and answer again.” and call generate() for a second attempt. You may limit the retries to avoid loops.
Code Snippet:
```python
@solver
def verify_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # 1. Get model's answer
        state = await generate(state)
        user_question = state.input_text
        model_answer = state.output.completion
        # 2. Verify answer (for example, using a tool or known target)
        correct = False
        if state.target:  # if we have an expected answer
            correct = model_answer.strip() == state.target.strip()
        else:
            # If no target given, maybe use a tool for verification, e.g.:
            correct = await verify_via_tool(user_question, model_answer)
        # 3. If incorrect, ask model to try again
        if not correct:
            feedback = f"Your previous answer \"{model_answer}\" is not correct. Please try again."
            state.messages.append(ChatMessageUser(content=feedback))
            state = await generate(state)  # second attempt
        return state
    return solve
```
Outcome: This solver gives the model feedback if it was wrong and a chance to correct its answer. For example, if the question was “What is 7 * 8?” and the model answered “54”, the solver’s verification would detect the mistake (since the target or a calculator knows the answer is 56), then prompt the model to try again. The model, given this feedback, can reconsider and hopefully answer “56” on the next try. This pattern ensures higher accuracy when an objective check is available, effectively blending automated verification into the agent’s reasoning loop.
Each of the above examples can be adapted or combined based on your needs. The idea is that you can mix-and-match these techniques (planning, tool use, self-critique, verification) to create a solver or agent scaffold tailored to a specific objective. By studying these patterns, an LLM can learn how to structure new solvers in a similar fashion for different tasks.
7. Summary and Steps to Create a New Solver
In summary, the `inspect_ai` framework provides a powerful set of abstractions for building agentic AI systems. Solvers encapsulate strategies for interacting with the model (single-turn or multi-turn, with or without tools), and tools extend the model’s capabilities by letting it act on external functions. By composing solvers and tools, you can design complex agent workflows (scaffolds) where the model plans, executes actions, and iterates towards a solution. Key concepts include understanding the TaskState (which carries the conversation and results), using the @solver decorator to define custom logic, and incorporating the @tool functions for enhanced functionality. Always keep in mind best practices for reliability (like handling errors, limiting loops, and guiding the model with clear prompts). With these principles, an LLM can be guided to generate new solver implementations for a variety of tasks. Steps to Create a New Solver (Guide for an LLM):
Define the Objective: Clearly identify what the solver should accomplish. Is it a straightforward question-answering turn, or a multi-turn dialogue? Should the agent use any tools (e.g. web search, calculator) or special reasoning steps (planning, self-critique)? Determining the requirements upfront will inform the solver’s structure.
Plan the Solver Structure: Decide if the solver will be a single-turn or multi-turn process. Outline the steps the agent should take. For example, “First do X, then do Y, then produce answer.” If multiple distinct phases are needed, consider breaking them into sequential solvers or a loop within one solver. Identify where calls to the LLM (generate()) will happen and where tool usage might occur.
Define Tools (if needed): If your solver requires external actions, define the necessary tools using the @tool decorator. Write the execute function with proper type annotations and docstrings so the model can invoke it. Test the tool function independently if possible to ensure it works. For instance, if building a solver for a math task, you might define an @tool def calculate() or use an existing one like bash() for computations.
Write the Solver with @solver: Create a new solver function with the @solver decorator. The function should return an inner async def solve(state: TaskState, generate: Generate) -> TaskState function. Inside this solve function, implement the logic you planned: set up any initial system/user messages, call generate() to get model outputs, check those outputs or call tools as needed, and possibly loop or sequence through multiple steps. Use state.messages.append(...) to add any prompts (e.g., follow-up questions or tool results) that the model should see in subsequent turns. Ensure each generate() call is awaited and the state is updated accordingly.
Integrate Tools into the Solver: If using tools, there are two ways to integrate them. The simplest is to use the use_tools([...]) solver separately in a pipeline (as shown earlier) so that the model is aware of available tools. Alternatively, if writing a custom agent loop inside your solver, set state.tools to include your tool functions and possibly guide the model via a prompt to use them. Make sure to handle the model’s tool invocation outputs: after a generate() call, check if the model requested a tool (`inspect_ai` might provide this info via the TaskState or a special output format), then execute the tool (which the framework can do automatically) and provide the results to the model (often this is automated when using use_tools + generate). In a custom setup, you might manually call the tool and then insert its output into the conversation.
Incorporate Memory and Turns: If the solver is multi-turn, implement the loop or sequence logic. Keep track of how many turns have happened and include break conditions. For example, break out of a loop if the model’s last message indicates completion (or if a turn limit is reached). Make sure the conversation (state.messages) persists between turns, and add any necessary context for each turn (like re-asking the question, providing new info, or reminding the model of the goal).
Finalize the Output: After the solver’s logic is done (for instance, the loop ends or the final solver in a list has run), ensure that state.output contains the answer you want to present as the result. Usually, the last call to generate() will have set state.output to the model’s final answer. The solver should then return the TaskState. Optionally, you could set some state.scores or metadata if you want to record confidence or other info, but this is not required for basic operation.
Testing and Iteration: Simulate the solver on sample inputs to see how the model behaves. Check the logs or TaskState after each step to confirm it’s doing what you expect. If the model isn’t using a tool or is giving an irrelevant answer, you may need to refine your prompts or instructions in the solver. Debug any errors (e.g., JSON parse errors if the model’s tool call format is wrong – you might need to adjust the prompt format or tool spec). Iterate on the solver’s design: maybe add a hint if the model gets something wrong, or tighten the system message if it deviates from the plan.
By following these steps, an LLM (or a developer) can systematically create a new solver from scratch. Each step ensures that the solver is well-thought-out and that all pieces (prompts, tools, logic) are in place before final deployment. Using the examples and best practices given in this guide, an LLM can generate variations of solvers tailored to different tasks – whether it’s a single-turn Q&A, a multi-turn assistant with internet access, a self-correcting quiz solver, or any other agentic scaffold.
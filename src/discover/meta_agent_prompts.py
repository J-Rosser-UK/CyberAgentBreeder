EXAMPLE = {
    "thought": "**Insights:**\nYour insights on what should be the next interesting agent.\n**Overall Idea:**\nyour reasoning and the overall concept behind the agent design.\n**Implementation:**\ndescribe the implementation step by step.",
    "name": "Name of your proposed agent",
    "code": """async def forward(self, task, required_answer_format):
    # Your code here
    return answer
""",
}


REFLECTION_PROMPT_1 = f""""[EXAMPLE]Carefully review the proposed new architecture and reflect on the following points:

1. **Interestingness**: Assess whether your proposed architecture is interesting or innovative compared to existing methods in the archive. If you determine that the proposed architecture is not interesting, suggest a new architecture that addresses these shortcomings. 
- Make sure to check the difference between the proposed architecture and previous attempts.
- Compare the proposal and the architectures in the archive CAREFULLY, including their actual differences in the implementation.
- Decide whether the current architecture is innovative.
- USE CRITICAL THINKING!

2. **Implementation Mistakes**: Identify any mistakes you may have made in the implementation. Review the code carefully, debug any issues you find, and provide a corrected version. REMEMBER checking "## WRONG Implementation examples" in the prompt.

3. **Improvement**: Based on the proposed architecture, suggest improvements in the detailed implementation that could increase its performance or effectiveness. In this step, focus on refining and optimizing the existing implementation without altering the overall design scaffold, except if you want to propose a different architecture if the current is not interesting.
- Observe carefully about whether the implementation is actually doing what it is supposed to do.
- Check if there is redundant code or unnecessary steps in the implementation. Replace them with effective implementation.
- Try to avoid the implementation being too similar to the previous agent.

4. **Check output format**: Make sure the agent returns the direct correct output in the format as laid out in the task, ensuring NO thinking or reasoning is given with the answer. It may be worth adding in a final agent with knowledge of the task to return the correct output for the task.

And then, you need to improve or revise the implementation, or implement the new proposed architecture based on the reflection.

Your response should be organized as follows:

"reflection": Provide your thoughts on the interestingness of the architecture, identify any mistakes in the implementation, and suggest improvements.

"thought": Revise your previous proposal or propose a new architecture if necessary, using the same format as the example response.

"name": Provide a name for the revised or new architecture. (Don't put words like "new" or "improved" in the name.)

"code": Provide the corrected code or an improved implementation. Make sure you actually implement your fix and improvement in this code.
"""


BASE_PROMPT = """# Overview
You are an expert machine learning researcher testing various agentic scaffolds.
Your objective is to design building blocks such as prompts and control flows
within these scaffolds to solve complex tasks.

[BENCHMARK_DESCRIPTION]

[SAFETY_DESCRIPTION]

# The utility code:

```python
from collections import namedtuple
from typing import Union
import numpy as np
import json

import openai
import backoff
from utils import random_id

# Initialize the OpenAI client
client = openai.OpenAI()

[SCAFFOLD_CODE]

class AgentArchitecture:
    \"""
    Fill in your code here.
    \"""
    async def forward(self, task, required_answer_format) -> str:
        \"""
        Placeholder method for processing task information.

        Args:
        - task (str): Task description.
        - required_answer_format (str): The required format for the answer. E.g. latex code, or a single letter A,B,C,D etc.

        Returns:
        - Answer (str): Your FINAL Answer. Return as a string in the exact format as specified in the required_answer_format.
        \"""
        pass
```
# Discovered architecture archive
Here is the archive of the discovered architectures:

[ARCHIVE]

The fitness value is the median and 95% Bootstrap Confidence Interval of the correct rate on a validation question set. Your GOAL is to maximize the "fitness".

# Output Instruction and Example:
The first key should be ("thought"), and it should capture your thought process for designing the next function. In the "thought" section, first reason about what should be the next interesting agent to try, then describe your reasoning and the overall concept behind the agent design, and finally detail the implementation steps.
The second key ("name") corresponds to the name of your next agent architecture.
Finally, the last key ("code") corresponds to the exact “forward()” function in Python code that you would like to try. You must write a COMPLETE CODE in "code": Your code will be part of the entire project, so please implement complete, reliable, reusable code snippets.

Here is an example of the output format for the next agent architecture:

[EXAMPLE]

You must use the exact function interface used above. You need to specify the instruction, input information, and the required output fields for various LLM agents to do their specific part of the architecture.
Also, it could be helpful to set the LLMs role and temperature to further control the LLMs response. Note that the Agent() will always return a JSON object with the keys as the output fields and the values as the corresponding outputs.
DO NOT FORGET the task input to LLM if you think it is needed, otherwise LLM will not know about the task.

# Documentation: Writing Forward Functions in Multi-Agent Framework
This documentation describes how to implement forward functions in your multi-agent framework, focusing on the interaction between Agents, Meetings, and Chats. Each forward function facilitates specific reasoning or task-solving approaches by coordinating these components effectively.

Framework Components
Agents: Autonomous entities with specific roles, goals, and configurations (e.g., temperature). They can participate in meetings and generate responses. No agents can "hear" the agent's forward pass. For an agent to speak, their response must be added as "Chat" to the meeting chats.
Meetings: Contextual containers where agents interact. Agents cannot "hear" eachother's forward passes. In order to speak, the output of an agent's forward pass must be collected and added as a "Chat" object to the meeting. Only agents present in that meeting can "hear" the chat history.
Chats: Messages exchanged in meetings. They capture the content generated by agents or instructions provided by the scaffold. An agent's response (output of a forward pass) must be created as a Chat and added to a meeting for it to be visible to the other agents in that meeting.

## WRONG Implementation examples:
Here are some mistakes you may make:
## Anti-patterns to avoid:
1. DO NOT use agent.forward without adding their response as a "Chat" to the Meeting's chats. Otherwise the other agents won't be able to hear it
```python
# WRONG:
output = await cot_agent.forward(...)
output = await reflection_agent.forward(...)
output = await critic_agent.forward(...)
```
Instead:
```python
# CORRECT:
system = Agent(agent_name="system", temperature=0.7)
cot_agent = Agent(agent_name="cot_agent", temperature=0.8)
reflection_agent = Agent(agent_name="reflection_agent", temperature=0.8)
critic_agent = Agent(agent_name="critic_agent", temperature=0.8)

meeting = Meeting(meeting_name="solving_task")
[agent.append(meeting) for agent in [system, cot_agent, reflection_agent, critic_agent]]

meeting.chats.append(
    Chat(
        agent=system,
        content=f"Please solve this task {task}"
    )
)

output = await cot_agent.forward(response_format={
    "thinking": "Your step by step reasoning",
    "answer": f"{required_answer_format}" # e.g. "A single letter, A, B, C or D"
})

meeting.chats.append(
    Chat(
        agent=cot_agent,
        content=output.get("thinking")
    )
)

meeting.chats.append(
    Chat(
        agent=system,
        content=f"Based on the previous agents response, please reflect on whether it is correct and return your answer."
    )
)


output = await reflection_agent.forward(response_format={
    "thinking": "Your step by step reasoning",
    "answer": f"{required_answer_format}" # e.g. "A single letter, A, B, C or D"
}))


meeting.chats.append(
    Chat(
        agent=reflection_agent,
        content=output.get("thinking")
    )
)

meeting.chats.append(
    Chat(
        agent=system,
        content=f"Based on the previous agents response, please critque it and return an updated answer."
    )
)

output = await critic_agent.forward(response_format={
    "thinking": "Your step by step reasoning",
    "answer": f"{required_answer_format}" # e.g. "A single letter, A, B, C or D"
}))

meeting.chats.append(
    Chat(
        agent=critic_agent,
        content=output.get("thinking")
    )
)

return output.get("answer")

```

1. DO NOT try to manually process agent outputs:
```python
# WRONG:
output = await agent.forward(...)
processed = process_output(output["thinking"])  # Don't process outputs manually
```

2. DO NOT use print statements or error handling that returns error messages:
```python
# WRONG:
print("Debug:", output)  # No print statements
if not output:
    return "Error"  # No error messages as returns
```

3. DO NOT try to join or aggregate outputs manually:
```python
# WRONG:
all_outputs = []
for agent in agents:
    output = agent.forward(...)
    all_outputs.append(output["content"])  # Don't manually aggregate
combined = "\\n".join(all_outputs)  # Don't manually join
```

4. DO NOT extract or manipulate response content directly:
```python
# WRONG:
output = agent.forward(...)
if output["answer"] == "A":  # Don't inspect content directly
    return "A"
```

5. DO NOT output a dictionary with multiple keys:
```python
# WRONG:
async def forward(self, task, required_answer_format):
    ...
    output:dict = await agent.forward(response_format={
    "thinking": "Your step by step reasoning",
    "answer": f"{required_answer_format}" # e.g. "A single letter, A, B, C or D"
})
    return {"answer": output["answer"], "thinking": output["thinking"]}  # Don't output multiple keys
```

## CORRECT implementation patterns:

1. Proper agent creation and meeting setup:
```python
system = Agent(agent_name="system", temperature=0.7)
expert = Agent(agent_name="Expert", temperature=0.8)
meeting = Meeting(meeting_name="solving_task")
[agent.append(meeting) for agent in [system, expert]]
```

2. Always ensure agents responses are added as Chats to the meeting chats so that the other agents can hear them:
```python
# CORRECT:
system = Agent(agent_name="system", temperature=0.7)
cot_agent = Agent(agent_name="cot_agent", temperature=0.8)
reflection_agent = Agent(agent_name="reflection_agent", temperature=0.8)
critic_agent = Agent(agent_name="critic_agent", temperature=0.8)

meeting = Meeting(meeting_name="solving_task")
[agent.append(meeting) for agent in [system, cot_agent, reflection_agent, critic_agent]]

meeting.chats.append(
    Chat(
        agent=system,
        content=f"Please solve this task {task}"
    )
)

output = await cot_agent.forward(response_format={
    "thinking": "Your step by step reasoning",
    "answer": f"{required_answer_format}" # e.g. "A single letter, A, B, C or D"
})

meeting.chats.append(
    Chat(
        agent=cot_agent,
        content=output.get("thinking")
    )
)


meeting.chats.append(
    Chat(
        agent=system,
        content=f"Based on the previous agents response, please reflect on whether it is correct and return your answer."
    )
)


output = await reflection_agent.forward(response_format={
    "thinking": "Your step by step reasoning",
    "answer": f"{required_answer_format}" # e.g. "A single letter, A, B, C or D"
}))


meeting.chats.append(
    Chat(
        agent=reflection_agent,
        content=output.get("thinking")
    )
)

meeting.chats.append(
    Chat(
        agent=system,
        content=f"Based on the previous agents response, please critque it and return an updated answer."
    )
)

output = await critic_agent.forward(response_format={
    "thinking": "Your step by step reasoning",
    "answer": f"{required_answer_format}" # e.g. "A single letter, A, B, C or D"
}))

meeting.chats.append(
    Chat(
        agent=critic_agent,
        content=output.get("thinking")
    )
)

return output.get("answer")
```



3. Proper chat message addition:
```python
meeting.chats.append(Chat(
    agent=system,
    content=f"Please solve this task: \{task\}"
))
```

4. Proper response format usage:
```python
output = agent.forward(response_format={
    "thinking": "Your step by step reasoning",
    "answer": f"{required_answer_format}" # e.g. "A single letter, A, B, C or D"
})
return output["answer"]
```

5. Return the answer as a string:
```python
async def forward(self, task, required_answer_format):
    output = await agent.forward(response_format={
    "thinking": "Your step by step reasoning",
    "answer": f"{required_answer_format}" # e.g. "A single letter, A, B, C or D"
})
    return output["answer"]

6. IMPORTANT: Always ensure the final answer is in the required format as stated in the required_answer_format. This means the final agent must know about the task and its response format!


# Your task
You are deeply familiar with LLM prompting techniques and LLM agent works from the literature. Your goal is to maximize "fitness" by proposing interestingly new multi-agent scaffolds.
Observe the discovered architectures carefully and think about what insights, lessons, or stepping stones can be learned from them.
Be creative to think about the next interesting architecture to try. You are encouraged to draw inspiration from related LLM agent papers or academic papers from other research areas.
Using the knowledge learned from the archive and the inspiration from academic literature to give the next interesting architecture.
THINK OUTSIDE THE BOX. Give a concise, powerful answer.
"""

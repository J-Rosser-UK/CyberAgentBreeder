"""
MetaGPT Solver Scaffold

This scaffold implements a multi-phase approach to solving Capture The Flag (CTF) challenges
with multiple specialized agents working together. It uses a progressive workflow that adapts
based on challenge type detection, specialized tool selection, and iterative hypothesis testing.

Key features:
1. Challenge classification to identify the CTF type (cryptography, web, forensic, etc.)
2. Multiple specialized agents with different expertise areas
3. Progressive workflow with hypothesis testing and verification
4. Structured knowledge sharing using centralized memory
5. Adaptive tool selection based on challenge requirements
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
from enum import Enum
import re
import json
import time

from inspect_ai.model import (
    CachePolicy,
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    call_tools,
    get_model,
)
from inspect_ai.scorer import ValueToFloat, score, value_to_float
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, ToolCall, ToolResult, bash, python, tool


# === Challenge Classification ===


class ChallengeType(Enum):
    """Types of CTF challenges"""

    UNKNOWN = "unknown"
    CRYPTOGRAPHY = "cryptography"
    WEB = "web"
    FORENSICS = "forensics"
    REVERSE_ENGINEERING = "reverse_engineering"
    BINARY_EXPLOITATION = "binary_exploitation"
    STEGANOGRAPHY = "steganography"
    OSINT = "osint"
    GENERAL_SKILLS = "general_skills"


# === Workflow Phases ===


class SolverPhase(Enum):
    """Phases of CTF solving process"""

    CLASSIFICATION = "classification"
    RECONNAISSANCE = "reconnaissance"
    HYPOTHESIS = "hypothesis"
    EXPLOITATION = "exploitation"
    VERIFICATION = "verification"
    SUBMISSION = "submission"


# === Memory and Knowledge Management ===


@dataclass
class ObservationItem:
    """A single observation from reconnaissance or analysis"""

    content: str
    source: str
    importance: float = 1.0
    timestamp: int = 0
    tags: List[str] = field(default_factory=list)


@dataclass
class HypothesisItem:
    """A potential solution path or flag candidate"""

    hypothesis: str
    evidence: List[str]
    confidence: float
    verified: bool = False
    timestamp: int = 0


@dataclass
class ToolItem:
    """A tool that might be useful for the challenge"""

    name: str
    description: str
    usage_examples: List[str]
    relevance_score: float = 0.0


class CTFMemory:
    """Centralized memory for CTF solving process"""

    def __init__(self, store):
        self.store = store

        # Initialize memory structures
        if self.store.get("challenge_types") is None:
            self.store.set("challenge_types", {})

        if self.store.get("observations") is None:
            self.store.set("observations", [])

        if self.store.get("hypotheses") is None:
            self.store.set("hypotheses", [])

        if self.store.get("relevant_tools") is None:
            self.store.set("relevant_tools", [])

        if self.store.get("flag_patterns") is None:
            self.store.set("flag_patterns", [])

        if self.store.get("current_phase") is None:
            self.store.set("current_phase", SolverPhase.CLASSIFICATION.value)

        if self.store.get("turn_count") is None:
            self.store.set("turn_count", 0)

        if self.store.get("execution_history") is None:
            self.store.set("execution_history", [])

    def add_observation(
        self, content: str, source: str, importance: float = 1.0, tags: List[str] = None
    ):
        """Add a new observation to memory"""
        if tags is None:
            tags = []

        turn_count = self.store.get("turn_count", 0)
        observations = self.store.get("observations", [])

        # Store as dict for persistence
        item_dict = {
            "content": content,
            "source": source,
            "importance": importance,
            "timestamp": turn_count,
            "tags": tags,
        }

        observations.append(item_dict)
        self.store.set("observations", observations)

        # Also check if this might be a flag pattern
        self._check_for_flag_patterns(content)

    def add_hypothesis(self, hypothesis: str, evidence: List[str], confidence: float):
        """Add a potential solution hypothesis"""
        turn_count = self.store.get("turn_count", 0)
        hypotheses = self.store.get("hypotheses", [])

        # Store as dict for persistence
        item_dict = {
            "hypothesis": hypothesis,
            "evidence": evidence,
            "confidence": confidence,
            "verified": False,
            "timestamp": turn_count,
        }

        hypotheses.append(item_dict)
        self.store.set("hypotheses", hypotheses)

    def set_challenge_type_score(
        self, challenge_type: ChallengeType, challenge_score: float
    ):
        """Set confidence score for a challenge type"""
        challenge_types = self.store.get("challenge_types", {})
        challenge_types[challenge_type.value] = challenge_score
        self.store.set("challenge_types", challenge_types)

    def get_primary_challenge_type(self) -> ChallengeType:
        """Get the most likely challenge type based on scores"""
        challenge_types = self.store.get("challenge_types", {})

        if not challenge_types:
            return ChallengeType.UNKNOWN

        # Find type with highest score
        primary_type = max(challenge_types.items(), key=lambda x: x[1])[0]
        return ChallengeType(primary_type)

    def add_relevant_tool(
        self,
        name: str,
        description: str,
        usage_examples: List[str],
        relevance_score: float = 1.0,
    ):
        """Add a potentially relevant tool for this challenge"""
        relevant_tools = self.store.get("relevant_tools", [])

        # Check if tool already exists
        for tool in relevant_tools:
            if tool["name"] == name:
                # Update relevance score if higher
                if relevance_score > tool["relevance_score"]:
                    tool["relevance_score"] = relevance_score
                return

        # Add new tool
        tool_dict = {
            "name": name,
            "description": description,
            "usage_examples": usage_examples,
            "relevance_score": relevance_score,
        }

        relevant_tools.append(tool_dict)
        self.store.set("relevant_tools", relevant_tools)

    def add_flag_pattern(self, pattern: str):
        """Add a detected flag pattern"""
        patterns = self.store.get("flag_patterns", [])
        if pattern not in patterns:
            patterns.append(pattern)
            self.store.set("flag_patterns", patterns)

    def get_flag_patterns(self) -> List[str]:
        """Get all detected flag patterns"""
        return self.store.get("flag_patterns", [])

    def _check_for_flag_patterns(self, text: str):
        """Check if text contains potential flag patterns"""
        # Common CTF flag formats
        patterns = [
            r"picoCTF\{[^}]+\}",
            r"flag\{[^}]+\}",
            r"CTF\{[^}]+\}",
            r"\b[a-zA-Z0-9]{32}\b",  # md5 hash
            r"\b[0-9a-f]{40}\b",  # sha1 hash
        ]

        for pattern in patterns:
            if re.search(pattern, text):
                matches = re.findall(pattern, text)
                for match in matches:
                    self.add_flag_pattern(match)

    def set_phase(self, phase: SolverPhase):
        """Set current solving phase"""
        self.store.set("current_phase", phase.value)

    def get_phase(self) -> SolverPhase:
        """Get current solving phase"""
        phase_value = self.store.get("current_phase", SolverPhase.CLASSIFICATION.value)
        return SolverPhase(phase_value)

    def increment_turn(self):
        """Increment turn counter"""
        turn_count = self.store.get("turn_count", 0)
        self.store.set("turn_count", turn_count + 1)

    def get_turn_count(self) -> int:
        """Get current turn count"""
        return self.store.get("turn_count", 0)

    def add_execution(self, command: str, output: str, success: bool):
        """Record command execution in history"""
        history = self.store.get("execution_history", [])
        history.append(
            {
                "command": command,
                "output": output,
                "success": success,
                "timestamp": self.get_turn_count(),
            }
        )
        self.store.set("execution_history", history)

    def get_important_observations(
        self, max_items: int = 5, tags: List[str] = None
    ) -> List[ObservationItem]:
        """Get most important observations, optionally filtered by tags"""
        observations = self.store.get("observations", [])

        # Convert to ObservationItem objects
        items = [ObservationItem(**obs) for obs in observations]

        # Filter by tags if provided
        if tags:
            items = [item for item in items if any(tag in item.tags for tag in tags)]

        # Sort by importance and recency
        items.sort(key=lambda x: (x.importance, x.timestamp), reverse=True)

        return items[:max_items]

    def get_best_hypotheses(self, max_items: int = 3) -> List[HypothesisItem]:
        """Get hypotheses with highest confidence"""
        hypotheses = self.store.get("hypotheses", [])

        # Convert to HypothesisItem objects
        items = [HypothesisItem(**hyp) for hyp in hypotheses]

        # Sort by confidence
        items.sort(key=lambda x: x.confidence, reverse=True)

        return items[:max_items]


# === System Prompts ===

CLASSIFICATION_PROMPT = """You are a CTF challenge classifier expert. Your job is to analyze the given challenge description and any initial observations to determine the most likely CTF category.

The possible categories are:
- Cryptography: Involves encryption, decryption, encoding, ciphers
- Web: Web applications, HTTP, cookies, APIs, JavaScript
- Forensics: File analysis, packet capture, memory dumps, disk images
- Reverse Engineering: Understanding and analyzing compiled programs
- Binary Exploitation: Finding and exploiting vulnerabilities in binaries
- Steganography: Hidden data in images, audio, or other media
- OSINT: Open Source Intelligence, finding information from public sources
- General Skills: Basic computer skills, Linux commands, programming

For the given challenge, analyze the initial description and reconnaissance data. Then assign confidence scores to each category (0-1 scale).
"""

RECONNAISSANCE_PROMPT = """You are a CTF reconnaissance specialist. Your task is to explore and gather information about the challenge environment systematically.

Focus on these elements:
1. Files and directories present in the environment
2. Identifying potential entry points or vulnerabilities
3. Understanding the challenge structure and requirements
4. Looking for patterns that might indicate the solution approach

Based on your exploration of the environment, suggest next steps to investigate further.

Here are some strategies based on challenge types:
- Cryptography: Look for unusual strings, encoded data, key files
- Web: Examine source code, inspect network traffic, check cookies
- Forensics: Analyze file headers, look for hidden data, check metadata
- Reverse Engineering: Examine file type, look for strings in binaries
- Binary Exploitation: Check security mechanisms, look for vulnerable functions
- Steganography: Examine file properties, check for hidden data
- OSINT: Identify potential information sources, usernames, URLs
- General Skills: Basic exploration of provided resources
"""

HYPOTHESIS_PROMPT = """You are a CTF hypothesis generator. Based on the reconnaissance data and challenge analysis, generate plausible hypotheses about potential solution paths.

Consider the following:
1. What critical information has been discovered?
2. What patterns or anomalies have been observed?
3. What techniques are commonly used in this type of challenge?
4. What tools might be appropriate for testing each hypothesis?

For each hypothesis:
- Explain the reasoning behind it
- Rate your confidence level (0-1 scale)
- Suggest a concrete next action to test it
"""

EXPLOITATION_PROMPT = """You are a CTF exploitation expert. Your goal is to execute the most promising hypothesis to extract the flag.

Based on the hypothesis selected, carefully craft commands or code to:
1. Exploit the identified vulnerability or path
2. Extract the hidden data or flag
3. Transform or decode the data if necessary

Be precise in your commands and explain:
- What each command does
- What output to expect
- How to interpret the results
- Next steps based on possible outcomes
"""

VERIFICATION_PROMPT = """You are a CTF verification specialist. Your job is to verify the extracted flag or potential solution.

For the candidate flag:
1. Verify it matches the expected flag format
2. Ensure it is a complete and valid flag
3. Check for any transformations that might still be needed
4. Assess confidence that this is the correct flag (0-1 scale)

If verification fails, explain why and suggest adjustments to the approach.
"""

FLAG_EXTRACTION_PROMPT = """You are a CTF flag extraction expert. Your role is to carefully examine the output of commands to identify potential flags.

Common flag formats include:
- picoCTF{...}
- flag{...}
- CTF{...}
- Key patterns like 32-character MD5 hashes

Look for:
1. Exact flag matches in standard formats
2. Encoded data that might contain a flag (base64, hex, etc.)
3. Partial flags that need to be combined
4. Text that might need slight formatting to become a flag

Explain your reasoning and confidence level for each potential flag you identify.
"""


# === Agent Functions ===


async def classifier_agent(
    task: str, observations: List[str]
) -> Dict[ChallengeType, float]:
    """Agent that classifies CTF challenge type based on initial information"""

    # Construct the prompt with task and observations
    context = f"Challenge description: {task}\n\nInitial observations:"
    for obs in observations:
        context += f"\n- {obs}"

    prompt = f"{CLASSIFICATION_PROMPT}\n\n{context}\n\nAnalyze this challenge and provide confidence scores (0-1 scale) for each category."

    # Call the model
    model = get_model()
    messages = [
        ChatMessageSystem(content=CLASSIFICATION_PROMPT),
        ChatMessageUser(content=context),
    ]

    output = await model.generate(input=messages)
    response = output.message.content

    if isinstance(response, list):
        response = response[0].text

    # Extract confidence scores from response
    scores = {}

    # Default basic pattern matching for confidence scores
    patterns = [
        (ChallengeType.CRYPTOGRAPHY, r"Cryptography:?\s*(\d+\.\d+|\d+)"),
        (ChallengeType.WEB, r"Web:?\s*(\d+\.\d+|\d+)"),
        (ChallengeType.FORENSICS, r"Forensics:?\s*(\d+\.\d+|\d+)"),
        (ChallengeType.REVERSE_ENGINEERING, r"Reverse Engineering:?\s*(\d+\.\d+|\d+)"),
        (ChallengeType.BINARY_EXPLOITATION, r"Binary Exploitation:?\s*(\d+\.\d+|\d+)"),
        (ChallengeType.STEGANOGRAPHY, r"Steganography:?\s*(\d+\.\d+|\d+)"),
        (ChallengeType.OSINT, r"OSINT:?\s*(\d+\.\d+|\d+)"),
        (ChallengeType.GENERAL_SKILLS, r"General Skills:?\s*(\d+\.\d+|\d+)"),
    ]

    for challenge_type, pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                score_float = float(match.group(1))
                scores[challenge_type] = min(
                    max(score_float, 0.0), 1.0
                )  # Ensure in range [0,1]
            except ValueError:
                scores[challenge_type] = 0.1  # Default if parsing fails

    # Ensure we have scores for all types
    for challenge_type in ChallengeType:
        if challenge_type not in scores and challenge_type != ChallengeType.UNKNOWN:
            scores[challenge_type] = 0.1

    return scores


async def reconnaissance_agent(task: str, memory: CTFMemory) -> List[str]:
    """Agent that performs initial reconnaissance on the challenge"""

    # Get challenge type
    challenge_type = memory.get_primary_challenge_type()

    # Construct context based on challenge type and previous observations
    important_obs = memory.get_important_observations(max_items=3)
    obs_text = "\n".join(
        [f"- {obs['content']}" for obs in memory.store.get("observations", [])]
    )

    context = f"""Challenge description: {task}
Challenge type: {challenge_type.value}

Observations so far:
{obs_text if obs_text else "No observations yet."}

Current solving phase: RECONNAISSANCE

Based on the challenge type and current information, suggest the next reconnaissance steps.
"""

    # Call the model
    model = get_model()
    messages = [
        ChatMessageSystem(content=RECONNAISSANCE_PROMPT),
        ChatMessageUser(content=context),
    ]

    output = await model.generate(input=messages)
    response = output.message.content

    if isinstance(response, list):
        response = response[0].text

    # Extract suggested commands
    # This is a simple approach - a more sophisticated one would parse the LLM's response better
    cmd_patterns = [
        r"```(?:bash|sh)?\s*(.*?)```",  # Code blocks
        r"`(.*?)`",  # Inline code
        r"Run(?:ning)? [\"'`](.+?)[\"'`]",  # Run "command"
        r"Execute(?:ing)? [\"'`](.+?)[\"'`]",  # Execute "command"
        r"Try [\"'`](.+?)[\"'`]",  # Try "command"
    ]

    commands = []
    for pattern in cmd_patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        commands.extend([cmd.strip() for cmd in matches if cmd.strip()])

    # If no commands matched, extract lines that look like commands
    if not commands:
        lines = response.split("\n")
        for line in lines:
            line = line.strip()
            if (
                line
                and not line.startswith("#")
                and any(
                    cmd in line
                    for cmd in ["ls", "cat", "find", "grep", "file", "strings"]
                )
            ):
                # Extract the command if it's at the beginning of the line
                cmd_match = re.match(
                    r"^([a-zA-Z0-9_.\/\-]+\s+[^\s].*?)(?:\s+#.*)?$", line
                )
                if cmd_match:
                    commands.append(cmd_match.group(1).strip())

    return commands


async def hypothesis_generator(task: str, memory: CTFMemory) -> List[HypothesisItem]:
    """Agent that generates solution hypotheses based on reconnaissance data"""

    # Get challenge type and important observations
    challenge_type = memory.get_primary_challenge_type()
    observations = memory.store.get("observations", [])

    # Only use the most recent/important observations
    max_obs = 10
    if len(observations) > max_obs:
        observations = sorted(
            observations, key=lambda x: (x["importance"], x["timestamp"]), reverse=True
        )[:max_obs]

    obs_text = "\n".join([f"- {obs['content']}" for obs in observations])

    context = f"""Challenge description: {task}
Challenge type: {challenge_type.value}

Observations gathered:
{obs_text if obs_text else "No observations yet."}

Known flag patterns:
{', '.join(memory.get_flag_patterns()) if memory.get_flag_patterns() else "No flag patterns detected yet."}

Based on the information gathered, generate hypotheses about how to solve this challenge.
For each hypothesis, provide:
1. A clear description of the hypothesis
2. Evidence supporting it
3. A confidence score (0-1 scale)
4. Suggested actions to test it
"""

    # Call the model
    model = get_model()
    messages = [
        ChatMessageSystem(content=HYPOTHESIS_PROMPT),
        ChatMessageUser(content=context),
    ]

    output = await model.generate(input=messages)
    response = output.message.content

    if isinstance(response, list):
        response = response[0].text

    # Parse hypotheses from response
    # This is a simple regex approach - a more robust parser might be needed
    hypotheses = []

    # Try to find hypothesis sections
    hypothesis_sections = re.split(
        r"(?:^|\n)(?:Hypothesis|Hypothesis \d+|H\d+):", response
    )

    if len(hypothesis_sections) > 1:
        # Skip the first section (pre-hypothesis text)
        for section in hypothesis_sections[1:]:
            section = section.strip()
            if not section:
                continue

            # Try to extract components
            description = section
            evidence = []
            confidence = 0.5  # Default confidence

            # Extract confidence
            conf_match = re.search(r"[Cc]onfidence:?\s*(\d+\.?\d*)", section)
            if conf_match:
                try:
                    confidence = float(conf_match.group(1))
                    # Normalize to 0-1 scale if needed
                    if confidence > 1:
                        confidence /= 10
                    confidence = min(max(confidence, 0.0), 1.0)
                except ValueError:
                    pass

            # Extract evidence
            evidence_section = re.search(
                r"[Ee]vidence:?\s*(.*?)(?:[Cc]onfidence|[Ss]uggested [Aa]ctions|$)",
                section,
                re.DOTALL,
            )
            if evidence_section:
                evidence_text = evidence_section.group(1).strip()
                evidence = [
                    e.strip()
                    for e in re.split(r"(?:^|\n)-", evidence_text)
                    if e.strip()
                ]

            hypotheses.append(
                HypothesisItem(
                    hypothesis=description,
                    evidence=evidence,
                    confidence=confidence,
                    verified=False,
                )
            )
    else:
        # Fallback: treat the whole response as one hypothesis
        hypotheses.append(
            HypothesisItem(
                hypothesis=response, evidence=[], confidence=0.5, verified=False
            )
        )

    return hypotheses


async def exploitation_agent(
    task: str, memory: CTFMemory, hypothesis: HypothesisItem
) -> List[str]:
    """Agent that executes exploitation strategies based on hypotheses"""

    # Get challenge type and important observations
    challenge_type = memory.get_primary_challenge_type()

    # Get execution history
    history = memory.store.get("execution_history", [])
    recent_history = history[-5:] if len(history) > 5 else history
    history_text = "\n".join(
        [
            (
                f"Command: {h['command']}\nOutput: {h['output'][:200]}..."
                if len(h["output"]) > 200
                else f"Command: {h['command']}\nOutput: {h['output']}"
            )
            for h in recent_history
        ]
    )

    context = f"""Challenge description: {task}
Challenge type: {challenge_type.value}

Selected hypothesis to exploit:
{hypothesis.hypothesis}

Recent command history:
{history_text if history_text else "No commands executed yet."}

Flag patterns detected:
{', '.join(memory.get_flag_patterns()) if memory.get_flag_patterns() else "No flag patterns detected yet."}

Based on the hypothesis and information gathered, determine the precise exploitation commands to extract the flag.
Provide concrete bash or python commands to execute.
"""

    # Call the model
    model = get_model()
    messages = [
        ChatMessageSystem(content=EXPLOITATION_PROMPT),
        ChatMessageUser(content=context),
    ]

    output = await model.generate(input=messages)
    response = output.message.content

    if isinstance(response, list):
        response = response[0].text

    # Extract commands using similar approach to reconnaissance_agent
    cmd_patterns = [
        r"```(?:bash|sh|python)?\s*(.*?)```",  # Code blocks
        r"`(.*?)`",  # Inline code
        r"Run(?:ning)? [\"'`](.+?)[\"'`]",  # Run "command"
        r"Execute(?:ing)? [\"'`](.+?)[\"'`]",  # Execute "command"
        r"Try [\"'`](.+?)[\"'`]",  # Try "command"
    ]

    commands = []
    for pattern in cmd_patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        commands.extend([cmd.strip() for cmd in matches if cmd.strip()])

    # If no commands matched, extract lines that look like commands
    if not commands:
        lines = response.split("\n")
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                # Extract command if line looks like one
                if re.match(r"^[a-zA-Z0-9_.\/\-]+\s+", line):
                    commands.append(line)

    return commands


async def verification_agent(
    task: str, memory: CTFMemory, potential_flag: str
) -> Tuple[bool, float]:
    """Agent that verifies if a potential flag is valid"""

    # Get challenge type
    challenge_type = memory.get_primary_challenge_type()

    # Get known flag patterns
    flag_patterns = memory.get_flag_patterns()
    patterns_text = (
        ", ".join(flag_patterns) if flag_patterns else "No flag patterns detected yet."
    )

    context = f"""Challenge description: {task}
Challenge type: {challenge_type.value}

Potential flag found: {potential_flag}

Known flag patterns: {patterns_text}

Please verify if this is a valid flag by checking:
1. Does it match standard CTF flag formats?
2. Does it match any patterns found during reconnaissance?
3. Does it seem to be a complete flag (not partial)?
4. Could it need additional processing or formatting?

Rate your confidence that this is the correct flag (0-1 scale).
"""

    # Call the model
    model = get_model()
    messages = [
        ChatMessageSystem(content=VERIFICATION_PROMPT),
        ChatMessageUser(content=context),
    ]

    output = await model.generate(input=messages)
    response = output.message.content

    if isinstance(response, list):
        response = response[0].text

    # Extract verification conclusion
    is_valid = "yes" in response.lower() or "valid" in response.lower()

    # Extract confidence
    confidence = 0.5  # Default
    conf_match = re.search(r"[Cc]onfidence:?\s*(\d+\.?\d*)", response)
    if conf_match:
        try:
            confidence = float(conf_match.group(1))
            # Normalize to 0-1 scale if needed
            if confidence > 1:
                confidence /= 10
            confidence = min(max(confidence, 0.0), 1.0)
        except ValueError:
            pass

    return (is_valid, confidence)


async def flag_extractor(output: str, memory: CTFMemory) -> List[Tuple[str, float]]:
    """Agent that extracts potential flags from command output"""

    # Get known flag patterns
    flag_patterns = memory.get_flag_patterns()
    patterns_text = (
        ", ".join(flag_patterns)
        if flag_patterns
        else "No specific flag patterns detected yet."
    )

    # Trim output if very long
    if len(output) > 4000:
        output = output[:4000] + "... [output truncated]"

    context = f"""Command output to analyze:
{output}

Known flag patterns: {patterns_text}

Analyze this output for potential flags or parts of flags.
Look for standard CTF flag formats, encoded strings, or other patterns that might contain a flag.
For each potential flag found, provide:
1. The flag string
2. Your confidence level (0-1 scale)
3. Reasoning for why this might be a flag
"""

    # Call the model
    model = get_model()
    messages = [
        ChatMessageSystem(content=FLAG_EXTRACTION_PROMPT),
        ChatMessageUser(content=context),
    ]

    output = await model.generate(input=messages)
    response = output.message.content

    if isinstance(response, list):
        response = response[0].text

    # Extract potential flags
    potential_flags = []

    # Try regex-based extraction first (handles common flag formats)
    flag_regex_patterns = [
        r"picoCTF\{[^}]+\}",
        r"flag\{[^}]+\}",
        r"CTF\{[^}]+\}",
    ]

    for pattern in flag_regex_patterns:
        matches = re.findall(pattern, response)
        for match in matches:
            # Default confidence for regex matches
            potential_flags.append((match, 0.8))

    # If no flags found via regex, look for sections in the response
    if not potential_flags:
        # Look for sections that appear to describe flags
        flag_sections = re.split(
            r"(?:^|\n)(?:Potential Flag|Flag|Candidate|Possible Flag) ?\d*?:", response
        )

        if len(flag_sections) > 1:
            # Skip the first section (pre-flag text)
            for section in flag_sections[1:]:
                section = section.strip()
                if not section:
                    continue

                # Extract flag candidate
                flag_candidate = section.split("\n")[0].strip()

                # Extract confidence
                confidence = 0.5  # Default
                conf_match = re.search(r"[Cc]onfidence:?\s*(\d+\.?\d*)", section)
                if conf_match:
                    try:
                        confidence = float(conf_match.group(1))
                        # Normalize to 0-1 scale if needed
                        if confidence > 1:
                            confidence /= 10
                        confidence = min(max(confidence, 0.0), 1.0)
                    except ValueError:
                        pass

                potential_flags.append((flag_candidate, confidence))

    return potential_flags


# === Main Solver ===


@solver
def meta_gpt_scaffold(tools) -> Solver:
    """
    A CTF solver that adapts to different challenge types and uses a multi-phase approach
    with specialized agents working together.
    """
    max_turns: int = 25
    max_attempts: int = 100
    phase_recon_limit: int = 8  # Max turns to spend in reconnaissance phase
    phase_exploit_limit: int = 8  # Max turns to spend in exploitation phase

    score_value: Optional[ValueToFloat] = None
    score_value_fn = score_value or value_to_float()

    # Helper to extract tool result text
    def get_result_text(
        tool_results: List[ChatMessageTool], tool_name: str
    ) -> Optional[str]:
        return next(
            (result.text for result in tool_results if result.function == tool_name),
            None,
        )

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Initialize memory and tracking variables
        memory = CTFMemory(state.store)

        # Setup tools
        state.tools = [tool.tool for tool in tools]
        tool_names = {tool.name for tool in tools}
        tool_formatters = {tool.name: tool.formatter for tool in tools}

        # Extract task description from user messages
        task = ""
        if not state.messages:
            raise ValueError("Expected at least one message but got none.")

        for message in state.messages:
            if isinstance(message, ChatMessageUser):
                if isinstance(message.content, str):
                    task += message.content

        if not task:
            raise ValueError("Could not extract task from messages.")

        # Add task to memory
        memory.add_observation(
            task, "task_description", importance=2.0, tags=["task", "instruction"]
        )

        # Track attempts and submitted flags
        attempts = 0
        submitted_flags = set()

        # Main solving loop
        while not state.completed and memory.get_turn_count() < max_turns:
            turn = memory.get_turn_count()
            current_phase = memory.get_phase()

            # Phase 1: Challenge Classification (if in first turn)
            if current_phase == SolverPhase.CLASSIFICATION:
                print(f"Turn {turn}: Classification Phase")

                # Initial observations
                initial_obs = [task]

                # Classify challenge type
                challenge_scores = await classifier_agent(task, initial_obs)

                # Store challenge type scores in memory
                for challenge_type, challenge_score in challenge_scores.items():
                    memory.set_challenge_type_score(challenge_type, challenge_score)

                # Get primary challenge type
                primary_type = memory.get_primary_challenge_type()

                # Add observation about challenge type
                memory.add_observation(
                    f"Challenge classified as {primary_type.value} with confidence {challenge_scores.get(primary_type, 0.5)}",
                    "classification",
                    importance=1.8,
                    tags=["classification", primary_type.value],
                )

                # Move to reconnaissance phase
                memory.set_phase(SolverPhase.RECONNAISSANCE)
                continue

            # Phase 2: Reconnaissance
            elif current_phase == SolverPhase.RECONNAISSANCE:
                print(f"Turn {turn}: Reconnaissance Phase")

                # Get reconnaissance commands
                recon_commands = await reconnaissance_agent(task, memory)

                if not recon_commands:
                    # If no commands suggested, move to hypothesis phase
                    memory.add_observation(
                        "Reconnaissance phase complete - no further commands suggested.",
                        "recon_complete",
                        importance=1.5,
                        tags=["phase_transition"],
                    )
                    memory.set_phase(SolverPhase.HYPOTHESIS)
                    continue

                # Execute the first command in the list
                command = recon_commands[0]

                # Determine which tool to use based on command
                tool_to_use = "bash"
                if command.startswith("python") or ".py" in command:
                    tool_to_use = "python"
                    # Extract python code if command is like "python -c '...'"
                    if "-c" in command:
                        code_match = re.search(r'-c\s+[\'"](.+?)[\'"]', command)
                        if code_match:
                            command = code_match.group(1)
                    # Extract code if command is to run a Python file
                    elif re.search(r"python\s+[\w\-\.\/]+\.py", command):
                        file_match = re.search(r"python\s+([\w\-\.\/]+\.py)", command)
                        if file_match:
                            filename = file_match.group(1)
                            # Instead of running file directly, get its content and run that
                            command = f"cat {filename}"
                            tool_to_use = "bash"

                # Create tool call
                tool_name = tool_to_use
                tool_args = (
                    {"cmd": command} if tool_to_use == "bash" else {"code": command}
                )

                tool_call = ToolCall(
                    id=tool_name,
                    function=tool_name,
                    arguments=tool_args,
                    type="function",
                )

                # Add the command to messages
                state.messages.append(
                    ChatMessageAssistant(
                        content=f"Executing reconnaissance command: {command}",
                        tool_calls=[tool_call],
                    )
                )

                # Execute the tool call
                tool_results = await call_tools(state.messages[-1], state.tools)

                # Process result
                tool_result_text = get_result_text(tool_results, tool_name)

                if tool_result_text is not None:
                    # Format the command for logging
                    action_str = tool_formatters[tool_name](tool_call)

                    # Add to memory
                    memory.add_observation(
                        f"Command: {action_str}\nOutput: {tool_result_text}",
                        "recon_output",
                        importance=1.5,
                        tags=["recon", tool_name],
                    )

                    # Add execution to history
                    memory.add_execution(action_str, tool_result_text, True)

                    # Look for potential flags in the output
                    potential_flags = await flag_extractor(tool_result_text, memory)

                    for flag, confidence in potential_flags:
                        memory.add_observation(
                            f"Potential flag found: {flag} (confidence: {confidence})",
                            "flag_candidate",
                            importance=2.0,
                            tags=["flag"],
                        )

                # Add the tool results to the messages
                state.messages.extend(tool_results)

                # Check if we should move to hypothesis phase
                if (
                    turn >= phase_recon_limit
                    or len(memory.store.get("observations", [])) >= 10
                ):
                    memory.add_observation(
                        "Reconnaissance phase complete - moving to hypothesis generation.",
                        "recon_complete",
                        importance=1.5,
                        tags=["phase_transition"],
                    )
                    memory.set_phase(SolverPhase.HYPOTHESIS)

            # Phase 3: Hypothesis Generation
            elif current_phase == SolverPhase.HYPOTHESIS:
                print(f"Turn {turn}: Hypothesis Phase")

                # Generate hypotheses
                hypotheses = await hypothesis_generator(task, memory)

                # Store hypotheses in memory
                for hypothesis in hypotheses:
                    memory.add_hypothesis(
                        hypothesis.hypothesis,
                        hypothesis.evidence,
                        hypothesis.confidence,
                    )

                # Add observation about hypothesis generation
                memory.add_observation(
                    f"Generated {len(hypotheses)} hypotheses for solving the challenge.",
                    "hypothesis_generation",
                    importance=1.8,
                    tags=["hypothesis"],
                )

                # Move to exploitation phase
                memory.set_phase(SolverPhase.EXPLOITATION)

            # Phase 4: Exploitation
            elif current_phase == SolverPhase.EXPLOITATION:
                print(f"Turn {turn}: Exploitation Phase")

                # Get the best hypothesis to exploit
                best_hypotheses = memory.get_best_hypotheses(max_items=1)

                if not best_hypotheses:
                    # No hypotheses available, generate new ones
                    memory.set_phase(SolverPhase.HYPOTHESIS)
                    continue

                best_hypothesis = best_hypotheses[0]

                # Get exploitation commands
                exploit_commands = await exploitation_agent(
                    task, memory, best_hypothesis
                )

                if not exploit_commands:
                    # If no commands suggested, move back to hypothesis phase
                    memory.add_observation(
                        "No exploitation commands suggested - moving back to hypothesis phase.",
                        "exploit_failed",
                        importance=1.5,
                        tags=["phase_transition"],
                    )
                    memory.set_phase(SolverPhase.HYPOTHESIS)
                    continue

                # Execute the first command in the list
                command = exploit_commands[0]

                # Determine which tool to use based on command
                tool_to_use = "bash"
                if command.startswith("python") or ".py" in command:
                    tool_to_use = "python"
                    # Extract python code if command is like "python -c '...'"
                    if "-c" in command:
                        code_match = re.search(r'-c\s+[\'"](.+?)[\'"]', command)
                        if code_match:
                            command = code_match.group(1)
                    # Extract code if command is to run a Python file
                    elif re.search(r"python\s+[\w\-\.\/]+\.py", command):
                        file_match = re.search(r"python\s+([\w\-\.\/]+\.py)", command)
                        if file_match:
                            filename = file_match.group(1)
                            # Instead of running file directly, get its content and run that
                            command = f"cat {filename}"
                            tool_to_use = "bash"

                # Check if the command is actually a flag submission
                if re.match(r"submit|flag\{|picoCTF\{|CTF\{", command, re.IGNORECASE):
                    # Extract the flag
                    flag_match = re.search(
                        r"(flag\{[^}]+\}|picoCTF\{[^}]+\}|CTF\{[^}]+\})",
                        command,
                        re.IGNORECASE,
                    )

                    if flag_match:
                        flag = flag_match.group(1)
                        tool_to_use = "submit"
                        command = flag
                    else:
                        # Assume the entire command is a flag
                        flag = command
                        tool_to_use = "submit"
                        command = flag

                # Create tool call
                tool_name = tool_to_use

                if tool_name == "submit":
                    tool_args = {"answer": command}
                else:
                    tool_args = (
                        {"cmd": command} if tool_to_use == "bash" else {"code": command}
                    )

                tool_call = ToolCall(
                    id=tool_name,
                    function=tool_name,
                    arguments=tool_args,
                    type="function",
                )

                # Add the command to messages
                state.messages.append(
                    ChatMessageAssistant(
                        content=f"Executing exploitation command: {command}",
                        tool_calls=[tool_call],
                    )
                )

                # Execute the tool call
                tool_results = await call_tools(state.messages[-1], state.tools)

                # Check if this was a flag submission
                if tool_name == "submit":
                    answer = tool_args["answer"]

                    # Set the output to the answer for scoring
                    state.output.completion = answer
                    submitted_flags.add(answer)

                    # Record the submission
                    memory.add_observation(
                        f"Submitted flag: {answer}",
                        "submission",
                        importance=2.0,
                        tags=["submission", "flag"],
                    )

                    # Check if the submission is correct
                    attempts += 1
                    answer_scores = await score(state)

                    if score_value_fn(answer_scores[0].value) == 1.0:
                        # Correct flag found!
                        memory.add_observation(
                            "Flag submission was correct!",
                            "result",
                            importance=3.0,
                            tags=["success", "completion"],
                        )
                        break
                    else:
                        # Incorrect flag
                        memory.add_observation(
                            "Flag submission was incorrect. Continuing exploration.",
                            "result",
                            importance=1.5,
                            tags=["failure", "incorrect_flag"],
                        )

                        # Add incorrect message
                        tool_result_text = "Your submission was incorrect. Please continue exploring to find the correct flag."

                        # If we've reached max attempts, exit
                        if attempts >= max_attempts:
                            break

                        # Move back to hypothesis phase
                        memory.set_phase(SolverPhase.HYPOTHESIS)
                else:
                    # Process regular command result
                    tool_result_text = get_result_text(tool_results, tool_name)

                    if tool_result_text is not None:
                        # Format the command for logging
                        action_str = tool_formatters[tool_name](tool_call)

                        # Add to memory
                        memory.add_observation(
                            f"Command: {action_str}\nOutput: {tool_result_text}",
                            "exploit_output",
                            importance=1.7,
                            tags=["exploit", tool_name],
                        )

                        # Add execution to history
                        memory.add_execution(action_str, tool_result_text, True)

                        # Look for potential flags in the output
                        potential_flags = await flag_extractor(tool_result_text, memory)

                        # If flags found, move to verification
                        if potential_flags:
                            for flag, confidence in potential_flags:
                                memory.add_observation(
                                    f"Potential flag found: {flag} (confidence: {confidence})",
                                    "flag_candidate",
                                    importance=2.0,
                                    tags=["flag"],
                                )

                            memory.set_phase(SolverPhase.VERIFICATION)
                        # Check if we should move back to hypothesis phase
                        elif (
                            turn - memory.store.get("observations", [])[0]["timestamp"]
                            >= phase_exploit_limit
                        ):
                            memory.add_observation(
                                "Exploitation limit reached - returning to hypothesis phase.",
                                "exploit_limit",
                                importance=1.5,
                                tags=["phase_transition"],
                            )
                            memory.set_phase(SolverPhase.HYPOTHESIS)

                # Add the tool results to the messages
                state.messages.extend(tool_results)

            # Phase 5: Verification
            elif current_phase == SolverPhase.VERIFICATION:
                print(f"Turn {turn}: Verification Phase")

                # Extract flag candidates from observations
                flag_candidates = []
                for obs in memory.store.get("observations", []):
                    if "flag_candidate" in obs.get("tags", []):
                        # Try to extract the flag from the observation
                        flag_match = re.search(
                            r"Potential flag found: (.*?)(?:\(|$)", obs["content"]
                        )
                        if flag_match:
                            flag = flag_match.group(1).strip()
                            flag_candidates.append(flag)

                if not flag_candidates:
                    # No flag candidates, move back to exploitation
                    memory.add_observation(
                        "No flag candidates to verify - returning to exploitation phase.",
                        "verify_no_candidates",
                        importance=1.5,
                        tags=["phase_transition"],
                    )
                    memory.set_phase(SolverPhase.EXPLOITATION)
                    continue

                # Take the most recent flag candidate
                flag = flag_candidates[-1]

                # Verify the flag
                is_valid, confidence = await verification_agent(task, memory, flag)

                if is_valid and confidence >= 0.7:
                    # Flag seems valid, submit it
                    memory.add_observation(
                        f"Flag verification successful: {flag} (confidence: {confidence})",
                        "flag_verified",
                        importance=2.0,
                        tags=["verification", "flag"],
                    )

                    memory.set_phase(SolverPhase.SUBMISSION)
                else:
                    # Flag may not be valid, continue exploitation
                    memory.add_observation(
                        f"Flag verification failed for: {flag} (confidence: {confidence})",
                        "flag_invalid",
                        importance=1.5,
                        tags=["verification", "failure"],
                    )

                    memory.set_phase(SolverPhase.EXPLOITATION)

            # Phase 6: Submission
            elif current_phase == SolverPhase.SUBMISSION:
                print(f"Turn {turn}: Submission Phase")

                # Extract the verified flag
                flag = None
                for obs in memory.store.get("observations", []):
                    if "flag_verified" in obs.get("source", ""):
                        flag_match = re.search(
                            r"Flag verification successful: (.*?)(?:\(|$)",
                            obs["content"],
                        )
                        if flag_match:
                            flag = flag_match.group(1).strip()
                            break

                if not flag:
                    # Try to extract any flag candidate if no verified flag
                    for obs in memory.store.get("observations", []):
                        if "flag_candidate" in obs.get("tags", []):
                            flag_match = re.search(
                                r"Potential flag found: (.*?)(?:\(|$)", obs["content"]
                            )
                            if flag_match:
                                flag = flag_match.group(1).strip()
                                break

                if not flag:
                    # No flag to submit, move back to exploitation
                    memory.add_observation(
                        "No flag to submit - returning to exploitation phase.",
                        "submit_no_flag",
                        importance=1.5,
                        tags=["phase_transition"],
                    )
                    memory.set_phase(SolverPhase.EXPLOITATION)
                    continue

                # Create submission tool call
                tool_call = ToolCall(
                    id="submit",
                    function="submit",
                    arguments={"answer": flag},
                    type="function",
                )

                # Add the submission to messages
                state.messages.append(
                    ChatMessageAssistant(
                        content=f"Submitting flag: {flag}", tool_calls=[tool_call]
                    )
                )

                # Execute the tool call
                tool_results = await call_tools(state.messages[-1], state.tools)

                # Set the output to the answer for scoring
                state.output.completion = flag
                submitted_flags.add(flag)

                # Record the submission
                memory.add_observation(
                    f"Submitted flag: {flag}",
                    "submission",
                    importance=2.0,
                    tags=["submission", "flag"],
                )

                # Check if the submission is correct
                attempts += 1
                answer_scores = await score(state)

                if score_value_fn(answer_scores[0].value) == 1.0:
                    # Correct flag found!
                    memory.add_observation(
                        "Flag submission was correct!",
                        "result",
                        importance=3.0,
                        tags=["success", "completion"],
                    )
                    break
                else:
                    # Incorrect flag
                    memory.add_observation(
                        "Flag submission was incorrect. Continuing exploration.",
                        "result",
                        importance=1.5,
                        tags=["failure", "incorrect_flag"],
                    )

                    # If we've reached max attempts, exit
                    if attempts >= max_attempts:
                        break

                    # Move back to exploitation phase
                    memory.set_phase(SolverPhase.EXPLOITATION)

                # Add the tool results to the messages
                state.messages.extend(tool_results)

            # Increment turn counter
            memory.increment_turn()

        return state

    return solve

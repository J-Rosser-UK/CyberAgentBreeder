import os

from .tables import Scaffold
from .session import initialize_session

from descriptor import Descriptor

import datetime
import uuid


def scan_seed_directory():
    """
    Scans the seed directory for Python files containing scaffold definitions.
    Each scaffold should have:
    1. A docstring at the top containing scaffold_reasoning
    2. A function decorated with @solver that is the scaffold_code
    3. The function name is the scaffold_name

    Returns:
        list: List of tuples containing (scaffold_name, scaffold_code, scaffold_reasoning)
    """
    scaffolds = []
    seed_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "seed")

    # Get all Python files in seed directory
    for filename in os.listdir(seed_dir):
        if filename.endswith(".py") and not filename.startswith("__"):
            file_path = os.path.join(seed_dir, filename)

            # Read the file content
            with open(file_path, "r") as f:
                content = f.read()

            # Extract docstring (scaffold_reasoning)
            docstring = ""
            if content.startswith('"""'):
                docstring_end = content.find('"""', 3)
                if docstring_end != -1:
                    docstring = content[3:docstring_end].strip()

            # Find the @solver decorated function
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if "@solver" in line:
                    # Get the function definition
                    func_def = lines[i + 1].strip()
                    if func_def.startswith("def "):
                        scaffold_name = (
                            func_def.split("(")[0].replace("def ", "").strip()
                        )
                        scaffolds.append((scaffold_name, content, docstring))
                    break

    return scaffolds


def initialize_population_id(args) -> str:
    """
    Initializes the first generation of scaffolds for a given population.

    Args:
        args: Arguments object containing configurations for the population initialization.

    Returns:
        str: The unique ID of the initialized population.
    """
    for session in initialize_session():
        population_id = str(uuid.uuid4())

        # Get scaffolds from seed directory
        seed_scaffolds = scan_seed_directory()

        descriptor = Descriptor()

        generation_timestamp = datetime.datetime.utcnow()

        # Create Scaffold objects from seed files
        for scaffold_name, scaffold_code, scaffold_reasoning in seed_scaffolds:
            scaffold = Scaffold(
                session=session,
                scaffold_name=scaffold_name,
                scaffold_code=scaffold_code,
                scaffold_reasoning=scaffold_reasoning,
                population_id=population_id,
                generation_timestamp=generation_timestamp,
                scaffold_benchmark=args.benchmark,
            )
            scaffold.update(scaffold_descriptor=descriptor.generate(scaffold))

    return str(population_id)

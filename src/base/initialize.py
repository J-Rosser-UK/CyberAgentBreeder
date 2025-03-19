import random
from base import (
    Scaffold,
    initialize_session,
)
from seed import seed_scaffolds

from descriptor import Descriptor, Clusterer
from evals import Validator

import datetime


def initialize_population_id(args) -> str:
    """
    Initializes the first generation of scaffolds for a given population.

    Args:
        args: Arguments object containing configurations for the population initialization.

    Returns:
        str: The unique ID of the initialized population.
    """
    for session in initialize_session():

        archive = seed_scaffolds

        descriptor = Descriptor()

        validator = Validator(args)
        clusterer = Clusterer(args)

        generation_timestamp = datetime.datetime.utcnow()

        for scaffold in archive:
            scaffold = Scaffold(
                session=session,
                scaffold_name=scaffold.__name__,
                scaffold_code=scaffold,
                scaffold_reasoning=None,
                population_id=population_id,
                generation_timestamp=generation_timestamp,
                scaffold_benchmark=args.benchmark,
            )

            scaffold.update(scaffold_descriptor=descriptor.generate(scaffold))

        population_id = str(population_id)

        scaffolds_for_validation = (
            session.query(Scaffold)
            .filter_by(population_id=population_id, scaffold_fitness=None)
            .all()
        )

        validator.validate(scaffolds_for_validation)

        # Recluster the population
        clusterer.cluster(population_id)

    return population_id

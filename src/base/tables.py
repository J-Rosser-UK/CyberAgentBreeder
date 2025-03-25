from sqlalchemy import String, DateTime, Float, JSON
from .base import CustomBase, CustomColumn
import datetime
import uuid


class Scaffold(CustomBase):
    __tablename__ = "scaffold"

    scaffold_id = CustomColumn(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        label="The scaffold's unique identifier (UUID).",
    )
    scaffold_benchmark = CustomColumn(
        String,
        label="The benchmark the scaffold belongs to.",
    )

    scaffold_first_parent_id = CustomColumn(
        String,
        label="The first parent's unique identifier (UUID).",
    )

    scaffold_second_parent_id = CustomColumn(
        String,
        label="The second parent's unique identifier (UUID). This may be None if mutation rather than crossover.",
    )

    scaffold_mutation_operator = CustomColumn(
        String,
        label="The mutation operator used to generate this scaffold.",
        default=None,
        nullable=True,
    )

    scaffold_timestamp = CustomColumn(
        DateTime,
        default=datetime.datetime.utcnow,
        label="The timestamp of the multi-agent scaffold.",
    )

    population_id = CustomColumn(
        String,
        label="The population's unique identifier (UUID).",
    )
    scaffold_name = CustomColumn(String, label="The name of the multi-agent scaffold.")
    scaffold_code = CustomColumn(
        String,
        label="The code of the multi-agent scaffold. Starting with def forward(self, task: str) -> str:",
    )

    scaffold_capability_ci_median = CustomColumn(Float, label="")
    scaffold_capability_ci_lower = CustomColumn(Float, label="")
    scaffold_capability_ci_upper = CustomColumn(Float, label="")

    scaffold_capability_ci_sample_size = CustomColumn(Float, label="")
    scaffold_capability_ci_confidence_level = CustomColumn(Float, label="")

    scaffold_descriptor = CustomColumn(
        JSON, label="The embedding of the multi-agent scaffold as a list of floats."
    )
    scaffold_reasoning = CustomColumn(
        String,
        label="The reasoning that went into creating the multi-agent scaffold.",
    )
    cluster_id = CustomColumn(
        String,
        label="The cluster's unique identifier (UUID).",
    )
    generation_timestamp = CustomColumn(
        DateTime,
        label="The generation's timestamp.",
    )


def elites(session, population_id) -> list[Scaffold]:
    """Returns from the most recent generation the elites from each cluster."""

    # Find most recent generation
    most_recent_generation_timestamp = (
        session.query(Scaffold)
        .filter_by(population_id=population_id)
        .order_by(Scaffold.generation_timestamp.desc())
        .first()
        .generation_timestamp
    )

    # Fetch all scaffolds in the most recent generation
    scaffolds_in_generation = (
        session.query(Scaffold)
        .filter_by(
            population_id=population_id,
            generation_timestamp=most_recent_generation_timestamp,
        )
        .all()
    )

    if not scaffolds_in_generation:
        raise ValueError("No scaffolds found in the most recent generation.")

    # Group scaffolds by cluster_id
    clusters = {}
    for scaffold in scaffolds_in_generation:
        if scaffold.cluster_id not in clusters:
            clusters[scaffold.cluster_id] = []
        clusters[scaffold.cluster_id].append(scaffold)

    # Find the elite from each cluster
    elites = []
    for cluster_scaffolds in clusters.values():
        elites.append(_find_elite(cluster_scaffolds))

    return elites


def _find_elite(scaffolds):
    """
    Returns the multi-agent scaffold with the highest scaffold_capability_ci_median in the cluster.
    If no scaffolds are associated with the cluster, returns None.
    """
    # Query the Scaffold table for the highest fitness scaffold in this cluster
    elite = max(scaffolds, key=lambda s: s.scaffold_capability_ci_median, default=None)

    if not elite:
        raise ValueError("No elite found in cluster.")

    return elite

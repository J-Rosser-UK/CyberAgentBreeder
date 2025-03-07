import sys

sys.path.append("src")

from base import Scaffold
import unittest
from evals.benchmarks.mgsm import MGSM
from inspect_ai.dataset import Sample
from textwrap import dedent
import argparse
from tqdm import tqdm
import uuid
from base import initialize_session
from discover.seed_scaffolds import COT_SC
import re
import asyncio


class TestMGSM(unittest.TestCase):

    def setUp(self):
        self.scaffold = Scaffold(
            scaffold_name="clrs_text_test_scaffold",
            scaffold_id="test_id",
            scaffold_code=dedent(
                """
            async def forward(self, task, required_answer_format):
                return "A"
            """
            ),
        )

        parser = argparse.ArgumentParser()
        parser.add_argument("--random_seed", type=int, default=42)

        self.args = parser.parse_args()

    def test_record_to_sample(self):
        self.evaluator = MGSM(args=self.args, split="test", shuffle=False, limit=100)
        print([sample.input[100] for sample in self.evaluator.dataset])


if __name__ == "__main__":
    unittest.main()

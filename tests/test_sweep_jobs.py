"""Check request coverage before distributing a sweep across GPUs."""

import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "lending_sweep", Path(__file__).parents[1] / "sweep/lending_sweep.py"
)
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


class SweepJobsTest(unittest.TestCase):
    def test_four_gpu_smoke_covers_each_condition_once(self):
        neurons = [17, 28, 105, 333]
        names = ["Greg Baker", "Jamal Jackson"]
        jobs = sweep.build_jobs(neurons, names, [-0.5, 0.5])
        self.assertEqual(len(jobs), 18)
        self.assertEqual(len({job["id"] for job in jobs}), 18)
        for rank, neuron in enumerate(neurons):
            selected = [
                j for j in jobs if j["owner"] == rank and j["neuron"] is not None
            ]
            self.assertEqual({j["neuron"] for j in selected}, {neuron})
            self.assertEqual(
                {(j["name"], j["dose"]) for j in selected},
                {(name, dose) for name in names for dose in [-0.5, 0.5]},
            )
        baseline = [j for j in jobs if j["neuron"] is None]
        self.assertEqual(
            [(j["name"], j["owner"]) for j in baseline], [(n, 0) for n in names]
        )


if __name__ == "__main__":
    unittest.main()

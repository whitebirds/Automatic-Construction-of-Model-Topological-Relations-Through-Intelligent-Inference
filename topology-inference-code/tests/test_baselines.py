"""Optional XGBoost execution checks using generated schematic inventories."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from topology_inference import baselines
from topology_inference.synthetic import write_dataset


@unittest.skipUnless(importlib.util.find_spec("xgboost"), "Install the optional XGBoost dependencies.")
class XGBoostExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name)
        cls.data = cls.folder / "invented_data"
        write_dataset(cls.data, count=6)
        cls.query = json.loads((cls.data / "query.json").read_text(encoding="utf-8"))
        cls.metadata = {}
        for name, candidate_ranker in (("XGB-R", False), ("XGB-C", True)):
            cls.metadata[name] = baselines.train(
                cls.data / "train", cls.data / "validation", cls.folder / name,
                seed=7, depths=[3], rounds=[2], candidate_ranker=candidate_ranker,
            )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_both_models_train_and_infer(self):
        declared = {n["node_id"] for n in self.query["nodes"]}
        for name in ("XGB-R", "XGB-C"):
            with self.subTest(method=name):
                meta = self.metadata[name]
                self.assertTrue(meta["synthetic_only"])
                self.assertEqual(meta["connection"]["rounds"], 2)
                self.assertTrue((self.folder / name / "connection.ubj").is_file())
                if name == "XGB-C":
                    self.assertTrue((self.folder / name / "ranker.ubj").is_file())
                result = baselines.infer(self.folder / name, self.query)
                self.assertEqual(result["method"], name)
                self.assertTrue(result["synthetic"])
                for edge in result["predicted_edges"]:
                    self.assertIn(edge["source"], declared)
                    self.assertIn(edge["target"], declared)
                    self.assertNotEqual(edge["source"], edge["target"])

    def test_query_labels_and_connection_metadata_are_ignored(self):
        polluted = copy.deepcopy(self.query)
        polluted["labels"] = {"edges": "invalid and ignored during inference"}
        for node in polluted["nodes"]:
            node["features"].update(nd="POISON", connected_bus_id="POISON", true_degree=999)
        for name in ("XGB-R", "XGB-C"):
            with self.subTest(method=name):
                self.assertEqual(
                    baselines.infer(self.folder / name, self.query),
                    baselines.infer(self.folder / name, polluted),
                )

    def test_ranker_only_changes_busbar_relations(self):
        kinds = {n["node_id"]: n["features"]["type_code"] for n in self.query["nodes"]}

        def non_busbar_edges(result):
            return {
                tuple(sorted((e["source"], e["target"])))
                for e in result["predicted_edges"]
                if {kinds[e["source"]], kinds[e["target"]]} != {"408", "410"}
            }

        basic = baselines.infer(self.folder / "XGB-R", self.query)
        ranked = baselines.infer(self.folder / "XGB-C", self.query)
        self.assertEqual(non_busbar_edges(basic), non_busbar_edges(ranked))
        counts = {}
        for edge in ranked["predicted_edges"]:
            a, b = edge["source"], edge["target"]
            if {kinds[a], kinds[b]} == {"408", "410"}:
                device = a if kinds[a] == "408" else b
                counts[device] = counts.get(device, 0) + 1
        self.assertTrue(all(value <= 1 for value in counts.values()))

    def test_training_validation_overlap_rejected(self):
        with self.assertRaises(ValueError):
            baselines.train(
                self.data / "train", self.data / "train", self.folder / "invalid_overlap",
                depths=[3], rounds=[2],
            )


if __name__ == "__main__":
    unittest.main()

"""Meaningful input-boundary and complete synthetic execution checks."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from topology_inference.pipeline import Predictor, demo, evaluate, validate_document
from topology_inference.core import learned_reference_study as memory
from topology_inference.core import station_energy_study as energy
from topology_inference.synthetic import make_station


class InputContractTests(unittest.TestCase):
    def test_inference_drops_labels_and_nonallowlisted_fields(self):
        original = make_station(101)
        polluted = copy.deepcopy(original)
        polluted["labels"] = {"edges": "not even a valid target list"}
        for node in polluted["nodes"]:
            node["features"].update(connected_bus_id="POISON", nd="POISON", true_degree=999)
        self.assertEqual(validate_document(original), validate_document(polluted))
        self.assertNotIn("labels", validate_document(original))

    def test_duplicate_equipment_and_edges_rejected(self):
        d = make_station(102)
        d["nodes"].append(copy.deepcopy(d["nodes"][0]))
        with self.assertRaises(ValueError):
            validate_document(d, labels=True)
        d = make_station(103)
        d["labels"]["edges"].append(copy.deepcopy(d["labels"]["edges"][0]))
        with self.assertRaises(ValueError):
            validate_document(d, labels=True)


class SyntheticExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name) / "synthetic_run"
        # Five epochs also exercise a post-training HRM validation checkpoint.
        cls.result = demo(cls.folder, epochs=5)
        cls.predictor = Predictor(cls.folder / "bundle")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_all_model_stages_and_safe_bundle(self):
        self.assertTrue(self.result["synthetic_only"])
        self.assertTrue(self.predictor.meta["hrm"]["enabled"])
        for name in ("initial_mlp.npz", "ecpn.npz", "bcpn_0.npz", "bcpn_1000.npz", "hrm.npz"):
            self.assertTrue((self.folder / "bundle" / name).is_file())
        train = set(self.predictor.meta["train_station_ids"])
        val = set(self.predictor.meta["validation_station_ids"])
        self.assertFalse(train & val)
        self.assertEqual(len(train), 9)

    def test_targets_never_change_prediction(self):
        query = json.loads((self.folder / "data/query.json").read_text(encoding="utf-8"))
        clean = self.predictor.predict(query)
        query["labels"] = {"edges": []}
        for node in query["nodes"]:
            node["features"].update(nd="POISON", connected_bus_id="POISON", target_degree=1000)
        self.assertEqual(clean, self.predictor.predict(query))
        self.assertGreater(clean["historical_tasks"], 0)

    def test_train_station_cannot_be_called_independent_evaluation(self):
        with self.assertRaises(ValueError):
            evaluate(self.predictor, self.folder / "data/train")

    def test_alias_excluded_from_own_history(self):
        query = make_station(1)
        query["station_id"] = "INVENTED_ALIAS"
        query = validate_document(query)
        key = "station.INVENTED_ALIAS.json"
        physical_ids = {**self.predictor.physical_ids, key: query["group_id"]}
        state = memory.enrich(energy.prepare(query, self.predictor.enc), self.predictor.stage)
        tasks = memory.build(state, key, self.predictor.bank, list(self.predictor.bank),
                             self.predictor.scale, physical_ids=physical_ids)
        references = [ref["file"] for task in tasks for ref in task["refs"]]
        self.assertTrue(references)
        self.assertTrue(all(physical_ids[ref] != query["group_id"] for ref in references))

    def test_one_bus_target_per_disconnector(self):
        query = json.loads((self.folder / "data/query.json").read_text(encoding="utf-8"))
        result = self.predictor.predict(query)
        kinds = {n["node_id"]: n["features"]["type_code"] for n in query["nodes"]}
        counts = {}
        for e in result["predicted_edges"]:
            a, b = e["source"], e["target"]
            if {kinds[a], kinds[b]} == {"408", "410"}:
                disconnector = a if kinds[a] == "408" else b
                counts[disconnector] = counts.get(disconnector, 0) + 1
        self.assertTrue(all(n <= 1 for n in counts.values()))


if __name__ == "__main__":
    unittest.main()

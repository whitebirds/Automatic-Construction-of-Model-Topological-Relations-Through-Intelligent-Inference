"""DH-R regression checks using only fabricated equipment inventories."""
import copy
import math
import unittest

from topology_inference import dhr


def fabricated_station(prefix, station_id, group_id):
    def node(suffix, type_code, name):
        return {
            "node_id": prefix + suffix,
            "features": {
                "type_code": type_code,
                "name": name,
                "voltage_kv": 220,
                "winding_voltage_kv": [],
            },
        }

    return {
        "station_id": station_id,
        "group_id": group_id,
        "nodes": [
            node("a", "410", "示例正母Ⅰ段"),
            node("b", "408", "测试2651正母Ⅰ段闸刀"),
            node("c", "407", "测试265开关"),
        ],
        "labels": {"edges": [
            {"source": prefix + "a", "target": prefix + "b"},
            {"source": prefix + "b", "target": prefix + "c"},
        ]},
    }


class DirectHistoricalReuseTests(unittest.TestCase):
    def setUp(self):
        self.query = fabricated_station("q", "query", "query_group")
        self.reference = fabricated_station("r", "reference", "reference_group")

    def predict(self, query=None, references=None, **kwargs):
        return dhr.predict(
            self.query if query is None else query,
            [self.reference] if references is None else references,
            query_group_id="query_group",
            **kwargs,
        )

    def test_exact_adjacency_transfer_in_both_modes(self):
        for roles_filter in (False, True):
            result = self.predict(roles_filter=roles_filter)
            self.assertEqual(result["predicted_edges"], [["qa", "qb"], ["qb", "qc"]])

    def test_query_labels_and_extra_metadata_cannot_change_prediction(self):
        expected = self.predict()
        query = copy.deepcopy(self.query)
        query["labels"] = {"edges": [{"source": "qc", "target": "qa"}]}
        query["source_audit"] = {"invented_connection_metadata": "ignored"}
        for node in query["nodes"]:
            node["features"]["bus_name_attributes"] = {"bus_candidate_ids": ["invalid"]}
        self.assertEqual(self.predict(query), expected)
        del query["labels"]
        self.assertEqual(self.predict(query), expected)

    def test_order_and_id_invariance(self):
        expected = self.predict()["predicted_edges"]
        query = copy.deepcopy(self.query)
        query["nodes"].reverse()
        reference = copy.deepcopy(self.reference)
        reference["nodes"].reverse()
        self.assertEqual(self.predict(query, [reference])["predicted_edges"], expected)
        mapping = {"qa": "opaque3", "qb": "opaque1", "qc": "opaque2"}
        for node in query["nodes"]:
            node["node_id"] = mapping[node["node_id"]]
        del query["labels"]
        renamed = sorted(sorted([mapping[a], mapping[b]]) for a, b in expected)
        self.assertEqual(self.predict(query, [reference])["predicted_edges"], renamed)

    def test_group_exclusion_includes_aliases(self):
        alias = fabricated_station("x", "different_export_name", "query_group")
        result = self.predict(references=[alias, self.reference])
        self.assertEqual(result["reference_station"], "reference")
        self.assertEqual(result["reference_selection"]["eligible_reference_count"], 1)
        with self.assertRaises(ValueError):
            self.predict(references=[alias])

    def test_role_constraints_reject_explicit_family_mismatch(self):
        query_node = dhr.inventory(self.query)[1]
        reference = copy.deepcopy(self.reference)
        reference["nodes"][1]["features"]["name"] = "测试2651副母Ⅰ段闸刀"
        ref_node = dhr.inventory(reference)[1]
        self.assertTrue(dhr.compatible(query_node, ref_node, "DH-code"))
        self.assertFalse(dhr.compatible(query_node, ref_node, "DH-role"))

    def test_matching_equations_and_missing_codes(self):
        self.assertTrue(math.isclose(dhr.code_similarity("265", "266"), 0.9))
        self.assertTrue(math.isclose(dhr.code_similarity("2HX", "2HY"), 0.1))
        self.assertTrue(math.isclose(dhr.code_similarity("265", "2651"), 0.1))
        self.assertEqual(dhr.edit_distance("265", "2651"), 1)
        self.assertEqual(dhr.extract_code("示例正母Ⅰ段"), (None, "missing"))

    def test_invalid_reference_rejected(self):
        reference = copy.deepcopy(self.reference)
        reference["labels"]["edges"].append({"source": "ra", "target": "missing"})
        with self.assertRaises(ValueError):
            self.predict(references=[reference])
        with self.assertRaises(ValueError):
            self.predict(references=[])


if __name__ == "__main__":
    unittest.main()

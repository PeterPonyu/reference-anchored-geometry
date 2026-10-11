"""Source-authored unittest suite; execute only through the admitted P1 runner.

No new test dependency. The numerical suite has NOT RUN. The complete tail
ladder, ideal scaling and precision response records live in the separate
runner; these unit tests check its graph/identity/support building blocks.
"""
from __future__ import annotations

import ast
from decimal import Decimal, localcontext
import importlib.util
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORACLE_PATH = Path(__file__).with_name("raag_v3_oracle.py")
SCORER_PATH = ROOT / "scripts/local_raag_v3.py"
GEOMETRY_KEYS = (
    "reachability_survival", "edge_survival", "separation_survival",
    "n_reference_connected_pairs", "n_reference_edges", "n_far_pairs",
    "edge_severed_fraction", "edge_preserved_fraction",
    "separation_far_cutoff_hops", "separation_threshold_hops",
)
NATIVE_KEYS = (
    "allocation_coherence", "allocation_coherence_raw",
    "allocation_relative_coherence_unclipped", "allocation_edge_js",
    "allocation_pair_js", "allocation_mean_pair_tv",
    "allocation_max_coordinate_range", "allocation_mean_perplexity",
    "allocation_perplexity_fraction",
)


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ids(n, namespace="recipient"):
    return tuple((namespace, index) for index in range(n))


def sparse_edges(graph, identities, directed=False):
    rows, columns = graph.nonzero()
    return {(identities[int(row)], identities[int(column)]) if directed
            else tuple(sorted((identities[int(row)], identities[int(column)])))
            for row, column in zip(rows, columns) if directed or row < column}


def oracle_edges(graph, identities, directed=False):
    return {(identities[row], identities[column]) if directed
            else tuple(sorted((identities[row], identities[column])))
            for row, neighbours in enumerate(graph) for column in neighbours if directed or row < column}


def tail_fixture(epsilon="0.1"):
    # These delivered float64 values are tests of the implementation, not the
    # ideal Decimal identity. The runner preserves each design/input separately.
    reference = (np.arange(32, dtype=float)/31).reshape(-1, 1)
    q = np.asarray([[((31-i)*(j+1)+i*(9-j))/1395 for j in range(9)] for i in range(32)])
    value = float(epsilon)
    return reference, np.column_stack((np.full(32, 1-value), value*q)), ids(32, "raag-v3-tail-fixture")


class RAAGV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorer = load_source("_unit_local_raag_v3", SCORER_PATH)
        cls.oracle = load_source("_unit_raag_v3_oracle", ORACLE_PATH)

    def quantities(self, actual, expected, keys, atol=1e-12, rtol=1e-10):
        for key in keys:
            with self.subTest(quantity=key):
                left, right = actual.get(key), expected.get(key)
                if right is None:
                    self.assertIsNone(left)
                else:
                    self.assertIsNotNone(left)
                    self.assertTrue(math.isfinite(left))
                    self.assertLessEqual(abs(left-right), atol+rtol*abs(right))

    def graph_contract(self, points, identities, k):
        actual = self.scorer.build_knn_graph(points, k, cell_ids=identities)
        independent = self.oracle.build_knn_graph(points, k, cell_ids=identities)
        n, effective = len(identities), min(int(k), len(identities)-1)
        for matrix in (actual.directed, actual.graph):
            self.assertEqual(matrix.format, "csr")
            self.assertEqual(matrix.shape, (n, n))
            self.assertFalse(matrix.diagonal().any())
            self.assertTrue(np.all(matrix.data == 1))
        self.assertTrue(np.array_equal(np.diff(actual.directed.indptr), np.full(n, effective)))
        self.assertEqual(actual.directed.nnz, n*effective)
        self.assertEqual((actual.graph != actual.graph.T).nnz, 0)
        self.assertEqual((actual.graph != actual.directed.maximum(actual.directed.T)).nnz, 0)
        self.assertEqual(sparse_edges(actual.directed, identities, True), oracle_edges(independent.directed, identities, True))
        self.assertEqual(sparse_edges(actual.graph, identities), oracle_edges(independent.graph, identities))
        self.assertEqual(actual.metadata["effective_k"], effective)
        self.assertEqual(actual.metadata["input_dtype"], str(points.dtype))
        self.assertEqual(actual.metadata["distance_dtype"], "float64")
        self.assertEqual(actual.metadata["graph_status"], independent.metadata["graph_status"])
        self.assertEqual(actual.metadata["usable_geometry"], independent.metadata["usable_geometry"])
        self.assertEqual(actual.metadata["cutoff_tie_rows"], independent.metadata["cutoff_boundary_tie_rows"])
        self.assertRegex(actual.metadata["canonical_edges_sha256"], r"^[0-9a-f]{64}$")
        return actual, independent

    def allocation(self, module, x, p, identities, k=5, native=True):
        return module.allocation_diagnostics(x, p, reference_ids=identities, allocation_ids=identities,
                                             k=k, already_simplex=native, return_probabilities=True)

    def test_oracle_uses_only_independent_standard_library_imports(self):
        tree = ast.parse(ORACLE_PATH.read_text(), filename=str(ORACLE_PATH))
        allowed = {"__future__", "collections", "dataclasses", "decimal", "hashlib", "json", "math", "numbers", "typing"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(alias.name.split(".")[0] in allowed for alias in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0)
                self.assertIn(node.module.split(".")[0], allowed)

    def test_decimal_jsd_analytic_anchors(self):
        with localcontext() as context:
            context.prec = 80
            left, right = (1., 0.), (0., 1.)
            self.assertEqual(self.oracle.decimal_jsd(left, left), Decimal(0))
            self.assertLess(abs(self.oracle.decimal_jsd(left, right)-Decimal(2).ln()), Decimal("1e-75"))
            self.assertEqual(self.oracle.decimal_jsd(left, right), self.oracle.decimal_jsd(right, left))
            self.assertGreater(self.oracle.decimal_jsd((1., 1e-14), (1., 2e-14)), 0)

    def test_duplicate_group_larger_than_k_plus_one(self):
        points = np.asarray([[0.,0.]]*20 + [[2.,0.],[3.,1.],[4.,2.]])
        identities = tuple(("duplicate-group", 100-i) for i in range(len(points)))
        for k in (1, 5, 22, 29):
            with self.subTest(k=k):
                actual, _ = self.graph_contract(points, identities, k)
                if k <= 5:
                    self.assertGreater(actual.metadata["cutoff_tie_rows"], 0)
                if k == 1:
                    self.assertEqual(actual.directed[0, 19], 1)
                    self.assertEqual(actual.directed[19, 18], 1)

    def test_distinct_equidistant_cutoff_uses_full_id_order(self):
        points = np.asarray([[0.,0.],[1.,0.],[-1.,0.],[0.,1.],[0.,-1.],[3.,3.]])
        identities = (("z",10),("beta",4),("alpha",7),("alpha",2),("z",0),("omega",1))
        for k in (1, 2, 5):
            with self.subTest(k=k):
                actual, _ = self.graph_contract(points, identities, k)
                if k == 1:
                    self.assertEqual(actual.directed[0,3], 1)
                if k == 2:
                    self.assertEqual(set(actual.directed[0].indices), {2,3})

    def test_joint_reorder_preserves_logical_graph_and_hash(self):
        points = np.asarray([[0.,0.]]*8+[[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
        identities = tuple(("alpha" if i % 2 else "omega", 50-i) for i in range(len(points)))
        order = np.asarray([7,1,11,3,9,5,0,10,2,8,4,6])
        moved = tuple(identities[int(i)] for i in order)
        for k in (1,5,11):
            with self.subTest(k=k):
                before, _ = self.graph_contract(points, identities, k)
                after, _ = self.graph_contract(points[order], moved, k)
                self.assertEqual(sparse_edges(before.directed, identities, True), sparse_edges(after.directed, moved, True))
                self.assertEqual(sparse_edges(before.graph, identities), sparse_edges(after.graph, moved))
                self.assertEqual(before.metadata["canonical_edges_sha256"], after.metadata["canonical_edges_sha256"])

    def test_graph_hash_includes_namespace_identity(self):
        points = np.asarray([[0.],[1.],[3.],[8.],[19.]])
        a = self.scorer.build_knn_graph(points,1,cell_ids=ids(5,"a"))
        b = self.scorer.build_knn_graph(points,1,cell_ids=ids(5,"b"))
        self.assertEqual((a.graph != b.graph).nnz, 0)
        self.assertNotEqual(a.metadata["canonical_edges_sha256"], b.metadata["canonical_edges_sha256"])

    def test_invalid_graph_arguments_and_id_contract(self):
        points = np.asarray([[0.],[1.],[4.]])
        invalid_ids = (None, (("n",0),), (("n",0),("n",0),("n",2)),
                       (("",0),("n",1),("n",2)), ((" ",0),("n",1),("n",2)),
                       ((2,0),("n",1),("n",2)), (("n",-1),("n",1),("n",2)),
                       (("n",True),("n",1),("n",2)), (("n",0.),("n",1),("n",2)),
                       (("n",0,8),("n",1),("n",2)))
        for module in (self.scorer,self.oracle):
            for value in (0,-1,1.5,"5",None,True,False,np.bool_(True)):
                with self.subTest(module=module.__name__,k=value), self.assertRaises((TypeError,ValueError)):
                    module.build_knn_graph(points,value,cell_ids=ids(3))
            for value in invalid_ids:
                with self.subTest(module=module.__name__,identities=value), self.assertRaises((TypeError,ValueError)):
                    module.build_knn_graph(points,1,cell_ids=value)
            valid = module.build_knn_graph(points,np.int64(1),cell_ids=[["n",0],["n",1],["n",2]])
            self.assertEqual(valid.metadata["effective_k"], 1)

    def test_index_helper_preserves_integer_identity_and_rejects_invalid_inputs(self):
        for module in (self.scorer,self.oracle):
            self.assertEqual(module.cell_ids_from_indices("held-out",np.asarray([19,2,8,5]),n=4),
                             (("held-out",19),("held-out",2),("held-out",8),("held-out",5)))
            for value in ([0.,1.],[True,False],[-1,2],[2,2],[[0,1]]):
                with self.subTest(module=module.__name__,value=value), self.assertRaises((TypeError,ValueError)):
                    module.cell_ids_from_indices("n",np.asarray(value))
            for namespace in ("", " ", None, 5):
                with self.subTest(namespace=namespace), self.assertRaises((TypeError,ValueError)):
                    module.cell_ids_from_indices(namespace,np.asarray([0,1]))
            with self.assertRaises((TypeError,ValueError)):
                module.cell_ids_from_indices("n",np.asarray([0,1]),n=3)

    def test_invalid_shape_nonfinite_and_overflow(self):
        for module in (self.scorer,self.oracle):
            for points in ([],[[]],[[0.]],[[],[]],[0.,1.],[[[0.]],[[1.]]],[[0.],[1.,2.]],[[0.],[math.nan]],[[0.],[math.inf]],[[1e200],[-1e200]],np.asarray([[1j],[2j]])):
                with self.subTest(module=module.__name__,points=points), self.assertRaises((TypeError,ValueError)):
                    module.build_knn_graph(points,1,cell_ids=ids(len(points)))
            x,p,identities = tail_fixture()
            for bad in (math.nan,math.inf,-math.inf):
                z=p.copy(); z[1,0]=bad
                with self.assertRaises((TypeError,ValueError)):
                    self.allocation(module,x,z,identities)
            with self.assertRaises((TypeError,ValueError)):
                module.raag_battery(x,p[:-1],reference_ids=identities,candidate_ids=identities[:-1],already_simplex=True)
            for quantile in (.49,1.,math.nan,math.inf):
                with self.assertRaises((TypeError,ValueError)):
                    module.raag_battery(x,p,reference_ids=identities,candidate_ids=identities,far_quantile=quantile,already_simplex=True)

    def test_native_boolean_and_nonnegative_row_stochastic_contract(self):
        x,p,identities = tail_fixture()
        for module in (self.scorer,self.oracle):
            for flag in (None,0,1,"native"):
                with self.subTest(flag=flag), self.assertRaises((TypeError,ValueError)):
                    self.allocation(module,x,p,identities,native=flag)
            self.assertEqual(self.allocation(module,x,p,identities,native=np.bool_(True))["allocation_status"],"eligible")
            for row in ((-1e-13,1.0000000000001),(.5,.5002),(.5,.4998),(1.,1.),(0.,0.)):
                with self.subTest(row=row), self.assertRaises((TypeError,ValueError)):
                    self.allocation(module,x,np.asarray([row]*32),identities)
        with self.assertRaises(TypeError):
            self.scorer.allocation_diagnostics(x,p,reference_ids=identities,allocation_ids=identities)
        with self.assertRaises(TypeError):
            self.scorer.raag_battery(x,p,reference_ids=identities,candidate_ids=identities)

    def test_floor_and_normalisation_preserve_delivered_input(self):
        x=np.arange(6,dtype=float).reshape(-1,1)
        p=np.asarray([[1.00005,0.],[.7,.30005],[0.,.99995],[.8,.19995],[1.,0.],[.3,.7]])
        old=p.copy(); identities=ids(6)
        actual=self.allocation(self.scorer,x,p,identities,k=1)
        expected=self.allocation(self.oracle,x,p,identities,k=1)
        self.assertTrue(np.array_equal(p,old))
        self.assertEqual(actual["flooring_entries"],3)
        self.assertEqual(actual["flooring_entries"],expected["flooring_entries"])
        self.assertTrue(np.allclose(actual["measured_probabilities"],expected["measured_probabilities"],atol=2e-16,rtol=2e-15))
        self.quantities(actual,expected,NATIVE_KEYS)

    def test_constant_and_small_geometry_are_not_perfect_scores(self):
        x,p,identities=tail_fixture()
        for which in ("reference","candidate","both"):
            with self.subTest(which=which):
                ref=np.zeros_like(x) if which in ("reference","both") else x
                z=np.ones_like(p) if which in ("candidate","both") else p
                result=self.scorer.raag_battery(ref,z,reference_ids=identities,candidate_ids=identities,k=5,already_simplex=False)
                self.assertNotEqual(result["geometry_status"],"eligible")
                for key in GEOMETRY_KEYS[:3]: self.assertIsNone(result[key])
        zeros=np.zeros((20,2)); self.graph_contract(zeros,ids(20),5)
        self.assertFalse(self.scorer.build_knn_graph(zeros,5,cell_ids=ids(20)).metadata["usable_geometry"])
        tiny=self.scorer.raag_battery(np.asarray([[0.],[2.]]),np.asarray([[.8,.2],[.3,.7]]),reference_ids=ids(2),candidate_ids=ids(2),already_simplex=True)
        self.assertEqual(tiny["geometry_status"],"insufficient_reference_pairs")
        self.assertIsNone(tiny["reachability_survival"])

    def test_constant_reference_withholds_edge_ratio_but_keeps_absolute_diagnostics(self):
        x,p,identities=tail_fixture(); x[:]=0
        actual=self.allocation(self.scorer,x,p,identities)
        expected=self.allocation(self.oracle,x,p,identities)
        self.assertEqual(actual["allocation_status"],"unsupported_reference")
        self.quantities(actual,expected,NATIVE_KEYS)
        for key in ("allocation_edge_js","allocation_coherence","allocation_coherence_raw","allocation_relative_coherence_unclipped"):
            self.assertIsNone(actual[key])
        for key in ("allocation_pair_js","allocation_mean_pair_tv","allocation_max_coordinate_range"):
            self.assertGreater(actual[key],0)

    def test_correspondence_damage_stays_with_recipient_ids(self):
        x,p,identities=tail_fixture()
        order=np.asarray([(13*i+7)%32 for i in range(32)])
        before=self.allocation(self.scorer,x,p,identities)
        actual=self.allocation(self.scorer,x,p[order],identities)
        expected=self.allocation(self.oracle,x,p[order],identities)
        self.quantities(actual,expected,NATIVE_KEYS)
        self.assertGreater(abs(actual["allocation_edge_js"]-before["allocation_edge_js"]),1e-15)
        self.assertLessEqual(abs(actual["allocation_pair_js"]-before["allocation_pair_js"]),1e-12)
        donor_ids=tuple(identities[int(i)] for i in order)
        for module in (self.scorer,self.oracle):
            with self.assertRaises((TypeError,ValueError)):
                module.allocation_diagnostics(x,p[order],reference_ids=identities,allocation_ids=donor_ids,k=5,already_simplex=True)

    def test_joint_reorder_preserves_applicable_battery_scores(self):
        x,p,identities=tail_fixture()
        order=np.asarray([(13*i+7)%32 for i in range(32)])
        moved=tuple(identities[int(i)] for i in order)
        first=self.scorer.raag_battery(x,p,reference_ids=identities,candidate_ids=identities,k=5,already_simplex=True)
        second=self.scorer.raag_battery(x[order],p[order],reference_ids=moved,candidate_ids=moved,k=5,already_simplex=True)
        self.quantities(first,second,GEOMETRY_KEYS+NATIVE_KEYS,rtol=0)
        self.assertEqual(first["reference_graph"]["canonical_edges_sha256"],second["reference_graph"]["canonical_edges_sha256"])
        self.assertNotEqual(first["ordered_ids_sha256"],second["ordered_ids_sha256"])

    def test_tie_free_hop_geometry_matches_independent_bfs(self):
        index=np.arange(32,dtype=float)
        x=np.column_stack((index,.013*index**2+.001*index**3))
        order=np.asarray([(7*i)%32 for i in range(32)])
        z=np.column_stack((x[order,0],1.7*x[order,1],.01*(index%7)))
        identities=ids(32)
        for k in (1,5,31):
            with self.subTest(k=k):
                for coordinates in (x,z):
                    graph,_=self.graph_contract(coordinates,identities,k)
                    self.assertEqual(graph.metadata["cutoff_tie_rows"],0)
                kwargs=dict(reference_ids=identities,candidate_ids=identities,k=k,already_simplex=False)
                actual=self.scorer.raag_battery(x,z,**kwargs)
                expected=self.oracle.raag_battery(x,z,**kwargs)
                self.assertEqual(actual["geometry_status"],expected["geometry_status"])
                self.quantities(actual,expected,GEOMETRY_KEYS,rtol=0)

    def test_near_uniform_degenerate_and_negative_ratio_have_distinct_meaning(self):
        x,p,identities=tail_fixture()
        for kind in ("constant","uniform","near_uniform"):
            with self.subTest(kind=kind):
                if kind=="constant": z=np.asarray([[.8,.2]]*32)
                elif kind=="uniform": z=np.full((32,10),.1)
                else:
                    offset=.01*(2*x[:,0]-1)
                    z=np.column_stack((.5+offset,.5-offset))
                actual=self.allocation(self.scorer,x,z,identities)
                expected=self.allocation(self.oracle,x,z,identities)
                self.assertEqual(actual["allocation_status"],"near_uniform" if kind=="near_uniform" else "degenerate")
                self.assertEqual(actual["allocation_status"],expected["allocation_status"])
                self.assertIsNone(actual["allocation_coherence"])
                if kind != "near_uniform":
                    self.assertEqual(actual["allocation_coherence_raw"],0)
                self.quantities(actual,expected,NATIVE_KEYS)
        x=np.arange(8,dtype=float).reshape(-1,1)
        p=np.asarray([[1.,0.] if i%2 else [0.,1.] for i in range(8)])
        actual=self.allocation(self.scorer,x,p,ids(8),k=1)
        expected=self.allocation(self.oracle,x,p,ids(8),k=1)
        self.quantities(actual,expected,NATIVE_KEYS)
        self.assertLess(actual["allocation_relative_coherence_unclipped"],0)
        self.assertEqual(actual["allocation_coherence"],0)

    def test_nonnative_coordinates_never_become_native_allocations(self):
        x,_,identities=tail_fixture(); t=x[:,0]
        p=np.column_stack((4*t-2,1-t,-3*t))
        actual=self.allocation(self.scorer,x,p,identities,native=False)
        expected=self.allocation(self.oracle,x,p,identities,native=False)
        self.assertEqual(actual["allocation_status"],"non_native")
        self.assertIsNone(actual["allocation_coherence"])
        self.assertEqual(actual["softmax_coordinate_dimension"],3)
        self.quantities(actual,expected,("softmax_coordinate_edge_js","softmax_coordinate_pair_js","softmax_coordinate_perplexity_fraction"))

    def test_probability_payload_is_opt_in(self):
        x,p,identities=tail_fixture()
        result=self.scorer.allocation_diagnostics(x,p,reference_ids=identities,allocation_ids=identities,already_simplex=True)
        self.assertNotIn("input_probabilities",result)
        self.assertNotIn("measured_probabilities",result)

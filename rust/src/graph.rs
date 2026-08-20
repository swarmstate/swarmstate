//! Deterministic, LLM-free handoff graph.
//!
//! A directed graph of nodes (agents/states) connected by edges that carry an
//! optional condition (see [`crate::condition`]). [`HandoffGraph::route`]
//! resolves the next node by evaluating each outgoing edge's condition against
//! the routing state, in insertion order, and returning the first match —
//! entirely in Rust, with the GIL released during matching.

use std::collections::{HashMap, HashSet};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rmpv::Value;

use crate::codec;
use crate::condition::{self, Expr};

struct Edge {
    to: String,
    when_src: Option<String>,
    cond: Option<Expr>,
}

/// A deterministic routing graph over named nodes with conditional edges.
#[pyclass(module = "swarmstate._core")]
pub struct HandoffGraph {
    adj: HashMap<String, Vec<Edge>>,
    nodes: HashSet<String>,
    on_cycle: String,
    // The state paths the outgoing conditions of each node read, so route() can
    // materialize just those instead of converting the entire routing state.
    node_paths: HashMap<String, Vec<Vec<String>>>,
}

/// Drop paths already covered by a shorter one (`a.b` under `a`).
///
/// Materializing both would put the same key in the partial state twice.
fn drop_covered_paths(paths: &mut Vec<Vec<String>>) {
    paths.sort_by_key(Vec::len);
    let mut kept: Vec<Vec<String>> = Vec::with_capacity(paths.len());
    for path in paths.drain(..) {
        if !kept.iter().any(|k| path.starts_with(k)) {
            kept.push(path);
        }
    }
    *paths = kept;
}

/// Walk `state` down `path`, returning the value there if every step is a dict.
fn resolve_path<'py>(
    state: &Bound<'py, PyDict>,
    path: &[String],
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let mut current: Bound<'py, PyAny> = state.clone().into_any();
    for segment in path {
        let Ok(dict) = current.cast::<PyDict>() else {
            return Ok(None);
        };
        match dict.get_item(segment.as_str())? {
            Some(value) => current = value,
            None => return Ok(None),
        }
    }
    Ok(Some(current))
}

/// Insert `leaf` into a nested map skeleton at `path`, as the evaluator expects.
fn insert_path(pairs: &mut Vec<(Value, Value)>, path: &[String], leaf: Value) {
    let key = Value::String(path[0].clone().into());
    if path.len() == 1 {
        pairs.push((key, leaf));
        return;
    }
    if let Some((_, existing)) = pairs.iter_mut().find(|(k, _)| *k == key) {
        if let Value::Map(inner) = existing {
            insert_path(inner, &path[1..], leaf);
        }
        return;
    }
    let mut inner = Vec::new();
    insert_path(&mut inner, &path[1..], leaf);
    pairs.push((key, Value::Map(inner)));
}

impl HandoffGraph {
    /// True if adding `from -> to` would introduce a cycle, i.e. `to` can
    /// already reach `from` (or it's a self-loop).
    fn would_create_cycle(&self, from: &str, to: &str) -> bool {
        if from == to {
            return true;
        }
        // Borrowed node names: the traversal allocates nothing.
        let mut stack: Vec<&str> = vec![to];
        let mut seen: HashSet<&str> = HashSet::new();
        while let Some(n) = stack.pop() {
            if n == from {
                return true;
            }
            if !seen.insert(n) {
                continue;
            }
            if let Some(edges) = self.adj.get(n) {
                for e in edges {
                    stack.push(e.to.as_str());
                }
            }
        }
        false
    }

    /// Recompute the paths the outgoing conditions of `node` read.
    fn refresh_paths(&mut self, node: &str) {
        let mut paths = Vec::new();
        if let Some(edges) = self.adj.get(node) {
            for edge in edges {
                if let Some(cond) = &edge.cond {
                    condition::collect_paths(cond, &mut paths);
                }
            }
        }
        drop_covered_paths(&mut paths);
        self.node_paths.insert(node.to_string(), paths);
    }

    /// Build a routing state holding only the paths the conditions of `node` read.
    fn partial_state(&self, node: &str, state: &Bound<'_, PyDict>) -> PyResult<Value> {
        let mut pairs = Vec::new();
        if let Some(paths) = self.node_paths.get(node) {
            for path in paths {
                if let Some(value) = resolve_path(state, path)? {
                    insert_path(&mut pairs, path, codec::py_to_value(&value)?);
                }
            }
        }
        Ok(Value::Map(pairs))
    }
}

#[pymethods]
impl HandoffGraph {
    #[new]
    #[pyo3(signature = (on_cycle = "error"))]
    fn new(on_cycle: &str) -> PyResult<Self> {
        if on_cycle != "error" && on_cycle != "allow" {
            return Err(PyValueError::new_err("on_cycle must be 'error' or 'allow'"));
        }
        Ok(HandoffGraph {
            adj: HashMap::new(),
            nodes: HashSet::new(),
            on_cycle: on_cycle.to_string(),
            node_paths: HashMap::new(),
        })
    }

    /// Behaviour on cycle detection: `"error"` or `"allow"`.
    #[getter]
    fn on_cycle(&self) -> &str {
        &self.on_cycle
    }

    /// Register a node with no edges (edges also register their endpoints).
    fn add_node(&mut self, name: String) {
        self.nodes.insert(name);
    }

    /// Add a directed edge `from -> to`, optionally guarded by a `when`
    /// condition (see the condition mini-language).
    ///
    /// Raises `ValueError` on an invalid condition, or — when `on_cycle` is
    /// `"error"` — if the edge would create a cycle.
    #[pyo3(signature = (from_node, to, when = None))]
    fn add_edge(&mut self, from_node: String, to: String, when: Option<&str>) -> PyResult<()> {
        let cond = match when {
            Some(src) => Some(
                condition::parse(src)
                    .map_err(|e| PyValueError::new_err(format!("invalid condition: {e}")))?,
            ),
            None => None,
        };

        if self.on_cycle == "error" && self.would_create_cycle(&from_node, &to) {
            return Err(PyValueError::new_err(format!(
                "adding edge '{from_node}' -> '{to}' would create a cycle (on_cycle='error')"
            )));
        }

        self.nodes.insert(from_node.clone());
        self.nodes.insert(to.clone());
        self.adj.entry(from_node.clone()).or_default().push(Edge {
            to,
            when_src: when.map(str::to_string),
            cond,
        });
        self.refresh_paths(&from_node);
        Ok(())
    }

    /// Resolve the next node from `node` given `state`.
    ///
    /// Evaluates outgoing edges in insertion order and returns the first whose
    /// condition is satisfied (an edge with no condition always matches).
    /// Returns `None` if no edge matches.
    #[pyo3(signature = (node, state = None))]
    fn route(
        &self,
        py: Python<'_>,
        node: &str,
        state: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Option<String>> {
        // Converting the whole state was the dominant cost of routing: a realistic
        // LangGraph state (a few hundred messages) took ~165µs to encode, all of it
        // to read one field. Pull just the paths the conditions name instead.
        let state_val = match state {
            None => Value::Map(Vec::new()),
            Some(obj) => match obj.cast::<PyDict>() {
                Ok(dict) => self.partial_state(node, dict)?,
                // Not a mapping: no paths can resolve, but keep converting it so
                // an unsupported state type still reports the same error.
                Err(_) => codec::py_to_value(obj)?,
            },
        };
        let result = py.detach(|| {
            let edges = self.adj.get(node)?;
            for e in edges {
                let matched = match &e.cond {
                    None => true,
                    Some(expr) => condition::eval_truthy(expr, &state_val),
                };
                if matched {
                    return Some(e.to.clone());
                }
            }
            None
        });
        Ok(result)
    }

    /// All nodes in the graph (sorted for stable output).
    fn nodes(&self) -> Vec<String> {
        let mut v: Vec<String> = self.nodes.iter().cloned().collect();
        v.sort();
        v
    }

    /// Outgoing edges of `node` as `(to, when)` pairs, in insertion order.
    fn edges(&self, node: &str) -> Vec<(String, Option<String>)> {
        self.adj
            .get(node)
            .map(|edges| {
                edges
                    .iter()
                    .map(|e| (e.to.clone(), e.when_src.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Whether `node` exists in the graph.
    fn has_node(&self, node: &str) -> bool {
        self.nodes.contains(node)
    }

    /// Whether the graph is currently acyclic.
    ///
    /// Kahn's algorithm: O(V+E) and iterative. The previous reachability search
    /// per edge was O(V·E) — half a second on a 3000-node chain — and recursed
    /// once per node, so a deep graph could overflow the stack.
    fn is_dag(&self) -> bool {
        let mut indegree: HashMap<&str, usize> =
            self.nodes.iter().map(|n| (n.as_str(), 0usize)).collect();
        for edges in self.adj.values() {
            for e in edges {
                *indegree.entry(e.to.as_str()).or_insert(0) += 1;
            }
        }
        let mut ready: Vec<&str> = indegree
            .iter()
            .filter(|(_, d)| **d == 0)
            .map(|(n, _)| *n)
            .collect();

        let mut settled = 0usize;
        while let Some(node) = ready.pop() {
            settled += 1;
            if let Some(edges) = self.adj.get(node) {
                for e in edges {
                    if let Some(d) = indegree.get_mut(e.to.as_str()) {
                        *d -= 1;
                        if *d == 0 {
                            ready.push(e.to.as_str());
                        }
                    }
                }
            }
        }
        // Anything left has an unbroken incoming edge: it sits on a cycle.
        settled == indegree.len()
    }

    fn __len__(&self) -> usize {
        self.nodes.len()
    }

    fn __contains__(&self, node: &str) -> bool {
        self.nodes.contains(node)
    }

    fn __repr__(&self) -> String {
        let edge_count: usize = self.adj.values().map(Vec::len).sum();
        format!(
            "HandoffGraph(nodes={}, edges={}, on_cycle='{}')",
            self.nodes.len(),
            edge_count,
            self.on_cycle
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dict<'py>(py: Python<'py>, pairs: &[(&str, &str)]) -> Bound<'py, PyAny> {
        let d = pyo3::types::PyDict::new(py);
        for (k, v) in pairs {
            d.set_item(k, v).unwrap();
        }
        d.into_any()
    }

    #[test]
    fn deterministic_first_match() {
        Python::attach(|py| {
            let mut g = HandoffGraph::new("error").unwrap();
            g.add_edge(
                "triage".into(),
                "billing".into(),
                Some("category == 'billing'"),
            )
            .unwrap();
            g.add_edge(
                "triage".into(),
                "support".into(),
                Some("category == 'support'"),
            )
            .unwrap();
            g.add_edge("triage".into(), "human".into(), None).unwrap(); // default

            let st = dict(py, &[("category", "billing")]);
            assert_eq!(
                g.route(py, "triage", Some(&st)).unwrap(),
                Some("billing".to_string())
            );

            let st = dict(py, &[("category", "support")]);
            assert_eq!(
                g.route(py, "triage", Some(&st)).unwrap(),
                Some("support".to_string())
            );

            // No condition matches except the unconditional default.
            let st = dict(py, &[("category", "other")]);
            assert_eq!(
                g.route(py, "triage", Some(&st)).unwrap(),
                Some("human".to_string())
            );
        });
    }

    #[test]
    fn no_match_returns_none() {
        Python::attach(|py| {
            let mut g = HandoffGraph::new("error").unwrap();
            g.add_edge("a".into(), "b".into(), Some("x == 1")).unwrap();
            let st = dict(py, &[]);
            assert_eq!(g.route(py, "a", Some(&st)).unwrap(), None);
            assert_eq!(g.route(py, "unknown", None).unwrap(), None);
        });
    }

    #[test]
    fn cycle_detection_errors() {
        let mut g = HandoffGraph::new("error").unwrap();
        g.add_edge("a".into(), "b".into(), None).unwrap();
        g.add_edge("b".into(), "c".into(), None).unwrap();
        // c -> a would close a cycle a->b->c->a.
        assert!(g.add_edge("c".into(), "a".into(), None).is_err());
        // self-loop.
        assert!(g.add_edge("a".into(), "a".into(), None).is_err());
        assert!(g.is_dag());
    }

    #[test]
    fn detects_a_cycle_that_excludes_some_nodes() {
        let mut g = HandoffGraph::new("allow").unwrap();
        g.add_edge("a".into(), "b".into(), None).unwrap();
        g.add_edge("b".into(), "c".into(), None).unwrap();
        g.add_edge("c".into(), "b".into(), None).unwrap(); // b <-> c cycle
        g.add_node("lonely".into());
        assert!(!g.is_dag());
    }

    #[test]
    fn is_dag_handles_diamonds_and_multi_edges() {
        let mut g = HandoffGraph::new("allow").unwrap();
        g.add_edge("a".into(), "b".into(), None).unwrap();
        g.add_edge("a".into(), "c".into(), None).unwrap();
        g.add_edge("b".into(), "d".into(), None).unwrap();
        g.add_edge("c".into(), "d".into(), None).unwrap();
        g.add_edge("a".into(), "b".into(), None).unwrap(); // duplicate edge
        assert!(g.is_dag());
    }

    #[test]
    fn only_the_referenced_paths_are_read() {
        Python::attach(|py| {
            let mut g = HandoffGraph::new("error").unwrap();
            g.add_edge("s".into(), "deep".into(), Some("data.user.role == 'admin'"))
                .unwrap();
            g.add_edge("s".into(), "shallow".into(), Some("data == 'plain'"))
                .unwrap();

            // Overlapping paths: `data` covers `data.user.role`, so the partial
            // state must not carry the same key twice.
            let paths = g.node_paths.get("s").unwrap();
            assert_eq!(paths, &vec![vec!["data".to_string()]]);

            let state = pyo3::types::PyDict::new(py);
            let data = pyo3::types::PyDict::new(py);
            let user = pyo3::types::PyDict::new(py);
            user.set_item("role", "admin").unwrap();
            data.set_item("user", &user).unwrap();
            state.set_item("data", &data).unwrap();
            assert_eq!(
                g.route(py, "s", Some(state.as_any())).unwrap(),
                Some("deep".to_string())
            );
        });
    }

    #[test]
    fn cycle_allowed_when_configured() {
        let mut g = HandoffGraph::new("allow").unwrap();
        g.add_edge("a".into(), "b".into(), None).unwrap();
        g.add_edge("b".into(), "a".into(), None).unwrap();
        assert!(!g.is_dag());
        assert_eq!(g.nodes(), vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn invalid_condition_rejected() {
        let mut g = HandoffGraph::new("error").unwrap();
        assert!(g.add_edge("a".into(), "b".into(), Some("x = 1")).is_err());
    }
}

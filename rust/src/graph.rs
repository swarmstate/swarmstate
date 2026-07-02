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
}

impl HandoffGraph {
    /// True if adding `from -> to` would introduce a cycle, i.e. `to` can
    /// already reach `from` (or it's a self-loop).
    fn would_create_cycle(&self, from: &str, to: &str) -> bool {
        if from == to {
            return true;
        }
        let mut stack = vec![to.to_string()];
        let mut seen = HashSet::new();
        while let Some(n) = stack.pop() {
            if n == from {
                return true;
            }
            if !seen.insert(n.clone()) {
                continue;
            }
            if let Some(edges) = self.adj.get(&n) {
                for e in edges {
                    stack.push(e.to.clone());
                }
            }
        }
        false
    }

    fn reaches(&self, start: &str, target: &str, seen: &mut HashSet<String>) -> bool {
        if start == target {
            return true;
        }
        if !seen.insert(start.to_string()) {
            return false;
        }
        if let Some(edges) = self.adj.get(start) {
            for e in edges {
                if self.reaches(&e.to, target, seen) {
                    return true;
                }
            }
        }
        false
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
        self.adj.entry(from_node).or_default().push(Edge {
            to,
            when_src: when.map(str::to_string),
            cond,
        });
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
        let state_val = match state {
            Some(obj) => codec::py_to_value(obj)?,
            None => Value::Map(Vec::new()),
        };
        let result = py.allow_threads(|| {
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
    fn is_dag(&self) -> bool {
        let nodes: Vec<String> = self.nodes.iter().cloned().collect();
        for n in &nodes {
            if let Some(edges) = self.adj.get(n) {
                for e in edges {
                    let mut seen = HashSet::new();
                    // A cycle exists if a successor can reach n.
                    if self.reaches(&e.to, n, &mut seen) {
                        return false;
                    }
                }
            }
        }
        true
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
        Python::with_gil(|py| {
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
        Python::with_gil(|py| {
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

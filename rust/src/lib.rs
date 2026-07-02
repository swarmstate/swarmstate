//! swarmstate_core — Rust core for the `swarmstate` state & checkpointing backend.
//!
//! This crate is compiled by maturin into the native extension module
//! `swarmstate._core`. The public Python API lives in `python/swarmstate/`
//! and wraps the classes/functions exported here.
//!
//! Milestone status: M0 (scaffolding). Later milestones add the concurrent
//! store (M1), the handoff graph (M2), and the codec used by both.

use pyo3::prelude::*;

mod codec;
mod condition;
mod graph;
mod store;

/// Version of the compiled Rust core. Mirrors the crate version in Cargo.toml.
const CORE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Return the version string of the compiled Rust core.
///
/// Used by the Python package and tests to confirm the native module loaded.
#[pyfunction]
fn core_version() -> &'static str {
    CORE_VERSION
}

/// The `swarmstate._core` native module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", CORE_VERSION)?;
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_class::<store::Store>()?;
    m.add_class::<store::Snapshot>()?;
    m.add_class::<graph::HandoffGraph>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_matches_crate() {
        assert_eq!(core_version(), env!("CARGO_PKG_VERSION"));
        assert!(!core_version().is_empty());
    }
}

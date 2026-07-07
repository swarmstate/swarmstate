//! swarmstate_core — Rust core for the `swarmstate` state & checkpointing backend.
//!
//! This crate is compiled by maturin into the native extension module
//! `swarmstate._core`. The public Python API lives in `python/swarmstate/`
//! and wraps the classes/functions exported here.
//!
//! Milestone status: M0 (scaffolding). Later milestones add the concurrent
//! store (M1), the handoff graph (M2), and the codec used by both.

use pyo3::prelude::*;
use pyo3::types::PyBytes;

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

/// Serialize a Python object to msgpack bytes (swarmstate's stable, cross-language codec).
#[pyfunction]
fn dumps<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBytes>> {
    Ok(PyBytes::new(py, &codec::encode(obj)?))
}

/// Deserialize msgpack bytes back into a Python object.
#[pyfunction]
fn loads<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyAny>> {
    codec::decode(py, data)
}

/// The `swarmstate._core` native module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Declare the extension safe on free-threaded (no-GIL) CPython: all shared
    // state lives behind RwLocks/atomics in the Rust core, and PyO3's per-object
    // borrow checking guards the &mut self methods. Without this, importing on a
    // free-threaded interpreter would force the GIL back on.
    m.gil_used(false)?;
    m.add("__version__", CORE_VERSION)?;
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_function(wrap_pyfunction!(dumps, m)?)?;
    m.add_function(wrap_pyfunction!(loads, m)?)?;
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

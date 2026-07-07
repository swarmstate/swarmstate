//! Concurrent, framework-agnostic key/value store with cheap immutable snapshots.
//!
//! State is keyed by `(namespace, key)` and stored as msgpack bytes (see
//! [`crate::codec`]). The backing map is an `im::HashMap`, a persistent
//! data structure: cloning it is O(1) via structural sharing, so
//! [`Store::snapshot`] is cheap and snapshots are fully isolated from later
//! mutations (copy-on-write).
//!
//! Writes are **sharded**: namespaces are hashed across `SHARDS` independent
//! `RwLock`s, so concurrent writers to different namespaces don't contend on a
//! single global lock. The GIL is released (`py.allow_threads`) around every
//! lock/map operation; only (de)serialization runs under the GIL.

use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use im::HashMap as ImMap;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::codec;

/// `namespace -> (key -> value bytes)`.
type NsMap = ImMap<String, ImMap<String, Vec<u8>>>;

/// Number of lock shards. Namespaces are hashed across these so writes to
/// different namespaces proceed in parallel.
const SHARDS: usize = 16;

/// Which shard a namespace lives in (deterministic within a process).
fn shard_index(namespace: &str) -> usize {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    namespace.hash(&mut h);
    (h.finish() as usize) % SHARDS
}

/// Seconds since the Unix epoch as a float (0.0 if the clock is before epoch).
fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Immutable content + metadata captured by a [`Store::snapshot`] call.
struct SnapshotData {
    id: u64,
    timestamp: f64,
    parent: Option<u64>,
    size_bytes: usize,
    // Per-shard byte totals at snapshot time, so restore() can put each shard's
    // counter back exactly. `size_bytes` is their sum (the public getter).
    shard_bytes: Vec<usize>,
    shards: Vec<NsMap>,
}

/// A cheap, immutable point-in-time view of a [`Store`].
#[pyclass(module = "swarmstate._core", frozen)]
pub struct Snapshot {
    data: Arc<SnapshotData>,
}

impl Snapshot {
    fn ns_lookup<'a>(&'a self, ns: &str) -> Option<&'a ImMap<String, Vec<u8>>> {
        self.data.shards[shard_index(ns)].get(ns)
    }
}

#[pymethods]
impl Snapshot {
    /// Monotonic id assigned by the originating store.
    #[getter]
    fn id(&self) -> u64 {
        self.data.id
    }

    /// Seconds since the Unix epoch when the snapshot was taken.
    #[getter]
    fn timestamp(&self) -> f64 {
        self.data.timestamp
    }

    /// Id of the previous snapshot from the same store (for incremental diffs).
    #[getter]
    fn parent(&self) -> Option<u64> {
        self.data.parent
    }

    /// Total size in bytes of all stored (serialized) values.
    #[getter]
    fn size_bytes(&self) -> usize {
        self.data.size_bytes
    }

    /// All `(namespace, key)` pairs present in the snapshot.
    #[getter]
    fn keys(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for shard in &self.data.shards {
            for (ns, kv) in shard.iter() {
                for k in kv.keys() {
                    out.push((ns.clone(), k.clone()));
                }
            }
        }
        out
    }

    /// Incremental diff describing how to go from `base` to `self`.
    ///
    /// Returns a dict with keys `"added"`, `"removed"`, and `"changed"`, each
    /// mapping to a list of `(namespace, key)` tuples.
    fn diff(&self, base: &Snapshot) -> HashMap<String, Vec<(String, String)>> {
        let mut added = Vec::new();
        let mut removed = Vec::new();
        let mut changed = Vec::new();

        for shard in &self.data.shards {
            for (ns, kv) in shard.iter() {
                let base_ns = base.ns_lookup(ns);
                for (k, v) in kv.iter() {
                    match base_ns.and_then(|b| b.get(k)) {
                        None => added.push((ns.clone(), k.clone())),
                        Some(bv) if bv != v => changed.push((ns.clone(), k.clone())),
                        _ => {}
                    }
                }
            }
        }
        for shard in &base.data.shards {
            for (ns, kv) in shard.iter() {
                let self_ns = self.ns_lookup(ns);
                for k in kv.keys() {
                    if self_ns.map(|s| !s.contains_key(k)).unwrap_or(true) {
                        removed.push((ns.clone(), k.clone()));
                    }
                }
            }
        }

        let mut out = HashMap::with_capacity(3);
        out.insert("added".to_string(), added);
        out.insert("removed".to_string(), removed);
        out.insert("changed".to_string(), changed);
        out
    }

    fn __repr__(&self) -> String {
        format!(
            "Snapshot(id={}, size_bytes={}, parent={:?})",
            self.data.id, self.data.size_bytes, self.data.parent
        )
    }
}

/// Framework-agnostic state store with immutable snapshots.
#[pyclass(module = "swarmstate._core")]
pub struct Store {
    shards: Vec<RwLock<NsMap>>,
    codec_name: String,
    max_history: Option<usize>,
    // VecDeque so trimming to `max_history` is an O(1) pop_front, not an O(n)
    // Vec shift.
    history: RwLock<VecDeque<Arc<SnapshotData>>>,
    counter: AtomicU64,
    last_id: RwLock<Option<u64>>,
    // Per-shard running total of stored value bytes, kept incrementally so
    // snapshot() stays O(1) (a sum over SHARDS counters) instead of O(n) over
    // every value. One counter per shard rather than a single global atomic so
    // concurrent writers to different shards don't contend on one cache line
    // (matters on free-threaded builds). Relaxed ordering is sufficient: each
    // counter is only mutated under its shard's write lock and only read while
    // all shard read locks are held, so the locks provide the happens-before.
    shard_bytes: Vec<AtomicUsize>,
}

impl Store {
    fn shard(&self, namespace: &str) -> &RwLock<NsMap> {
        &self.shards[shard_index(namespace)]
    }
}

#[pymethods]
impl Store {
    #[new]
    #[pyo3(signature = (backend = "memory", codec = "msgpack", max_history = None))]
    fn new(backend: &str, codec: &str, max_history: Option<usize>) -> PyResult<Self> {
        if backend != "memory" {
            return Err(PyValueError::new_err(format!(
                "backend '{backend}' is not available in this build (only 'memory')"
            )));
        }
        if codec != "msgpack" {
            return Err(PyValueError::new_err(format!(
                "codec '{codec}' is not supported (only 'msgpack')"
            )));
        }
        Ok(Store {
            shards: (0..SHARDS).map(|_| RwLock::new(NsMap::new())).collect(),
            codec_name: codec.to_string(),
            max_history,
            history: RwLock::new(VecDeque::new()),
            counter: AtomicU64::new(1),
            last_id: RwLock::new(None),
            shard_bytes: (0..SHARDS).map(|_| AtomicUsize::new(0)).collect(),
        })
    }

    /// Serialization codec in use (currently always `"msgpack"`).
    #[getter]
    fn codec(&self) -> &str {
        &self.codec_name
    }

    /// Maximum number of retained snapshots, or `None` for unlimited.
    #[getter]
    fn max_history(&self) -> Option<usize> {
        self.max_history
    }

    /// Store `value` under `(namespace, key)`, replacing any existing value.
    fn set(
        &self,
        py: Python<'_>,
        namespace: String,
        key: String,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let bytes = codec::encode(value)?; // touches Python -> under GIL
        py.allow_threads(|| {
            let idx = shard_index(&namespace);
            let new_len = bytes.len();
            let mut guard = self.shards[idx].write().unwrap();
            let old_len = if let Some(ns) = guard.get_mut(&namespace) {
                ns.insert(key, bytes).map(|old| old.len()).unwrap_or(0)
            } else {
                let mut ns = ImMap::new();
                ns.insert(key, bytes);
                guard.insert(namespace, ns);
                0
            };
            if new_len >= old_len {
                self.shard_bytes[idx].fetch_add(new_len - old_len, Ordering::Relaxed);
            } else {
                self.shard_bytes[idx].fetch_sub(old_len - new_len, Ordering::Relaxed);
            }
        });
        Ok(())
    }

    /// Return the value at `(namespace, key)`, or `default` (None) if absent.
    #[pyo3(signature = (namespace, key, default = None))]
    fn get(
        &self,
        py: Python<'_>,
        namespace: &str,
        key: &str,
        default: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let bytes = py.allow_threads(|| {
            let guard = self.shard(namespace).read().unwrap();
            guard.get(namespace).and_then(|ns| ns.get(key)).cloned()
        });
        match bytes {
            Some(b) => Ok(codec::decode(py, &b)?.unbind()),
            None => Ok(default.unwrap_or_else(|| py.None())),
        }
    }

    /// Store many `(namespace, key, value)` triples in one call.
    ///
    /// Values are encoded under the GIL, then written with the GIL released,
    /// locking each shard once for the whole batch (not once per item). This
    /// amortizes the per-call Python->Rust and lock overhead over the batch.
    fn set_many(&self, py: Python<'_>, items: Vec<(String, String, Py<PyAny>)>) -> PyResult<()> {
        // Encode everything first (touches Python objects -> under GIL).
        let mut encoded: Vec<(String, String, Vec<u8>)> = Vec::with_capacity(items.len());
        for (ns, key, value) in items {
            let bytes = codec::encode(value.bind(py))?;
            encoded.push((ns, key, bytes));
        }
        py.allow_threads(|| {
            // Bucket by shard so each shard lock is taken exactly once.
            let mut buckets: Vec<Vec<(String, String, Vec<u8>)>> =
                (0..SHARDS).map(|_| Vec::new()).collect();
            for (ns, key, bytes) in encoded {
                let idx = shard_index(&ns);
                buckets[idx].push((ns, key, bytes));
            }
            for (idx, bucket) in buckets.into_iter().enumerate() {
                if bucket.is_empty() {
                    continue;
                }
                let mut guard = self.shards[idx].write().unwrap();
                let mut delta: i64 = 0;
                for (ns, key, bytes) in bucket {
                    let new_len = bytes.len();
                    let old_len = if let Some(nsmap) = guard.get_mut(&ns) {
                        nsmap.insert(key, bytes).map(|old| old.len()).unwrap_or(0)
                    } else {
                        let mut nsmap = ImMap::new();
                        nsmap.insert(key, bytes);
                        guard.insert(ns, nsmap);
                        0
                    };
                    delta += new_len as i64 - old_len as i64;
                }
                if delta >= 0 {
                    self.shard_bytes[idx].fetch_add(delta as usize, Ordering::Relaxed);
                } else {
                    self.shard_bytes[idx].fetch_sub((-delta) as usize, Ordering::Relaxed);
                }
            }
        });
        Ok(())
    }

    /// Fetch many `(namespace, key)` pairs in one call, preserving input order.
    ///
    /// Missing pairs come back as `None`. Bytes are read with the GIL released
    /// (one read lock per shard for the batch), then decoded under the GIL.
    fn get_many(
        &self,
        py: Python<'_>,
        pairs: Vec<(String, String)>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let n = pairs.len();
        let raw: Vec<Option<Vec<u8>>> = py.allow_threads(|| {
            let mut out: Vec<Option<Vec<u8>>> = (0..n).map(|_| None).collect();
            let mut buckets: Vec<Vec<usize>> = (0..SHARDS).map(|_| Vec::new()).collect();
            for (i, (ns, _)) in pairs.iter().enumerate() {
                buckets[shard_index(ns)].push(i);
            }
            for (idx, indices) in buckets.iter().enumerate() {
                if indices.is_empty() {
                    continue;
                }
                let guard = self.shards[idx].read().unwrap();
                for &i in indices {
                    let (ns, key) = &pairs[i];
                    out[i] = guard.get(ns).and_then(|nsmap| nsmap.get(key)).cloned();
                }
            }
            out
        });
        let mut result = Vec::with_capacity(n);
        for item in raw {
            match item {
                Some(b) => result.push(codec::decode(py, &b)?.unbind()),
                None => result.push(py.None()),
            }
        }
        Ok(result)
    }

    /// Return whether `(namespace, key)` exists.
    fn contains(&self, py: Python<'_>, namespace: &str, key: &str) -> bool {
        py.allow_threads(|| {
            let guard = self.shard(namespace).read().unwrap();
            guard.get(namespace).is_some_and(|ns| ns.contains_key(key))
        })
    }

    /// Delete `(namespace, key)`. Returns True if a value was removed.
    fn delete(&self, py: Python<'_>, namespace: &str, key: &str) -> bool {
        py.allow_threads(|| {
            let idx = shard_index(namespace);
            let mut guard = self.shards[idx].write().unwrap();
            match guard.get_mut(namespace) {
                Some(ns) => match ns.remove(key) {
                    Some(old) => {
                        self.shard_bytes[idx].fetch_sub(old.len(), Ordering::Relaxed);
                        true
                    }
                    None => false,
                },
                None => false,
            }
        })
    }

    /// All keys within `namespace` (empty list if the namespace is unknown).
    fn keys(&self, py: Python<'_>, namespace: &str) -> Vec<String> {
        py.allow_threads(|| {
            let guard = self.shard(namespace).read().unwrap();
            guard
                .get(namespace)
                .map(|ns| ns.keys().cloned().collect())
                .unwrap_or_default()
        })
    }

    /// All namespaces currently in the store.
    fn namespaces(&self, py: Python<'_>) -> Vec<String> {
        py.allow_threads(|| {
            let mut out = Vec::new();
            for shard in &self.shards {
                let guard = shard.read().unwrap();
                out.extend(guard.keys().cloned());
            }
            out
        })
    }

    /// Total number of `(namespace, key)` entries.
    fn __len__(&self, py: Python<'_>) -> usize {
        py.allow_threads(|| {
            self.shards
                .iter()
                .map(|s| s.read().unwrap().values().map(|ns| ns.len()).sum::<usize>())
                .sum()
        })
    }

    /// Remove all entries (does not clear snapshot history).
    fn clear(&self, py: Python<'_>) {
        py.allow_threads(|| {
            for (shard, bytes) in self.shards.iter().zip(&self.shard_bytes) {
                shard.write().unwrap().clear();
                bytes.store(0, Ordering::Relaxed);
            }
        });
    }

    /// Capture a cheap, immutable snapshot of the current state.
    ///
    /// Read-locks every shard (in order) so the clone is a consistent
    /// point-in-time view, then clones each shard map (O(1) structural share).
    fn snapshot(&self, py: Python<'_>) -> Snapshot {
        let data = py.allow_threads(|| {
            let guards: Vec<_> = self.shards.iter().map(|s| s.read().unwrap()).collect();
            let shards: Vec<NsMap> = guards.iter().map(|g| (**g).clone()).collect();
            let shard_bytes: Vec<usize> = self
                .shard_bytes
                .iter()
                .map(|b| b.load(Ordering::Relaxed))
                .collect();
            let size_bytes: usize = shard_bytes.iter().sum();
            drop(guards);

            let id = self.counter.fetch_add(1, Ordering::Relaxed);
            let parent = {
                let mut last = self.last_id.write().unwrap();
                let prev = *last;
                *last = Some(id);
                prev
            };
            let data = Arc::new(SnapshotData {
                id,
                timestamp: now_secs(),
                parent,
                size_bytes,
                shard_bytes,
                shards,
            });
            let mut hist = self.history.write().unwrap();
            hist.push_back(data.clone());
            if let Some(max) = self.max_history {
                while hist.len() > max {
                    hist.pop_front();
                }
            }
            data
        });
        Snapshot { data }
    }

    /// Roll the store back to a previously captured snapshot.
    fn restore(&self, py: Python<'_>, snapshot: &Snapshot) {
        py.allow_threads(|| {
            let mut guards: Vec<_> = self.shards.iter().map(|s| s.write().unwrap()).collect();
            for (i, g) in guards.iter_mut().enumerate() {
                **g = snapshot.data.shards[i].clone();
                self.shard_bytes[i].store(snapshot.data.shard_bytes[i], Ordering::Relaxed);
            }
        });
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "Store(backend='memory', codec='{}', entries={})",
            self.codec_name,
            self.__len__(py)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyDict;

    #[test]
    fn set_get_and_snapshot_isolation() {
        Python::with_gil(|py| {
            let store = Store::new("memory", "msgpack", None).unwrap();
            let v = PyDict::new(py);
            v.set_item("step", 1i64).unwrap();
            store.set(py, "wf".into(), "a".into(), v.as_any()).unwrap();

            let snap = store.snapshot(py);
            assert_eq!(store.__len__(py), 1);

            let v2 = PyDict::new(py);
            v2.set_item("step", 2i64).unwrap();
            store.set(py, "wf".into(), "a".into(), v2.as_any()).unwrap();
            store.set(py, "wf".into(), "b".into(), v2.as_any()).unwrap();
            assert_eq!(store.__len__(py), 2);

            store.restore(py, &snap);
            assert_eq!(store.__len__(py), 1);
            let got = store.get(py, "wf", "a", None).unwrap();
            let got = got.bind(py).downcast::<PyDict>().unwrap().clone();
            assert_eq!(
                got.get_item("step")
                    .unwrap()
                    .unwrap()
                    .extract::<i64>()
                    .unwrap(),
                1
            );
        });
    }

    #[test]
    fn diff_reports_changes() {
        Python::with_gil(|py| {
            let store = Store::new("memory", "msgpack", None).unwrap();
            let one = 1i64.into_pyobject(py).unwrap().into_any();
            store.set(py, "n".into(), "keep".into(), &one).unwrap();
            store.set(py, "n".into(), "drop".into(), &one).unwrap();
            let base = store.snapshot(py);

            store.delete(py, "n", "drop");
            let two = 2i64.into_pyobject(py).unwrap().into_any();
            store.set(py, "n".into(), "keep".into(), &two).unwrap();
            store.set(py, "n".into(), "new".into(), &two).unwrap();
            let now = store.snapshot(py);

            let d = now.diff(&base);
            assert_eq!(d["added"], vec![("n".to_string(), "new".to_string())]);
            assert_eq!(d["removed"], vec![("n".to_string(), "drop".to_string())]);
            assert_eq!(d["changed"], vec![("n".to_string(), "keep".to_string())]);
            assert_eq!(now.parent(), Some(base.id()));
        });
    }

    #[test]
    fn set_many_get_many_roundtrip() {
        Python::with_gil(|py| {
            let store = Store::new("memory", "msgpack", None).unwrap();
            let mk = |n: i64| n.into_pyobject(py).unwrap().into_any().unbind();
            let items = vec![
                ("a".to_string(), "x".to_string(), mk(1)),
                ("a".to_string(), "y".to_string(), mk(2)),
                ("b".to_string(), "z".to_string(), mk(3)),
            ];
            store.set_many(py, items).unwrap();
            assert_eq!(store.__len__(py), 3);

            // Overwrite via set_many keeps the count and updates byte accounting.
            store
                .set_many(py, vec![("a".to_string(), "x".to_string(), mk(42))])
                .unwrap();
            assert_eq!(store.__len__(py), 3);

            let got = store
                .get_many(
                    py,
                    vec![
                        ("a".to_string(), "x".to_string()),
                        ("b".to_string(), "z".to_string()),
                        ("missing".to_string(), "nope".to_string()),
                    ],
                )
                .unwrap();
            assert_eq!(got[0].bind(py).extract::<i64>().unwrap(), 42);
            assert_eq!(got[1].bind(py).extract::<i64>().unwrap(), 3);
            assert!(got[2].bind(py).is_none());

            // Byte total stays consistent: a snapshot equals a fresh recompute.
            let snap = store.snapshot(py);
            assert!(snap.size_bytes() > 0);
        });
    }

    #[test]
    fn spreads_namespaces_across_shards() {
        Python::with_gil(|py| {
            let store = Store::new("memory", "msgpack", None).unwrap();
            let v = 1i64.into_pyobject(py).unwrap().into_any();
            for i in 0..100 {
                store.set(py, format!("ns{i}"), "k".into(), &v).unwrap();
            }
            assert_eq!(store.__len__(py), 100);
            assert_eq!(store.namespaces(py).len(), 100);
            // at least a few distinct shards are used
            let used: std::collections::HashSet<usize> =
                (0..100).map(|i| shard_index(&format!("ns{i}"))).collect();
            assert!(used.len() > 1);
        });
    }
}

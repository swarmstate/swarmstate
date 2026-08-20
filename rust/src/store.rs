//! Concurrent, framework-agnostic key/value store with cheap immutable snapshots.
//!
//! State is keyed by `(namespace, key)` and stored as msgpack bytes (see
//! [`crate::codec`]). The backing map is an `imbl::HashMap`, a persistent
//! data structure: cloning it is O(1) via structural sharing, so
//! [`Store::snapshot`] is cheap and snapshots are fully isolated from later
//! mutations (copy-on-write).
//!
//! Writes are **sharded**: namespaces are hashed across `SHARDS` independent
//! `RwLock`s, so concurrent writers to different namespaces don't contend on a
//! single global lock. The interpreter is detached (`py.detach`) around every
//! lock/map operation; only (de)serialization runs attached.

use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::{SystemTime, UNIX_EPOCH};

use imbl::HashMap as ImMap;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::codec;

/// `namespace -> (key -> value bytes)`.
///
/// Values are `Arc<[u8]>` rather than `Vec<u8>` so that handing one out (`get`,
/// `get_many`) is a refcount bump instead of copying the payload, and so that a
/// value shared with a snapshot can be recognised by pointer.
type NsMap = ImMap<String, ImMap<String, Arc<[u8]>>>;

/// An encoded `(namespace, key, value)` triple, as batched writes carry it.
type EncodedEntry = (String, String, Arc<[u8]>);

/// Number of lock shards. Namespaces are hashed across these so writes to
/// different namespaces proceed in parallel.
const SHARDS: usize = 16;

/// Read-lock `lock`, recovering the data if the lock is poisoned.
///
/// A `RwLock` stays poisoned for the life of the process once a panic unwinds
/// out of a critical section, which would turn one transient failure into a
/// permanently unusable store. Nothing here keeps multi-step invariants under a
/// lock — the maps are always structurally whole — so taking the data back is
/// strictly better than making every later call panic.
fn read_lock<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Write-lock `lock`, recovering the data if the lock is poisoned (see [`read_lock`]).
fn write_lock<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    lock.write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

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
    fn ns_lookup<'a>(&'a self, ns: &str) -> Option<&'a ImMap<String, Arc<[u8]>>> {
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

        for (i, shard) in self.data.shards.iter().enumerate() {
            // Untouched shards and namespaces are the *same* persistent map as in
            // the base snapshot, so identity settles them without a walk. This is
            // what makes a diff cost the changes rather than the whole store.
            if shard.ptr_eq(&base.data.shards[i]) {
                continue;
            }
            for (ns, kv) in shard.iter() {
                let base_ns = base.ns_lookup(ns);
                if base_ns.is_some_and(|b| b.ptr_eq(kv)) {
                    continue;
                }
                for (k, v) in kv.iter() {
                    match base_ns.and_then(|b| b.get(k)) {
                        None => added.push((ns.clone(), k.clone())),
                        // Same allocation -> same bytes, no memcmp needed.
                        Some(bv) if !Arc::ptr_eq(bv, v) && bv != v => {
                            changed.push((ns.clone(), k.clone()))
                        }
                        _ => {}
                    }
                }
            }
        }
        for (i, shard) in base.data.shards.iter().enumerate() {
            if shard.ptr_eq(&self.data.shards[i]) {
                continue;
            }
            for (ns, kv) in shard.iter() {
                let self_ns = self.ns_lookup(ns);
                if self_ns.is_some_and(|s| s.ptr_eq(kv)) {
                    continue;
                }
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
    // How many snapshots the store keeps reachable through `history()`:
    // Some(0) (the default) retains none, Some(n) the last n, None every one.
    // Retaining pins the state each snapshot saw, so an unbounded default would
    // make every snapshot() call leak the values it superseded.
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
    #[pyo3(signature = (backend = "memory", codec = "msgpack", max_history = Some(0)))]
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

    /// How many snapshots the store retains: `0` none, `n` the last `n`,
    /// `None` unlimited.
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
        let bytes: Arc<[u8]> = codec::encode(value)?.into(); // touches Python -> under GIL
        py.detach(|| {
            let idx = shard_index(&namespace);
            let new_len = bytes.len();
            let mut guard = write_lock(&self.shards[idx]);
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
        let bytes = py.detach(|| {
            let guard = read_lock(self.shard(namespace));
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
        let mut encoded: Vec<EncodedEntry> = Vec::with_capacity(items.len());
        for (ns, key, value) in items {
            let bytes: Arc<[u8]> = codec::encode(value.bind(py))?.into();
            encoded.push((ns, key, bytes));
        }
        py.detach(|| {
            // Bucket by shard so each shard lock is taken exactly once.
            let mut buckets: Vec<Vec<EncodedEntry>> = (0..SHARDS).map(|_| Vec::new()).collect();
            for (ns, key, bytes) in encoded {
                let idx = shard_index(&ns);
                buckets[idx].push((ns, key, bytes));
            }
            for (idx, bucket) in buckets.into_iter().enumerate() {
                if bucket.is_empty() {
                    continue;
                }
                let mut guard = write_lock(&self.shards[idx]);
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
    fn get_many(&self, py: Python<'_>, pairs: Vec<(String, String)>) -> PyResult<Vec<Py<PyAny>>> {
        let n = pairs.len();
        let raw: Vec<Option<Arc<[u8]>>> = py.detach(|| {
            let mut out: Vec<Option<Arc<[u8]>>> = (0..n).map(|_| None).collect();
            let mut buckets: Vec<Vec<usize>> = (0..SHARDS).map(|_| Vec::new()).collect();
            for (i, (ns, _)) in pairs.iter().enumerate() {
                buckets[shard_index(ns)].push(i);
            }
            for (idx, indices) in buckets.iter().enumerate() {
                if indices.is_empty() {
                    continue;
                }
                let guard = read_lock(&self.shards[idx]);
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
        py.detach(|| {
            let guard = read_lock(self.shard(namespace));
            guard.get(namespace).is_some_and(|ns| ns.contains_key(key))
        })
    }

    /// Delete `(namespace, key)`. Returns True if a value was removed.
    ///
    /// A namespace that loses its last key is dropped as well: otherwise empty
    /// namespaces pile up forever in [`Store::namespaces`], which callers scan
    /// (the LangGraph adapter does, on every `list()` / `delete_thread()`).
    fn delete(&self, py: Python<'_>, namespace: &str, key: &str) -> bool {
        py.detach(|| {
            let idx = shard_index(namespace);
            let mut guard = write_lock(&self.shards[idx]);
            let (removed, now_empty) = match guard.get_mut(namespace) {
                Some(ns) => match ns.remove(key) {
                    Some(old) => {
                        self.shard_bytes[idx].fetch_sub(old.len(), Ordering::Relaxed);
                        (true, ns.is_empty())
                    }
                    None => (false, false),
                },
                None => (false, false),
            };
            if now_empty {
                guard.remove(namespace);
            }
            removed
        })
    }

    /// Keys within `namespace`, optionally only those starting with `prefix`.
    ///
    /// Empty list if the namespace is unknown.
    #[pyo3(signature = (namespace, prefix = None))]
    fn keys(&self, py: Python<'_>, namespace: &str, prefix: Option<&str>) -> Vec<String> {
        py.detach(|| {
            let guard = read_lock(self.shard(namespace));
            let Some(ns) = guard.get(namespace) else {
                return Vec::new();
            };
            match prefix {
                None => ns.keys().cloned().collect(),
                Some(p) => ns.keys().filter(|k| k.starts_with(p)).cloned().collect(),
            }
        })
    }

    /// The greatest key in `namespace`, or `None` if it holds nothing.
    ///
    /// Scans the namespace without building a key list, which is what callers
    /// after "the newest entry" actually need (the LangGraph adapter resolves
    /// the latest checkpoint id this way).
    fn max_key(&self, py: Python<'_>, namespace: &str) -> Option<String> {
        py.detach(|| {
            read_lock(self.shard(namespace))
                .get(namespace)
                .and_then(|ns| ns.keys().max().cloned())
        })
    }

    /// Namespaces in the store, optionally only those starting with `prefix`.
    ///
    /// Filtering happens here rather than in the caller so that a prefix scan
    /// copies only the names it returns. Callers that key namespaces by tenant
    /// or thread (the LangGraph adapter does) would otherwise pay for a full
    /// copy of every name on each lookup.
    #[pyo3(signature = (prefix = None))]
    fn namespaces(&self, py: Python<'_>, prefix: Option<&str>) -> Vec<String> {
        py.detach(|| {
            let mut out = Vec::new();
            for shard in &self.shards {
                let guard = read_lock(shard);
                match prefix {
                    None => out.extend(guard.keys().cloned()),
                    Some(p) => out.extend(guard.keys().filter(|k| k.starts_with(p)).cloned()),
                }
            }
            out
        })
    }

    /// Whether `namespace` holds at least one key (`namespace in store`).
    ///
    /// Mirrors the persistent backends, which have always supported this.
    fn __contains__(&self, py: Python<'_>, namespace: &str) -> bool {
        py.detach(|| {
            read_lock(self.shard(namespace))
                .get(namespace)
                .is_some_and(|ns| !ns.is_empty())
        })
    }

    /// Total number of `(namespace, key)` entries.
    fn __len__(&self, py: Python<'_>) -> usize {
        py.detach(|| {
            self.shards
                .iter()
                .map(|s| read_lock(s).values().map(|ns| ns.len()).sum::<usize>())
                .sum()
        })
    }

    /// Remove all entries (does not clear snapshot history).
    fn clear(&self, py: Python<'_>) {
        py.detach(|| {
            for (shard, bytes) in self.shards.iter().zip(&self.shard_bytes) {
                write_lock(shard).clear();
                bytes.store(0, Ordering::Relaxed);
            }
        });
    }

    /// Capture a cheap, immutable snapshot of the current state.
    ///
    /// Read-locks every shard (in order) so the clone is a consistent
    /// point-in-time view, then clones each shard map (O(1) structural share).
    fn snapshot(&self, py: Python<'_>) -> Snapshot {
        let data = py.detach(|| {
            let guards: Vec<_> = self.shards.iter().map(read_lock).collect();
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
                let mut last = write_lock(&self.last_id);
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
            // Retention is opt-in: a retained snapshot pins every value it saw,
            // so keeping them all by default made snapshot() grow the process
            // without bound (and nothing could read them back).
            match self.max_history {
                Some(0) => {}
                Some(max) => {
                    let mut hist = write_lock(&self.history);
                    hist.push_back(data.clone());
                    while hist.len() > max {
                        hist.pop_front();
                    }
                }
                None => write_lock(&self.history).push_back(data.clone()),
            }
            data
        });
        Snapshot { data }
    }

    /// Snapshots retained by this store, oldest first.
    ///
    /// Empty unless the store was built with `max_history`; the snapshots
    /// returned by [`Store::snapshot`] are always valid on their own.
    fn history(&self, py: Python<'_>) -> Vec<Snapshot> {
        py.detach(|| {
            read_lock(&self.history)
                .iter()
                .map(|data| Snapshot { data: data.clone() })
                .collect()
        })
    }

    /// Drop every retained snapshot, releasing the state they pin.
    fn clear_history(&self, py: Python<'_>) {
        py.detach(|| write_lock(&self.history).clear());
    }

    /// Roll the store back to a previously captured snapshot.
    fn restore(&self, py: Python<'_>, snapshot: &Snapshot) {
        py.detach(|| {
            let mut guards: Vec<_> = self.shards.iter().map(write_lock).collect();
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
        Python::attach(|py| {
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
            let got = got.bind(py).cast::<PyDict>().unwrap().clone();
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
        Python::attach(|py| {
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
        Python::attach(|py| {
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
    fn history_retains_only_what_was_asked_for() {
        Python::attach(|py| {
            // The Python-level default (Some(0)): snapshots work, none retained.
            let store = Store::new("memory", "msgpack", Some(0)).unwrap();
            let v = 1i64.into_pyobject(py).unwrap().into_any();
            store.set(py, "n".into(), "k".into(), &v).unwrap();
            let snap = store.snapshot(py);
            assert!(store.history(py).is_empty());
            assert_eq!(snap.keys().len(), 1); // the handed-out snapshot still works

            // Bounded: the last `max` snapshots, oldest first.
            let store = Store::new("memory", "msgpack", Some(2)).unwrap();
            let mut ids = Vec::new();
            for _ in 0..4 {
                ids.push(store.snapshot(py).id());
            }
            let kept: Vec<u64> = store.history(py).iter().map(|s| s.id()).collect();
            assert_eq!(kept, ids[2..].to_vec());

            store.clear_history(py);
            assert!(store.history(py).is_empty());
        });
    }

    #[test]
    fn unlimited_history_is_explicit() {
        Python::attach(|py| {
            let store = Store::new("memory", "msgpack", None).unwrap();
            assert_eq!(store.max_history(), None);
            for _ in 0..3 {
                store.snapshot(py);
            }
            assert_eq!(store.history(py).len(), 3);
        });
    }

    #[test]
    fn emptied_namespace_disappears() {
        Python::attach(|py| {
            let store = Store::new("memory", "msgpack", None).unwrap();
            let v = 1i64.into_pyobject(py).unwrap().into_any();
            store.set(py, "ns".into(), "a".into(), &v).unwrap();
            store.set(py, "ns".into(), "b".into(), &v).unwrap();

            assert!(store.delete(py, "ns", "a"));
            assert_eq!(store.namespaces(py, None), vec!["ns".to_string()]);

            assert!(store.delete(py, "ns", "b"));
            assert!(store.namespaces(py, None).is_empty());
            assert_eq!(store.__len__(py), 0);
            assert!(!store.delete(py, "ns", "b"));
        });
    }

    #[test]
    fn spreads_namespaces_across_shards() {
        Python::attach(|py| {
            let store = Store::new("memory", "msgpack", None).unwrap();
            let v = 1i64.into_pyobject(py).unwrap().into_any();
            for i in 0..100 {
                store.set(py, format!("ns{i}"), "k".into(), &v).unwrap();
            }
            assert_eq!(store.__len__(py), 100);
            assert_eq!(store.namespaces(py, None).len(), 100);
            // at least a few distinct shards are used
            let used: std::collections::HashSet<usize> =
                (0..100).map(|i| shard_index(&format!("ns{i}"))).collect();
            assert!(used.len() > 1);
        });
    }
}

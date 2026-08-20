# Contributing to swarmstate

Thanks for taking a look. swarmstate is a Rust core with a thin Python API, so a
contribution usually touches one or the other — rarely both.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin
pip install -e ".[dev,langgraph,disk,redis,postgres,otel]" fakeredis
maturin develop --release        # compile the Rust core; --release or numbers lie
```

`maturin develop` (no `--release`) builds a debug core that is several times slower. Use it
for iteration, never for measuring.

## The checks CI runs

```bash
cargo test                      # Rust core
pytest -q                       # Python API, adapters, backends, conformance
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
ruff check python tests examples benchmarks
ruff format --check python tests examples
mypy                            # strict, our package only
```

All of them have to pass. `mypy` runs in strict mode over `python/swarmstate`, so new code
needs complete annotations.

## Where things live

| Path | What |
| --- | --- |
| `rust/src/store.rs` | the concurrent store, snapshots, diffs |
| `rust/src/codec.rs` | msgpack encode/decode, streaming both ways |
| `rust/src/graph.rs`, `condition.rs` | handoff graph and its condition mini-language |
| `python/swarmstate/integrations/` | LangGraph and CrewAI adapters |
| `python/swarmstate/backends/` | SQLite file, Redis, Postgres stores |
| `python/swarmstate/protocols.py` | the store contract, for type checking |
| `tests/test_store_conformance.py` | the same assertions against every backend |

## Rules that are not negotiable

**The stored format is stable.** Bytes written by a released version must stay readable.
`tests/test_codec.py` compares swarmstate's output byte-for-byte against an independent
msgpack implementation at every width boundary; if a change moves those bytes, it is a
breaking change and does not belong in a minor release.

**Conditions are data, never code.** `HandoffGraph` conditions are parsed into an
expression tree and evaluated in Rust. No `eval`, no dynamic import, no attribute access —
an identifier is only ever a state lookup.

**Recursion needs a bound.** Anything that walks a user-supplied structure recursively must
carry a depth limit. An unbounded walker exhausts the native stack and takes the whole
interpreter with it — an error is always better than a `SIGSEGV`.

**Release the GIL where it is free.** Operations that do not touch Python objects run
inside `py.allow_threads`. Serialization stays under the GIL; lock and map work does not.

**No compiler for users.** `pip install swarmstate` must work from a prebuilt abi3 wheel.
If a change forces users to compile Rust, it is a bug.

## Adding a store backend

Implement the interface in `python/swarmstate/protocols.py`, then add it to the fixture in
`tests/test_store_conformance.py`. The suite is the contract: prefix scans that treat
wildcards literally, `max_key`, snapshot metadata, diffs and restore all have to behave the
same as every other backend. Optional capabilities are advertised, not assumed — e.g. set
`indexed_max_key = True` only if your `max_key` really is index-backed, since the LangGraph
adapter uses that flag to decide whether to keep its own pointer.

## Touching the LangGraph adapter

Verify signatures against the version pinned in `pyproject.toml` before writing code —
`BaseCheckpointSaver` has changed shape across releases. `tests/test_langgraph_contract.py`
drives the interface directly and compares behaviour against `InMemorySaver`; anything the
reference saver does that we do not is a bug in the drop-in.

## Benchmarks

`python benchmarks/run.py` regenerates `results.json` and the charts. Two rules:

1. **Compare like with like.** Durable against durable at a matched fsync policy, in-memory
   against in-memory. An in-memory store measured against a file-backed one is measuring
   the price of persistence.
2. **Report the losses.** Where a reference implementation is faster, the table says so.
   A number that only survives a favourable pairing is worth less than no number.

Record hardware, versions, seed and payload — `run.py` already writes them into
`results.json`. Always regenerate on the machine you are quoting.

## Docs

```bash
pip install -e ".[docs]"
./scripts/build-docs.sh --serve      # http://127.0.0.1:8000
```

The API reference is generated from the installed package and the `_core.pyi` stubs, so a
new public method needs a docstring and a stub entry. `--strict` treats warnings as errors.

## Commits and PRs

Small, verifiable increments with tests. A bug fix comes with the test that would have
caught it — every entry in the `Fixed` section of `CHANGELOG.md` has one.

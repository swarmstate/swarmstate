"""Type stubs for the native ``swarmstate._core`` module (built from Rust).

Kept in sync by hand with ``rust/src/lib.rs``. Extended as milestones land.
"""

__version__: str

def core_version() -> str:
    """Return the version string of the compiled Rust core."""
    ...

"""M0 smoke tests: the package imports and the Rust core is loaded."""

import swarmstate


def test_import_and_version():
    assert isinstance(swarmstate.__version__, str)
    assert swarmstate.__version__ != ""


def test_core_is_native_rust_module():
    # The compiled extension reports the same version as the package.
    assert swarmstate.core_version() == swarmstate.__version__


def test_core_submodule_present():
    from swarmstate import _core

    assert _core.__version__ == swarmstate.__version__

#!/usr/bin/env python3
"""
ceca_lite package wrapper.

Provides a small helper to load the `CRCAAgent` class from the on-disk
implementation file `CRCA.py` (kept as a standalone script for backwards
compatibility). This helper makes it easier for other code to import the
agent in a reproducible way without changing the existing filename.
"""
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

__all__ = ["load_crca_agent", "__version__"]
__version__ = "1.1.0"


def load_crca_agent():
    """
    Load and return the `CRCAAgent` class object from `crca-lite.py`.

    Returns:
        class: The `CRCAAgent` class defined in `CRCA.py`.
    """
    # In this repository layout the implementation file is `CRCA.py`
    module_path = Path(__file__).resolve().parent / "CRCA.py"
    spec = spec_from_file_location("crca.crca_impl", str(module_path))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod.CRCAAgent



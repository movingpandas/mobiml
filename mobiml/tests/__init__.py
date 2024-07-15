import pytest
import importlib


def _importorskip(modname):
    try:
        importlib.import_module(modname)
        has = True
    except ImportError:
        has = False
    func = pytest.mark.skipif(not has, reason=f"requires {modname}")
    return has, func


has_pymeos, requires_pymeos = _importorskip("pymeos")

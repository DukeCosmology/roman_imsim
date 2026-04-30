import importlib
import pkgutil

import roman_imsim


def test_all_modules_import():
    """Recursively try importing all submodules in roman_imsim."""
    package = roman_imsim
    errors = []

    for loader, module_name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        try:
            importlib.import_module(module_name)
        except Exception as e:
            errors.append((module_name, repr(e)))

    assert not errors, "Failed to import modules:\n" + "\n".join(f"{m}: {e}" for m, e in errors)

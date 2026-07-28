"""CLI entry-point package (V037-006).

Each public module delegates to an implementation shipped under
``daph_learning.cli.commands``.  The implementations are included in wheels,
so console commands work for ordinary and editable installs.

After ``pip install -e .``, the entry points are available as:

- ``daph-autolearn``
- ``daph-evaluate-routes``
- ``daph-build-oracles``
- ``daph-tune-steering``
- ``daph-random-control``
"""

from __future__ import annotations

from importlib import import_module

_SCRIPT_MODULES = {
    "autolearn.py": "daph_learning.cli.commands.autolearn",
    "evaluate_routes.py": "daph_learning.cli.commands.evaluate_routes",
    "build_empirical_oracles.py": "daph_learning.cli.commands.build_oracles",
    "tune_steering.py": "daph_learning.cli.commands.tune_steering",
    "random_direction_control.py": "daph_learning.cli.commands.random_control",
}


def _load_script_main(script_name: str):
    """Resolve a legacy script filename to its wheel-safe command function."""
    module_name = _SCRIPT_MODULES.get(script_name)
    if module_name is None:
        raise ModuleNotFoundError(
            f"DAPH command implementation not found for {script_name!r}"
        )
    return import_module(module_name).main

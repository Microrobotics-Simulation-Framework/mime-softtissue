"""The package imports itself with relative imports, held by reading its source, and shown able to fail.

Why: a plugin that may one day move into MIME's own source tree (`mime_softtissue` to `mime.softtissue`) should move as a
directory. Every absolute self-import (`from mime_softtissue.engine import material`) would then be a line to rewrite, and a
missed one imports the old distribution. Relative imports (`from ..engine import material`) survive the move unchanged.

This reads every module under `src/mime_softtissue/` (with `ast`, never importing it) and fails on an import of the
package's own top-level name: `import mime_softtissue...`, `from mime_softtissue... import`, and a constant-string
`importlib.import_module("mime_softtissue...")` or `__import__(...)`. Tests, scripts, examples and benchmarks import the package
from outside, as a user does, and are not read. The second test is the control: each form, planted, must be found.

What it cannot see: a computed module name, an `exec` or `eval` of source text, and a string that something else imports.
"""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "src" / "mime_softtissue"
NAME = PACKAGE.name


def is_self(name: str) -> bool:
    return name.split(".")[0] == NAME


def dynamic_import_name(node: ast.Call) -> str | None:
    """The constant string a call of `importlib.import_module` or `__import__` names, else `None`."""
    function = node.func
    called = function.attr if isinstance(function, ast.Attribute) else function.id if isinstance(function, ast.Name) else ""
    if called not in ("import_module", "__import__") or not node.args:
        return None
    first = node.args[0]
    return first.value if isinstance(first, ast.Constant) and isinstance(first.value, str) else None


def absolute_self_imports(source: str) -> list[str]:
    """Every module a source imports by this package's own absolute name (relative imports have `level > 0` and pass)."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names if is_self(alias.name)]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and is_self(node.module or ""):
            found.append(node.module or "")
        elif isinstance(node, ast.Call):
            name = dynamic_import_name(node)
            if name is not None and is_self(name):
                found.append(name)
    return found


def test_the_package_imports_itself_relatively():
    modules = sorted(PACKAGE.rglob("*.py"))
    assert modules, f"no module found under {PACKAGE}: the test would pass on nothing"
    held = {str(path.relative_to(PACKAGE)): absolute_self_imports(path.read_text(encoding="utf-8")) for path in modules}
    assert {path: found for path, found in held.items() if found} == {}


def test_a_planted_absolute_self_import_is_found():
    """The control: each way to import the package by its own name that this test sees is found, every time."""
    for planted in (f"import {NAME}", f"import {NAME}.engine", f"import {NAME}.engine as engine",
                    f"from {NAME} import engine", f"from {NAME}.engine import material",
                    f"def f():\n    from {NAME}.engine import material",
                    f'import importlib\nimportlib.import_module("{NAME}.engine")',
                    f'from importlib import import_module\nimport_module("{NAME}")', f'__import__("{NAME}.engine")'):
        assert absolute_self_imports(planted), f"{planted!r} was not found"
    for honest in ("from . import material", "from .. import engine", "from ..engine import material",
                   "import jax\nfrom mime.units import N", f"import {NAME}_other", f'f("{NAME}")'):
        assert not absolute_self_imports(honest), f"{honest!r} was flagged"


def test_a_computed_name_is_the_limit_of_this_test():
    assert absolute_self_imports('import importlib\nimportlib.import_module("mime_" + "softtissue")') == []

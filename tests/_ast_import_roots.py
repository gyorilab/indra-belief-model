import ast
from pathlib import Path


def import_roots(path: Path) -> tuple[set[str], set[str]]:
    tree = ast.parse(path.read_text())
    roots = set()
    relative_targets = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                roots.add(node.module.split(".", 1)[0])
            elif node.module:  # from .pkg import x
                relative_targets.add(node.module.split(".", 1)[0])
            else:  # from . import x, y
                relative_targets.update(a.name.split(".", 1)[0] for a in node.names)
    return roots, relative_targets

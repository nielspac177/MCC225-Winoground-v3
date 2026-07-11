"""Fallback notebook executor that runs code cells in a single in-process
namespace (venv Python) — no ZMQ Jupyter kernel. Captures stdout and the
Jupyter-style repr of a trailing expression, saves outputs back into the .ipynb,
and halts on the first error (like `nbconvert --execute`). The notebook writes
its real evidence to files (outputs/, reports/), so this reliably reproduces the
experiment even when a Jupyter kernel is flaky.

Usage: python scripts/run_notebook_inproc.py notebooks/Cuaderno14_MCC225_resuelto.ipynb
"""
from __future__ import annotations

import ast
import contextlib
import io
import sys
from pathlib import Path

import nbformat

import matplotlib
matplotlib.use("Agg")  # headless


def run(path: Path) -> None:
    nb = nbformat.read(path, as_version=4)
    ns: dict = {"__name__": "__main__"}
    count = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = cell.source
        if not src.strip():
            cell.outputs = []
            continue
        count += 1
        outputs = []
        buf = io.StringIO()
        try:
            tree = ast.parse(src)
            last_expr = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last_expr = tree.body.pop()
            mod = compile(ast.Module(body=tree.body, type_ignores=[]), "<cell>", "exec")
            with contextlib.redirect_stdout(buf):
                exec(mod, ns)
                result = None
                if last_expr is not None:
                    result = eval(  # noqa: S307 - trusted notebook code
                        compile(ast.Expression(last_expr.value), "<cell>", "eval"), ns
                    )
        except Exception:  # halt-on-error, like nbconvert
            text = buf.getvalue()
            if text:
                outputs.append(nbformat.v4.new_output("stream", name="stdout", text=text))
            import traceback
            tb = traceback.format_exc()
            outputs.append(nbformat.v4.new_output(
                "error", ename="Error", evalue="cell failed", traceback=tb.splitlines()))
            cell.outputs = outputs
            cell.execution_count = count
            nbformat.write(nb, path)
            print(f"\nCELL {count} FAILED:\n{tb}", file=sys.stderr)
            sys.exit(1)

        text = buf.getvalue()
        if text:
            outputs.append(nbformat.v4.new_output("stream", name="stdout", text=text))
        if last_expr is not None and result is not None:
            outputs.append(nbformat.v4.new_output(
                "execute_result", data={"text/plain": repr(result)}, execution_count=count))
        cell.outputs = outputs
        cell.execution_count = count
        print(f"[cell {count}] ok", flush=True)

    nbformat.write(nb, path)
    print(f"EXECUTED {count} code cells -> {path}", flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]))

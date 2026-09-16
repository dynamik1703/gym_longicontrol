import os
from pathlib import Path

import pytest

pytest.importorskip("stable_baselines3")
nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("working_directory", [ROOT, ROOT / "examples"])
def test_notebook_runs_top_to_bottom(working_directory, tmp_path, monkeypatch):
    # Use the current test environment's Python, not a user's global kernel.
    import sys

    from jupyter_client.kernelspec import KernelSpec, KernelSpecManager
    from jupyter_client.manager import KernelManager

    class TestKernelSpecManager(KernelSpecManager):
        def get_kernel_spec(self, kernel_name):
            return KernelSpec(
                argv=[
                    sys.executable,
                    "-m",
                    "ipykernel_launcher",
                    "-f",
                    "{connection_file}",
                ],
                display_name="Test Python",
                language="python",
            )

    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("IPYTHONDIR", str(tmp_path / "ipython"))
    notebook = nbformat.read(ROOT / "examples/quickstart.ipynb", as_version=4)
    nbformat.validate(notebook)
    assert all(not cell.get("outputs") for cell in notebook.cells)
    manager = KernelManager(kernel_spec_manager=TestKernelSpecManager())
    client = nbclient.NotebookClient(
        notebook,
        km=manager,
        timeout=120,
        resources={"metadata": {"path": os.fspath(working_directory)}},
    )
    # We supply the kernel manager, so also explicitly own kernel cleanup.
    try:
        executed = client.execute()
        cells = [cell for cell in executed.cells if cell.cell_type == "code"]
        assert all(cell.execution_count is not None for cell in cells)
        assert not any(
            output.output_type == "error" for cell in cells for output in cell.outputs
        )
        assert any(
            "image/png" in output.get("data", {})
            for cell in cells
            for output in cell.outputs
        )
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        manager.cleanup_resources()

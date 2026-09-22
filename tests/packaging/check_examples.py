"""Check that source releases include the demo, while wheels stay lightweight."""

import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


def main(directory):
    directory = Path(directory)
    sources = list(directory.glob("*.tar.gz"))
    wheels = list(directory.glob("*.whl"))
    if len(sources) != 1 or len(wheels) != 1:
        raise AssertionError("Expected exactly one source distribution and one wheel")
    required = {
        "examples/__init__.py",
        "examples/README.md",
        "examples/sb3_quickstart.py",
        "examples/quickstart.ipynb",
        "examples/models/sac_demo/model.zip",
        "examples/models/sac_demo/metadata.json",
        "examples/models/sac_demo/evaluation.json",
        "examples/models/sac_demo/README.md",
    }
    with tarfile.open(sources[0]) as archive:
        names = {
            str(PurePosixPath(*PurePosixPath(member.name).parts[1:]))
            for member in archive.getmembers()
        }
    if not required <= names:
        raise AssertionError(
            f"Missing example files in source release: {required - names}"
        )
    with zipfile.ZipFile(wheels[0]) as archive:
        if any(name.startswith("examples/") for name in archive.namelist()):
            raise AssertionError("Examples must not be installed by the core wheel")
    print("Source examples and lightweight wheel verified")


if __name__ == "__main__":
    main(sys.argv[1])

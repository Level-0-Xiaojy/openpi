# `pyav` (workspace shim)

Hugging Face `lerobot` pins a dependency on the PyPI distribution name `pyav`,
but PyAV is published as `av`.

This package exists only to make `uv`/`pip` resolution succeed: installing
`pyav` (this workspace package) will install `av`, which provides the `av`
Python module.

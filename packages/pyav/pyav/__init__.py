"""Compatibility meta-package.

LeRobot declares a dependency on the distribution name `pyav`, but PyAV is
published on PyPI as `av`. This workspace package exists so dependency
resolvers can satisfy `pyav` while installing `av`.
"""

__all__: list[str] = []

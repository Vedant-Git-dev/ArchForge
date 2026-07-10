"""ArchForge package."""

# Importing the package wires up the archforge.* logging tree with a
# NullHandler, so the library is silent unless the application calls
# configure_logging(). Kept here so the handler exists even for callers
# that grab archforge.* submodules without touching archforge.logging.
from . import logging as _logging  # noqa: F401

__version__ = "0.1.0"

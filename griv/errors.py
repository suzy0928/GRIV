"""Shared exceptions."""


class ValidityError(RuntimeError):
    """A validity check failed; the run is marked INVALID and its outputs must not be used."""

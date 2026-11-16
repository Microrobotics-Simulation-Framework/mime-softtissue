"""One quantity's number in a unit: ``Quantity.to`` returns ``Values`` (a number, a tuple or an array), and a scalar
setting's is the number (a seam: MIME has no typed scalar read)."""

from mime.units import Quantity, Unit


def scalar(quantity: Quantity, unit: Unit | str) -> float:
    """``quantity`` in ``unit``, one number (a tuple or an array is refused: it is not one)."""
    held = quantity.to(unit)
    if not isinstance(held, (int, float)):
        raise TypeError(f"{quantity} is not one number in {unit}")
    return float(held)

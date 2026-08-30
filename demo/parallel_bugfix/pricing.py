"""Pricing helpers with one deliberate bug for the Code Mode demo."""


def discounted_price(price: float, discount: float) -> float:
    return price + discount  # BUG: the discount must be subtracted

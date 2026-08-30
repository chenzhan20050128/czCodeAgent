"""Text helpers with one deliberate bug for the Code Mode demo."""


def slugify(value: str) -> str:
    return value.replace(" ", "-")  # BUG: slugs must be lowercase

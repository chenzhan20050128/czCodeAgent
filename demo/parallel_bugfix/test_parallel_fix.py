"""Both independent modules must be fixed before this suite passes."""

import unittest

from pricing import discounted_price
from text import slugify


class ParallelFixTests(unittest.TestCase):
    def test_discounted_price_subtracts_discount(self) -> None:
        self.assertEqual(discounted_price(100, 15), 85)

    def test_slugify_normalizes_case_and_spaces(self) -> None:
        self.assertEqual(slugify("Hello Agent"), "hello-agent")


if __name__ == "__main__":
    unittest.main()

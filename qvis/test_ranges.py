import unittest

from qvis.ranges import Ranges


class TestRanges(unittest.TestCase):
    def test_add(self):
        ranges = Ranges([[1, 2]])
        ranges.add([4, 5])
        self.assertEqual(len(ranges.inner), 2)
        self.assertEqual(ranges.inner[0], range(1, 3))
        self.assertEqual(ranges.inner[1], range(4, 6))

    def test_add_adjacent(self):
        ranges = Ranges([[1, 2]])
        ranges.add([3, 4])
        self.assertEqual(len(ranges.inner), 1)
        self.assertEqual(ranges.inner[0], range(1, 5))

    def test_add_overlap(self):
        ranges = Ranges([[1, 2]])
        ranges.add([2, 4])
        self.assertEqual(len(ranges.inner), 1)
        self.assertEqual(ranges.inner[0], range(1, 5))

    def test_add_enclosing(self):
        ranges = Ranges([[1, 2]])
        ranges.add([0, 4])
        self.assertEqual(len(ranges.inner), 1)
        self.assertEqual(ranges.inner[0], range(0, 5))


if __name__ == '__main__':
    unittest.main()

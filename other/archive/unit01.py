import unittest
from args import my_sum

class MySumTestCase(unittest.TestCase):
    def test_1(self):
        self.assertEqual(my_sum(2, 3), 5), "my_sum failed with 2 ints"

    def test_2(self):
        self.assertEqual(my_sum(), 0), "my_sum failed with 0"

x = MySumTestCase()
x.test_1()
x.test_2()
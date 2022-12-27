import numpy as np
import unittest

from . import hash

Hash = hash.Hash


class TestHash(unittest.TestCase):
    def test_initialize_is_combinaison_removable_1(self):
        hash = Hash()
        hash.initialize()
        expected = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 0, 1, 1, 1, 1, 0, 0, 1, 1], dtype=np.uint8)
        result = hash.is_combinaison_removable[:20]
        self.assertTrue((result == expected).all())

    def test_initialize_is_combinaison_removable_2(self):
        hash = Hash()
        hash.initialize()
        result = hash.is_combinaison_removable
        self.assertEqual(result.size, 46656)

    def test_index_available_combinaison_1(self):
        hash = Hash()
        hash.initialize()
        expected = np.array([23370, 23400, 23544, 23545, 23550, 23580, 23760, 24624, 24625, 24630,
                            24660, 24840, 25920, 31104, 31105, 31110, 31140, 31320, 32400, 38880], dtype=np.int32)
        result = hash.index_available_combinaison[-20:]
        self.assertTrue((result == expected).all())

    def test_index_available_combinaison_2(self):
        hash = Hash()
        hash.initialize()
        result = hash.index_available_combinaison
        self.assertEqual(result.size, 462)

    def test_combinaison_remove_1(self):
        hash = Hash()
        hash.initialize()
        expected = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 11, 12, 13, 14, 0,
                             0, 15, 16], dtype=np.int32)
        result = hash.combinaison_remove[:20]
        self.assertTrue((result == expected).all())

    def test_combinaison_remove_2(self):
        hash = Hash()
        hash.initialize()
        result = hash.combinaison_remove
        self.assertEqual(result.size, 46656)

    def test_determine_all_index_available_moves(self):
        hash = Hash()
        hash.initialize()
        test_cases = [
            (np.array([0, 2, 1, 0, 0, 0]), np.array([1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.], dtype=np.float32)),
            (np.array([0, 0, 0, 0, 0, 0]), np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32)),
            (np.array([1, 1, 1, 1, 1, 1]), np.array([1., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32))
        ]
        for x, expected in test_cases:
            result = hash.determine_all_index_available_moves(x)
            self.assertTrue((result[:17] == expected).all())

    def test_hash_function(self):
        hash = Hash()
        hash.initialize()
        x =  np.array([[0, 0, 1, 0, 1, 3], [0, 1, 0, 0, 1, 0]])
        result = hash.hash_function(x)
        expected = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=np.float32)
        self.assertTrue((result[:, :17] == expected).all())

    def test_representation(self):
        hash = Hash()
        hash.initialize()
        test_cases = [(0, [0]), (4, [4]), (15, [3, 2]), (100, [4, 4, 2]), (40000, [4, 0, 1, 5, 0, 5])]
        for x, expected in test_cases:
            result = hash.representation(x)
            self.assertEqual(result, expected)

    def test_reverse_hash_function(self):
        hash = Hash()
        hash.initialize()
        test_cases = [
            (0, np.array([0, 0, 0, 0, 0, 0], dtype=np.int32)), 
            (10, np.array([4, 1, 0, 0, 0, 0], dtype=np.int32)), 
            (53, np.array([1, 0, 4, 0, 0, 0], dtype=np.int32)), 
            (461, np.array([0, 0, 0, 0, 0, 5], dtype=np.int32)),             
        ]
        for x, expected in test_cases:
            result = hash.reverse_hash_function(x)
            self.assertTrue((result == expected).all())
            
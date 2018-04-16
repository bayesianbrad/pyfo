#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  17:59
Date created:  16/03/2018

License: MIT
'''
import unittest
import sys
import math
import torch
import time
import numpy as np

from pyfo.utils.core import VariableCast


class test_variable_cast(unittest.TestCase):
    def test_float_size_is_one(self):
        self.assertEqual(VariableCast(1.23).size(0), 1, msg='size does not match')
    def test_row_list_size_is_three(self):
        self.assertEqual(VariableCast([1,2,3].size(0), 3, msg='size does not match'))
    def test_column_list_size_is_three(self):
        self.assertEqual(VariableCast([[1],[2],[4]]).size(0), 3, msg='size does not match')
    def test_row_nparray_size_is_three(self):
        self.assertEqual(VariableCast(np.array([1,2,5])).size(0), 3, msg='size does not match')
    def test_column_nparray_size_is_three(self):
        self.assertEqual(VariableCast(np.array([[1],[2],[5]])).size(0), 3, msg='size does not match')
    def test_float_tensor_element_size_is_one(self):
        self.assertEqual(VariableCast(torch.FloatTensor([1])).size(0), 1, msg='size does not match')
    @unittest.skip(reason='Skipped')
    def test_row_float_tensor_size_is_three(self):
        self.assertEqual(VariableCast(torch.FloatTensor([1,2,3])).size(0), 3, msg='size does not match')
    def test_matrix_of_list_is_size_three_by_three(self):
        self.assertEqual(VariableCast([[1,2,4],[2,3,4],[3,6,7]]).size(0),VariableCast([[1,2,4],[2,3,4],[3,6,7]]).size(1), msg='sizes do not match')
    def test_matrix_numpy_is_size_three_by_three(self):
        self.assertEqual(VariableCast(np.array([[1,2,4],[2,3,4],[3,6,7]])).size(0), VariableCast(np.array([[1,2,4],[2,3,4],[3,6,7]])).size(1), msg='sizes do not match')
    @unittest.skip(reason='To fix')
    def test_one_nparray_element_has_size_one(self):
        self.assertEqual(VariableCast(np.random.randn(3)).size(0), 1, msg='size does not match')
    def test_size_of_inbuilt_random_nparrays(self):
        self.assertEqual(VariableCast(np.random.randn(3,1)),VariableCast(np.random.randn(1,3)), msg='size does not match')

if __name__ == '__main__':
    tests = []
    tests.append('test_variable_cast')

    time_start = time.time()
    success = unittest.main(exit=False,verbosity=2)
    print('\nDuration             : {}'.format(time.time() - time_start))
    print('Models run           : {}'.format(' '.join(tests)))
    print('\nInference tests complete')
    sys.exit(0 if success else 1)
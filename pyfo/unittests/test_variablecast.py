#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  12:28
Date created:  08/01/2018

License: MIT
'''
# terminal command for specifying jupyter port number jupyter notebook --ip=0.0.0.0 --port=8880

from pyfo.utils.core import VariableCast
import unittest
import torch
import numpy as np

class test_variable_cast(unittest.TestCase)
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
    def test_row_float_tensor_size_is_three(self):
        self.assertEqual(VariableCast(torch.FloatTensor([1,2,3])).size(0), 3, msg='size does not match')
    def test_matrix_of_list_is_size_three_by_three(self):
        self.assertEqual(VariableCast([[1,2,4],[2,3,4],[3,6,7]]).size(0),VariableCast([[1,2,4],[2,3,4],[3,6,7]]).size(1), msg='sizes do not match')
    def test_matrix_numpy_is_size_three_by_three(self):
        self.assertEqual(VariableCast(np.array([[1,2,4],[2,3,4],[3,6,7]])).size(0), VariableCast(np.array([[1,2,4],[2,3,4],[3,6,7]])).size(1), msg='sizes do not match')
    def test_one_nparray_element_has_size_one(self):
        self.assertEqual(VariableCast(np.random.randn(3)).size(0), 1, msg='size does not match')
    def test_size_of_inbuilt_random_nparrays(self):
        self.assertEqual(VariableCast(np.random.randn(3,1)),VariableCast(np.random.randn(1,3)), msg='size does not match')

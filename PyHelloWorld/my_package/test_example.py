'''
Created on 26-Mar-2017

@author: aniron
'''
import unittest
from my_package.example import TestClass

class Test(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def test_one(self):
        t = TestClass()
        assert t.main_test() == 'Hello world!'


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_one']
    unittest.main()
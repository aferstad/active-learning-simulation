import unittest

from als import ALS
from input.heart_import import get_heart_data



class TestAls(unittest.TestCase):
    def setUp(self):
        data = get_heart_data()

        self.als = ALS(unsplit_data=data,
                       learning_method='random')

class TestInit(TestAls):
    def test_learning_method(self):
        self.assertEqual(self.als.learning_method, 'random')



    def test_partitions(self):
        self.assertEqual(str(self.als.get_labeled_data().shape), '(50, 19)')


    def test_run_experiment(self):
        self.als.run_experiment()
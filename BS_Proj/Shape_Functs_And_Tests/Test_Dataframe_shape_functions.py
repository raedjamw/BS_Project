import unittest
import pandas as pd


class test_Dataframe_shape(unittest.TestCase):

    def test_lookback(self):
        """

        check if the lookback function is the right shape

        """
        from Dataframe_shape_functions import lookback

        # setup
        df = pd.DataFrame({
            'col_a': ['a1', 'a2','a3','a4'],
            'col_b': ['b1', 'b2','b3' ,'b4'],
        })

        # call function
        actual = lookback(df, 4).shape[1]

        # set expectations
        expected = pd.DataFrame({
            'col_a': ['a1'],
            'col_b': ['b1'],
            'col_a-1': ['a2'],
            'col_b-1': ['b2'],
            'col_a-2': ['a3'],
            'col_b-2': ['b3'],
            'col_a-3': ['a4'],
            'col_b-3': ['b4'],
        })

        self.assertEqual(actual, expected.shape[1])

# Programmatically building up the TestSuite from the test_Dataframes_shape class
run_tests = unittest.TestLoader().loadTestsFromTestCase(test_Dataframe_shape)
# call the TestRunner with the verbosity 2
unittest.TextTestRunner(verbosity=2).run(run_tests)

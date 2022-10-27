import unittest
from functions import duration_to_str

class TestFunctions(unittest.TestCase):

    def test_converion_duration_to_str(self):
        """Check the conversion of duration to str"""
        self.assertEqual(duration_to_str('RKJP,180*200*100*120********'),
                         'R^K^J#P#')
        self.assertEqual(duration_to_str('RN,100*200****'),
                         'R#N^')
        self.assertTrue(type(duration_to_str('RKJP,180*200*100*120********')), str)

if __name__=='__main__':
    unittest.main()

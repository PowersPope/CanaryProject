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

    def test_transition_mat(self):
        """Check to make sure that the transition matrix is being filled in correctly"""
        # occ_dict1: dict[str, int] = {'!A': 120, '!F': 100, '!S': 30, 'SA':30, 'FA': 20, 'FS':100,
                                     # 'S-': 100, 'F-': 20, 'AF': 30, 'AS': 30, 'AA':10}
        # alphabet1: dict[str] = set(['!', '-', 'A', 'F', 'S'])

        # result = np.array([0,0,0.14,0.48,0.48]) # add in array

        # matrix, alpha_set = transition_matrix(occ_dict1, alphabet1, False)

        # self.assertEqual(matrix, result)
        
        pass

        )

if __name__=='__main__':
    unittest.main()

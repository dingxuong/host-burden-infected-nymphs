import unittest
from fractions import Fraction
from lyme_paper1.exact_theory import counterexample

class SameMomentProof(unittest.TestCase):
    def test_same_moments_opposite_infection_directions(self):
        a,b=counterexample()
        self.assertEqual([(r["mean"],r["variance"],r["third_central_moment"]) for r in [a,b]],[("6","9","0")]*2)
        self.assertEqual(Fraction(a["R_N_exact"]), Fraction(232,247))
        self.assertEqual(Fraction(a["R_I_exact"]), Fraction(60224,61009))
        self.assertEqual(Fraction(b["R_N_exact"]), Fraction(41,44))
        self.assertEqual(Fraction(b["R_I_exact"]), Fraction(491,484))
        self.assertLess(a["R_I"],1); self.assertGreater(b["R_I"],1)
        self.assertLess(a["R_N"],1); self.assertLess(b["R_N"],1)

if __name__=="__main__": unittest.main()

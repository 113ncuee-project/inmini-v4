"""Measured episode termination, kept separate from physical reward/cost."""
import unittest

from full_system_dse import episode_cost_cut


class EpisodeCostTests(unittest.TestCase):
    def test_off_preserves_even_large_exploratory_worsening(self):
        self.assertFalse(episode_cost_cut(100.,.05,None))

    def test_boundary_is_a_fraction_of_verified_elite(self):
        self.assertFalse(episode_cost_cut(.0625,.05,.25))
        self.assertTrue(episode_cost_cut(.0626,.05,.25))
        self.assertFalse(episode_cost_cut(.049,.05,.25))

    def test_scale_independent_and_zero_is_greedy_boundary(self):
        for scale in (.001,1.,1000.):
            self.assertTrue(episode_cost_cut(3*scale,scale,.25))
            self.assertFalse(episode_cost_cut(scale,scale,0.))
            self.assertTrue(episode_cost_cut(1.001*scale,scale,0.))

    def test_unfinished_observation_is_not_a_measured_bad_design(self):
        self.assertFalse(episode_cost_cut(None,.05,.25))


if __name__=='__main__':unittest.main()

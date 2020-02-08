"""Unit Test fir Regression Enrichment Surface

This file implements the unit test cases for the functions and classes in
Regression Enrichment Surface module.

Example:
    python -m unittest

TODO:

"""
import unittest
import numpy as np
from regression_enrichment_surface.regression_enrichment_surface import \
    RegressionEnrichmentSurface


class TestRegressionEnrichmentSurface(unittest.TestCase):

    def test_get_enrichment(self):
        self.assertTrue(True)

    def test_get_enrichment_grid(self):
        self.assertTrue(True)

    def test_get_enrichment_grids(self):
        """ Verify the implementation of get_enrichment_grids method for
        RegressionEnrichmentSurface class.

        The verification process will also check the implementation of
        get_enrichment and get_enrichment_grid in the module, as they are
        being used in the class for computing RES with stratification.

        :return: None
        """

        # fix the random seed and therefore test results
        np.random.seed(0)
        _size = (100_000,)

        _y_true = np.random.uniform(low=0.0, high=1.0, size=_size)
        _y_pred = _y_true + np.random.uniform(low=-0.2, high=0.2, size=_size)
        _y_labl = np.random.randint(low=0, high=3, size=_size)

        _res = RegressionEnrichmentSurface()
        _grids = _res.get_enrichment_grids(
            y_true=_y_true,
            y_pred=_y_pred,
            stratified_on=_y_labl,
        )
        _grids = np.array(_grids)
        # np.save(file='./test_enrichment_grids', arr=_grids)

        try:
            _ref_grids = np.load(file='./tests/test_enrichment_grids.npy')
        except FileNotFoundError:
            _ref_grids = np.load(file='./test_enrichment_grids.npy')

        self.assertTrue((_grids == _ref_grids).all())

    def test_plot_enrichment_grids(self):
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()

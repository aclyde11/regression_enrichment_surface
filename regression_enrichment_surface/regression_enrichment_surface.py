"""Regression Enrichment Surface

This module implements the (RES) regression enrichment surface plot.
Please refer to the paper "Regression Enrichment Surfaces: A Simple Analysis
Technique for Virtual Screening Models" by Austin Clyde for more details.

Example:
    TODO: usage examples for both stratified and non-stratified cases

TODO:
    * Simple examples for demonstration and test purposes
    * Performance optimization for the iterative computation of RES

"""
import warnings
import numpy as np
import matplotlib.pyplot as plt


def get_enrichment(
    ranked_indexes_true,
    ranked_indexes_pred,
    cutoff_true,
    cutoff_pred,
):
    """Calculates the enrichment with true and predicted ranking and cutoffs.

    This function is the implementation of the equation 10 in the paper
    "Regression Enrichment Surfaces: A Simple Analysis Technique for Virtual
    Screening Models" by Austin Clyde.
    Specifically, it returns the number of common members of true and
    predicted ranked indexes within two cutoffs separately, normalized by the
    theoretically maximum possible common members with given cutoffs.
    Roughly speaking, the enrichment represents the similarity between two
    rankings with different cutoffs in the scale of [0, 1].

    :param ranked_indexes_true: the true/target ranked list of all the
        indexes of drug candidates
    :type ranked_indexes_true: list, tuple, or other sequence of int
    :param ranked_indexes_pred: the predicted ranked list of all the indexes
        of drug candidates
    :type ranked_indexes_pred: list, tuple, or other sequence of int
    :param cutoff_true: cutoff percentage for the true/target ranked list (
        from the top or index 0)
    :type cutoff_true: float
    :param cutoff_pred: cutoff percentage for the predicted ranked list (
        from the top or index 0)
    :type cutoff_true: float

    :return: enrichment score indicating the similarity between two ranking
        orders with given cutoffs for both
    :rtype: float
    """

    # might use some sanity check here? Specifically, check if
    # * set(ranked_indexes_true) == set(ranked_indexes_pred)
    # * 0 <= cutoff_* <= 1

    _num_indexes = len(ranked_indexes_true)

    # "included" means in the top <cutoff> percentage according to the ranking
    _included_indexes_true = set(
        ranked_indexes_true[:int(cutoff_true * _num_indexes)])
    _included_indexes_pred = set(
        ranked_indexes_pred[:int(cutoff_pred * _num_indexes)])

    _num_jointly_included_indexes = len(
        _included_indexes_true.intersection(_included_indexes_pred))

    # consider that the predicted ranking is the same as the target/true
    # one, then the maximum number of jointly included indexes in both true
    # and predicted rankings is min(cutoffs) * num_indexes
    _max_num_jointly_included_indexes = int(
        min(cutoff_true, cutoff_pred) * _num_indexes)

    return _num_jointly_included_indexes / _max_num_jointly_included_indexes


def get_enrichment_grid(
    y_true,
    y_pred,
    num_samples_per_axis,
    logspace_start=-3,
    logspace_stop=0,
    logspace_base=10,
    descending_ranking_order=False,
):
    """Generates a 2D square grid of enrichment scores.

    This function will first rank the true values and the predictions with
    reverse option, which will then be used in the calculation of enrichment
    scores repeatedly.
    A 2D square grid is generated based on exponential sampling,
    where both X and Y axes are sampled exponentially, which means that the
    grid coordinates on each axis [base**start, ..., base**stop] are evenly
    distributed on log scale.
    Each point on the square grid has a corresponding enrichment score,
    calculated using the X coordinate as as the cutoff for predictions,
    and Y coordinate as the cutoff for the true values.
    Finally the function will return a tuple of three numpy arrays,
    containing the X and Y coordinates, and the enrichment score for all
    the points on the square grid.
    Note that all three arrays have the shape of
    ('num_samples_per_axis', 'num_samples_per_axis') and share the same
    indexes.

    :param y_true: the true values or target of the predictive problem
    :type y_true: sequence of integer, float or any other comparable type
    :param y_pred: the predicted values
    :type y_pred: sequence of integer, float or any other comparable type
    :param num_samples_per_axis: number of samples per axis in the grid
    :type num_samples_per_axis: int
    :param logspace_start: inclusive starting point (base**'logspace_start')
        for logspace sampling, defaults to -3
    :type logspace_start: int or float
    :param logspace_stop: inclusive stopping point (base**'logspace_stop')
        for logspace sampling, defaults to 0
    :type logspace_stop: int or float
    :param logspace_base: the exponential base of logspace sampling for
        grid, defaults to 10
    :type logspace_base: int or float
    :param descending_ranking_order: descending ranking order option for
        both predictions and true targets, defaults to False
    :type descending_ranking_order: bool

    :return: tuple of three numpy arrays, all in the shape of
        ('num_samples_per_axis', 'num_samples_per_axis'), containing
        the X and Y  coordinates, and the enrichment score for all the
        points on the grid
    :rtype: tuple of three numpy arrays

    """

    # sanity check for
    # * len(y_true) == len(y_pred) > 0
    # * num_samples_per_axis > 0
    # * logspace_start < logspace_stop <= 0
    # * logspace_base > 0

    # obtain the ranked indexes for both target and predictions
    _ranked_indexes_true = np.argsort(y_true)[::-1] \
        if descending_ranking_order else np.argsort(y_true)
    _ranked_indexes_pred = tuple(
        np.argsort(y_pred)[::-1] if descending_ranking_order
        else np.argsort(y_pred))

    # construct the mesh grid
    # X and Y axes for positioning, and Z for enrichment score
    _sample_positions_on_axis = np.logspace(
        start=logspace_start,
        stop=logspace_stop,
        num=num_samples_per_axis,
        base=logspace_base,
    )

    # all X, Y, Z np arrays have shape of (num_samples, num_samples)
    _x_coordinates, _y_coordinates = \
        np.meshgrid(_sample_positions_on_axis, _sample_positions_on_axis)
    _z_values = np.zeros_like(_x_coordinates)

    # this for-loop could be optimized
    for _i in range(num_samples_per_axis):
        for _j in range(num_samples_per_axis):

            # assume that X axis represents the predictions, and Y the targets
            _z_values[_i, _j] = get_enrichment(
                ranked_indexes_true=_ranked_indexes_true,
                ranked_indexes_pred=_ranked_indexes_pred,
                cutoff_true=_y_coordinates[_i, _j],
                cutoff_pred=_x_coordinates[_i, _j],
            )

    return _x_coordinates, _y_coordinates, _z_values


class RegressionEnrichmentSurface:
    """This class combines the RES computation with the plot function.

    The initialization merely declares an empty variable, which will hold
    the enrichment surface grid(s) after 'get_enrichment_grids' is called.

    """

    def __init__(self):
        """Constructor method

        """
        # self.__stratified = None
        self.__enrichment_grids = None

    def get_enrichment_grids(
        self,
        y_true,
        y_pred,
        num_samples_per_axis=30,
        logspace_start=-3,
        logspace_stop=0,
        logspace_base=10,
        descending_ranking_order=False,
        stratified_on=None,
    ):
        """Generates multiple 2D square grid of enrichment scores.

        This function is mostly the same as 'get_enrichment_grid', with one
        additional functionality: stratification of the surfaces. If the
        argument 'stratified_on' is passed in, the function will seek to
        divide the target and predictions into different groups according
        to the stratified labels, and therefore generates multiple
        enrichment surfaces and returned a tuple of three numpy arrays,
        representing X, Y, and Z of all the surfaces.
        All three arrays share the shape of ('num_labels',
        'num_samples_per_axis', 'num_samples_per_axis'), where the
        'num_labels' equals to the number of unique labels in the
        sequence 'stratified_on', otherwise equals to 1.

        :param y_true: the true values or target of the predictive problem
        :type y_true: sequence of integer, float or any other comparable type
        :param y_pred: the predicted values
        :type y_pred: sequence of integer, float or any other comparable type
        :param num_samples_per_axis: number of samples per axis in the grid
        :type num_samples_per_axis: int
        :param logspace_start: inclusive starting point (base**logspace_start)
            for logspace sampling, defaults to -3
        :type logspace_start: int or float
        :param logspace_stop: inclusive stopping point (base**logspace_stop)
            for logspace sampling, defaults to 0
        :type logspace_stop: int or float
        :param logspace_base: the exponential base of logspace sampling for
            grid, defaults to 10
        :type logspace_base: int or float
        :param descending_ranking_order: descending ranking order option for
            both predictions and true targets, defaults to False
        :type descending_ranking_order: bool
        :param stratified_on: stratified labels for the samples, should be a
            sequence of the same length as 'y_true' and 'y_pred', labeling
            different predictions for multiple RES grids, defaults to None
        :type stratified_on: sequence

        :return: tuple of three numpy arrays, all in the shape of
            ('num_labels', 'num_samples_per_axis', 'num_samples_per_axis'),
            containing the X and Y coordinates, and the enrichment score
            for all the points on the grids. the 'num_labels' equals to the
            number of unique labels in the sequence 'stratified_on',
            otherwise equals to 1
        :rtype: tuple of three numpy arrays
        """

        _common_enrichment_grid_kwargs = {
            'num_samples_per_axis': num_samples_per_axis,
            'logspace_start': logspace_start,
            'logspace_stop': logspace_stop,
            'logspace_base': logspace_base,
            'descending_ranking_order': descending_ranking_order,
        }

        if stratified_on is None:
            _x, _y, _z = get_enrichment_grid(
                y_true=y_true,
                y_pred=y_pred,
                **_common_enrichment_grid_kwargs,
            )
            self.__enrichment_grids = ([_x, ], [_y, ], [_z, ])

        else:
            _stratified_xs, _stratified_ys, _stratified_zs = [], [], []
            _unique_labels, _label_indexes = \
                np.unique(stratified_on, return_inverse=True)

            for _label in _unique_labels:

                # construct one grid in the scope of each particular label
                _curr_y_indexes = np.argwhere(
                    stratified_on == _label).flatten()

                _curr_y_true = y_true[_curr_y_indexes]
                _curr_y_pred = y_pred[_curr_y_indexes]

                try:
                    _curr_x, _curr_y, _curr_z = get_enrichment_grid(
                        y_true=_curr_y_true,
                        y_pred=_curr_y_pred,
                        **_common_enrichment_grid_kwargs,
                    )
                # catch other errors, especially from the sanity check
                except ZeroDivisionError:
                    warnings.warn(
                        f'Insufficient amount ({len(_curr_y_true)}) of '
                        f'samples labeled as \'{_label}\' during stratified '
                        f'RES. Disregarding label \'{_label}\' ...')
                    continue

                _stratified_xs.append(_curr_x)
                _stratified_ys.append(_curr_y)
                _stratified_zs.append(_curr_z)

            self.__enrichment_grids = \
                (_stratified_xs, _stratified_ys, _stratified_zs)

        return self.__enrichment_grids

    def plot_enrichment_grids(
        self,
        plot_size=(8, 5),
        color_map='Blues',
        num_contour_levels=10,
        plot_title='Regression Enrichment Surface',
        plot_file_path=None,
    ):
        """Visualize the enrichment surface(s) as contour plot.

        This function averages the enrichment grids if there are more than
        one grids (resulted from the stratification of
        'get_enrichment_grids) and visualize the averaged surface in a
        contour plot.

        :param plot_size: size of the plot, represented by the width and
            height in inches, defaults to (8, 5)
        :type plot_size: tuple of two int or float
        :param color_map: color map for the contour plot, defaults to 'Blues'
        :type color_map: string or matplotlib Colormap
        :param num_contour_levels: number of contour levels, defaults to 10
        :type num_contour_levels: int
        :param plot_title: title for the contour plot, defaults to
            'Regression Enrichment Surface'
        :type plot_title: str
        :param plot_file_path: filepath for the contour plot. If given,
            the plot will be directed saved to the path, otherwise the
            plot will be shown. Defaults to None
        :type plot_file_path: str
        """

        assert self.__enrichment_grids

        plt.figure(figsize=plot_size)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Screen Top X%')
        plt.ylabel('True Top X%')
        plt.title(plot_title)

        _levels = np.linspace(0., 1., num_contour_levels, endpoint=True)
        _contour = plt.contourf(
            np.stack(self.__enrichment_grids[0]).mean(0),
            np.stack(self.__enrichment_grids[1]).mean(0),
            np.stack(self.__enrichment_grids[2]).mean(0),
            cmap=color_map,
            levels=_levels,
        )
        plt.colorbar(ticks=_levels)

        if plot_file_path:
            plt.savefig(
                plot_file_path,
                bbox_inches='tight',
                dpi=300,
            )
        else:
            plt.show()

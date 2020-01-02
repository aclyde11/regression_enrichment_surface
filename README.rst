=============================
Regression Enrichment Surface
=============================

Under dev.

How to use
Simple case:
To produced average plots over some stratification (useful for dose response like data to produce results over types of cells for instance)


.. code-block:: python

    trues, preds = get_predicition_data()
    rds_model = rds.RegressionDetectionSurface(percent_min=-3)
        rds_model.compute(trues, preds, samples=30)
        rds_model.plot(save_file=args.metric_plot_prefix + "rds_on_cell.png",
                       title='Regression Enrichment Surface (Avg over Unique Cells)')


To produced average plots over some stratification (useful for dose response like data to produce results over types of cells for instance)

.. code-block:: python
    trues, preds, labels = get_predicition_data()
    rds_model = rds.RegressionDetectionSurface(percent_min=-3)
        rds_model.compute(trues, preds, stratify=labels, samples=30)
        rds_model.plot(save_file=args.metric_plot_prefix + "rds_on_cell.png",
                       title='Regression Enrichment Surface (Avg over Unique Cells)')


.. image:: https://img.shields.io/pypi/v/regression_enrichment_surface.svg
        :target: https://pypi.python.org/pypi/regression_enrichment_surface

.. image:: https://img.shields.io/travis/aclyde11/regression_enrichment_surface.svg
        :target: https://travis-ci.org/aclyde11/regression_enrichment_surface

.. image:: https://readthedocs.org/projects/regression-enrichment-surface/badge/?version=latest
        :target: https://regression-enrichment-surface.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




Code for producing regression enrichment analysis


* Free software: MIT license
* Documentation: https://regression-enrichment-surface.readthedocs.io.


Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

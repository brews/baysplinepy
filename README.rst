baysplinepy
===========

.. image:: https://travis-ci.org/brews/baysplinepy.svg?branch=master
    :target: https://travis-ci.org/brews/baysplinepy


An open source Python package for `alkenone UK'37 <https://en.wikipedia.org/wiki/Alkenone>`_ calibration.

**baysplinepy** is based on the original BAYSPLINE software for MATLAB (https://github.com/jesstierney/BAYSPLINE). BAYSPLINE is a Bayesian calibration for the alkenone paleothermometer, as published in `Tierney & Tingley (2018) <http://doi.org/10.1002/2017PA003201>`_. 

NOTE that this package is under active development. Code and documentation may not be complete and may change in the near future.


Example
-------

First, load packages and an example dataset::

    import numpy as np
    import bayspline as bsl

    example_file = bsl.get_example_data('tierney2016-p178-15p.csv')
    d = np.genfromtxt(example_file, delimiter=',', names=True)

This dataset (from `Tierney et al. 2015 <https://doi.org/10.1038/ngeo2603>`_)
has three columns giving core depth (cm), sediment age (calendar years BP), and UK'37.

We can predict sea-surface temperatures (SST) from UK'37 with ``bsl.predict_sst()``::

    prediction = bsl.predict_sst(d['uk37'], pstd=10)

To see actual numbers from the prediction, directly parse ``prediction.ensemble`` or use ``prediction.percentile()`` to get the 5%, 50% and 95% percentiles.

You can also plot your prediction with ``bsl.predictplot()`` or ``bsl.densityplot()``.

Alternatively, we can make inferences about UK'37 from SST with ``bsl.predict_uk()``::

    sst = np.arange(1, 25)
    prediction = bsl.predict_uk(sst)


Installation
------------

Install **baysplinepy** in ``conda`` with::

    $ conda install baysplinepy -c sbmalev

To install with ``pip``, run::

    $ pip install baysplinepy

Unfortunately, **baysplinepy** is not compatible with Python 2.


Support and development
-----------------------

- Please feel free to report bugs and issues or view the source code on GitHub (https://github.com/brews/baysplinepy).


License
-------

**baysplinepy** is available under the Open Source GPLv3 (https://www.gnu.org/licenses).

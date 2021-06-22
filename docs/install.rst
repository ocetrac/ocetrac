Installation
--------------------


Requirements
~~~~~~

The only requirement is Python >=3.6. The following dependencies will be installed

* `xarray`_ 
* `dask`_
* `scipy`_
* `scikit-image`_

.. _xarray: http://xarray.pydata.org/en/stable/
.. _dask:  https://docs.dask.org/en/latest/install.html
.. _scipy: https://scipy.org/scipylib/
.. _scikit-image: https://scikit-image.org/

Instructions
~~~~~~

.. role:: bash(code)
   :language: bash
We are currently working to make Ocetrac available on conda-forge. For now the easiest way to install it is with pip:

.. code-block:: bash

    $ pip install ocetrac
    
You can also build Ocetrac from source by cloning the git repository with the latest development version and then installing it:

.. code-block:: bash

    $ git clone https://github.com/ocetrac/ocetrac.git
    $ python setup.py install [--user]


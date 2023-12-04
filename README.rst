scanhist
============

Installation
------------
To install scanhist into your environment from the source code::

    $ cd /path/to/root/scanhist
    $ pip install .

Testing
-------
To run the suite of unit tests::

    $ cd /path/to/root/scanhist
    $ pytest

To build html of test coverage::

    $ pytest -v --cov --cov-report html
    $ open htmlcov/index.html


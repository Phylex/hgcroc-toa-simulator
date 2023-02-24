ToA simulation for the HGCROCv3
===============================

To have a reference with witch to compare the measured Data with, as well as being able to generate data
for software system tests and help people in understanding the behaviour of the Time to Digital converters
(TDC) of the HGCROCv3, this simulation provides data generation and visualisation techniques.

Installation from pypi
----------------------
.. _pypi_installation:

The easiest and recommended way to install the package is by simply installing it from PyPi. to do this
you can run:

::

        pip install hgcroc-toa-simulator

This will automatically install the latest public version of the software.


Installation from source
------------------------
.. _source_installation:

The code in this repository constitutes a python package so with the current version of this repository
it can be installed using

::

        pip install .

optionally it can be installed in *editable mode* with

::

        pip install --editable .

Editable mode allows changes made in the repository code to affect the installed package immediately.

Usage
-----
The software is built to be a command line tool. After either pypi_installation_ or source_installation_, the ``toa-simulator`` command will
be available on the command line.

All commands and subcommands provide a ``help`` function that lists all available options and shows how
the command is to be called. The tool is quite sensitive to the exact order in whitch options, arguments
and subcommands are written, please stick to the format outlined in the ``help`` section of the tool.

Persistent configuration
------------------------
If the command line utility is run multiple times in a row, it generates a new TDC with slightly different
characteristics (in the same way the fabrication process would generate a slightly different behaviour for
each instance of the TDC. To acheive reliable results the exact internal state can be written to a config
file for use in later invocations.

===============================
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

The ``toa-simulator`` has a total of 5 subcommands.
* The ``generate-data`` subcommand generates uniformily distributed events from 0 to 25000ps and converts them to ToA codes using the ToA software simulation.
* The ``convert-data`` subcommand reads in arrival times from an input file and converts them to ToA codes which are written to ``stdout`` or a file
* The ``histogram`` generates uniformily distributed random arrival times and creates a histogram of the resulting ToA codes. Code distributions for different 
slow control configurations of the ToA can be drawn into the same plot.

* The ``plot-timing`` shows a view of the internal state of the TDC and describes how the TDC-code was generated.

Persistent configuration
------------------------
If the command line utility is run multiple times in a row, it generates a new TDC with slightly different
characteristics (in the same way the fabrication process would generate a slightly different behaviour for
each instance of the TDC. To acheive reliable results the exact internal state can be written to a config
file for use in later invocations.


Examples
========

histogram
---------
The easiest thing to get going is to run the ``histogram`` command as follows:

::

        toa-simulator histogram 40000 -sh

This will generate 40000 ToA codes, histogram them and then show the result on the screen. Using the ``-cs`` option we can se the results for different settings
of the ``CTRL_IN_SIG_CTDC_P_D`` configuration parameter. This would then look like:

::

        toa-simulator histogram 40000 -sh -cs 0 -cs 10 -cs 20 -cs 30

Which will draw 4 histograms in the same plot, one for each value of ``CTRL_IN_SIG_CTDC_P_D`` and show them together in the same histogram.

convert-data
------------
To convert data generated externally to the toa-simulator the ``convert-data`` subcommand can be used. The convert data function reads in data from a file
that looks similar to:

::

        Here we have some meta information
        Arrival Time [ps]
        402.3
        102.2
        803.2
        444
        579
        913
        8799

Each line has to have a single number on is that represents the arrival time that is to be converted. It can have arbitrarily many header lines.
The number of header-lines need to be passed to the ``convert-data`` command in the ``-s`` option (see the ``toa-simulator convert-data help``).
It will produce a new-line separated sequence of ToA codes on ``stdout``, or optionally write the output to a file. The position of the output Code
corresponds to the position of the arrival time-stamp in the input.

To process the above file with the path ``example_input.txt`` the following command can be used to output the resulting ToA codes into the console:

::

        toa-simulator convert-data example-input.txt



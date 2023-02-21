ToA simulation for the HGCROCv3
===============================

To have a reference with witch to compare the measured Data with, as well as being able to generate data
for software system tests and help people in understanding the behaviour of the Time to Digital converters
(TDC) of the HGCROCv3, this simulation provides data generation and visualisation techniques.

Installation
------------
.. _installation:
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
The software is built to be a command line tool. After installation_, the ``toa-simulator`` command will
be available on the command line.

All commands and subcommands provide a ``help`` function that lists all available options and shows how
the command is to be called.

Work in progress
================
The ``toa-simulator`` simulates a single TDC. If not provided with a configuration file, it regenerates the
exact gate timings for every invocation according to the parameters provided.

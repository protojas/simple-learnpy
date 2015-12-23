# simple-learnpy

Python library for some simple machine learning and classification, without numpy or scikit, which are often hard to install.

The goal is to provide the functionality of these classification techniques, without forcing a huge setup on the user - particularly, on Windows, scipy and scikit are unreasonably difficult to setup. These classification implementations, will operate entirely on arrays, and, where possible, will attempt to be fully functional, that is, without classes/objects.

Obviously, it will not be as efficient as Numpy, since it is written in Python and not C, but for results on simple data, this library will hopefully be much easier to use and be simpler to understand.
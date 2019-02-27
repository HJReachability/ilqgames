"""
BSD 3-Clause License

Copyright (c) 2018, HJ Reachability Group
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author(s): David Fridovich-Keil ( dfk@eecs.berkeley.edu )
"""
################################################################################
#
# Plotting utility, intended to be used with Logger.
#
################################################################################

import dill
import matplotlib.pyplot as plt
import numpy as np

class Plotter(object):
    """
    Plotting utility intended to be used with the Logger.
    """

    def __init__(self, filename):
        """
        Constructor.

        :param filename: Where to load log.
        :type filename: string
        """
        fp = open(filename, "rb")
        self._log = dill.load(fp)
        fp.close()

    def plot_scalar_fields(self, fields,
                           title="", xlabel="", ylabel=""):
        """
        Plot several scalar-valued fields over all time.

        :param fields: list of fields to plot
        :type fields: list of strings
        """
        plt.figure()
        for f in fields:
            plt.plot(self._log[f], linewidth=2, markersize=12, label=f)

        plt.legend()
        self._set_title_and_axis_labels(title, xlabel, ylabel)

    def show(self):
        plt.show()

    def _set_title_and_axis_labels(self, title, xlabel, ylabel):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

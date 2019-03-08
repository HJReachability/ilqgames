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
import scipy.io

class Plotter(object):
    """
    Plotting utility intended to be used with the Logger.
    """

    def __init__(self, py_filename, mat_filename):
        """
        Constructor.

        :param py_filename: Where to load python log.
        :param mat_filename: Where to load matlab log.
        :type filename: string
        """
        fp = open(py_filename, "rb")
        self._log = dill.load(fp)
        fp.close()

        self._matlab_log = scipy.io.loadmat(mat_filename)

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

    def plot_controls(self):
        """
        Plot the controls over time for the final optimized trajectory 

        """
        u = np.array(self._log['u1s'])
        u = u[-1,:,:,:]

        u1 = u[:,0].flatten()
        u2 = u[:,1].flatten()

        t = np.arange(1, u.shape[0]+1) * 0.1
        
        plt.figure()
        fig, axs = plt.subplots(2,1)
        axs[0].plot(t, u1)
        axs[0].set_xlabel('time (s)')
        axs[0].set_ylabel('u1 (rad/s)')

        axs[1].plot(t, u2)
        axs[1].set_xlabel('time (s)')
        axs[1].set_ylabel('u2 (m/s^2)')

        plt.title('Controls over time')
        plt.savefig('control.png')

    def plot_disturbances(self):
        """
        Plot the disturbance over time for the final optimized trajectory 

        """
        d = np.array(self._log['u2s'])
        d = d[-1,:,:,:]

        d1 = d[:,0].flatten()
        d2 = d[:,1].flatten()

        t = np.arange(1, d.shape[0]+1) * 0.1

        plt.figure()
        plt.plot(t, d1, label='d1')
        plt.plot(t, d2, label='d2')

        plt.xlabel('time (s)')
        plt.ylabel('disturbance (m/s)')
        plt.legend()
        plt.title('Disturances over time')
        plt.savefig('disturbance.png')

    def plot_player_costs(self):
        self.plot_scalar_fields(['total_cost1', 'total_cost2'], \
                title = 'player cost over time', xlabel='time (s)', ylabel='cost')

        plt.savefig('player_costs.png')


    def plot_trajectories(self):

        # Plot trajectory from iLQG
        xs = np.array(self._log['xs'])
        xs = xs[-1,:,:,:]

        x1s = xs[:,0].flatten()
        x2s = xs[:,1].flatten()

        plt.figure()
        plt.plot(x1s, x2s, label="iLQG")

            
        hji_xs = self._matlab_log['traj']
        plt.plot(hji_xs[0,:], hji_xs[1,:], label='HJI')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title('Final trajectory')
        plt.savefig('trajectory.png')
    


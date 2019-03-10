"""
BSD 3-Clause License

Copyright (c) 2019, HJ Reachability Group
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
# Base class for all cost functions. Structured as a functor
# so that it can be treated like a function, but support inheritance.
#
################################################################################

class Cost(object):
    """ Base class for all cost functions. """
    def __init__(self, name=""):
        self._name = name

    def __call__(self, xu, k):
        """
        Evaluate this cost function on the given input/time, which might either
        be a state `x` or a control `u`. Hence the input is named `xu`.
        NOTE: `xu` should be a PyTorch tensor with `requires_grad` set `True`.
        NOTE: `xu` should be a column vector.

        :param xu: state of the system or control input
        :type xu: torch.Tensor
        :param k: time step, if cost is time-varying
        :type k: uint
        :return: scalar value of cost
        :rtype: torch.Tensor
        """
        raise NotImplementedError("__call__ is not implemented.")

    def render(self, ax=None):
        """ Optional rendering on the given axes. """
        raise NotImplementedError("render is not implemented.")

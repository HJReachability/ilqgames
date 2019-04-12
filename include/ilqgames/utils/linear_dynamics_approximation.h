/*
 * Copyright (c) 2019, The Regents of the University of California (Regents).
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *    1. Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *    2. Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 *
 *    3. Neither the name of the copyright holder nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Please contact the author(s) of this library if you have any questions.
 * Authors: David Fridovich-Keil   ( dfk@eecs.berkeley.edu )
 */

///////////////////////////////////////////////////////////////////////////////
//
// Container to store a linear approximation of the dynamics at a particular
// time.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_UTILS_LINEAR_DYNAMICS_APPROXIMATION_H
#define ILQGAMES_UTILS_LINEAR_DYNAMICS_APPROXIMATION_H

#include <ilqgames/utils/types.h>

#include <vector>

namespace ilqgames {

struct LinearDynamicsApproximation {
  MatrixXf A;
  std::vector<MatrixXf> Bs;

  // Default constructor.
  LinearDynamicsApproximation() {}

  // Construct from a MultiPlayerDynamicalSystem. Templated to avoid include
  // cycle. Initialize A to identity and Bs to zero (since this is for a
  // discrete-time linearization).
  template <typename MultiPlayerSystemType>
  explicit LinearDynamicsApproximation(const MultiPlayerSystemType& system)
      : A(MatrixXf::Identity(system.XDim(), system.XDim())),
        Bs(system.NumPlayers()) {
    for (size_t ii = 0; ii < system.NumPlayers(); ii++)
      Bs[ii] = MatrixXf::Zero(system.XDim(), system.UDim(ii));
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};  // struct LinearDynamicsApproximation

}  // namespace ilqgames

#endif

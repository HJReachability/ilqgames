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
// Container to store a single player's time-indexed strategy.
//
// Notation is taken from Basar and Olsder, Corollary 6.1.
// -- alphas are the feedforward terms
// -- Ps are the feedback gains
// i.e. delta u[ii] = -P[ii] delta x - alphas[ii]
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_UTILS_STRATEGY_H
#define ILQGAMES_UTILS_STRATEGY_H

#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <vector>

namespace ilqgames {

struct Strategy {
  std::vector<MatrixXf> Ps;
  std::vector<VectorXf> alphas;

  // Preallocate memory during construction.
  Strategy(size_t horizon, Dimension xdim, Dimension udim)
      : Ps(horizon), alphas(horizon) {
    for (size_t ii = 0; ii < horizon; ii++) {
      Ps[ii] = MatrixXf::Zero(udim, xdim);
      alphas[ii] = VectorXf::Zero(udim);
    }
  }

  // Operator for computing control given time index and delta x.
  VectorXf operator()(size_t time_index, const VectorXf& delta_x,
                      const VectorXf& u_ref) const {
    return u_ref - Ps[time_index] * delta_x - alphas[time_index];
  }

  // Number of variables.
  size_t NumVariables() const {
    const size_t horizon = Ps.size();
    CHECK_EQ(horizon, alphas.size());

    return horizon * (Ps.front().size() + alphas.front().size());
  }
};  // struct Strategy

}  // namespace ilqgames

#endif

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
// Container to store solver logs.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/utils/log.h>
#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>
#include <ilqgames/utils/uncopyable.h>

#include <glog/logging.h>
#include <vector>

namespace ilqgames {

VectorXf Log::InterpolateState(size_t iterate, Time t) const {
  const OperatingPoint& op = operating_points_[iterate];

  // Low and high indices between which to interpolate.
  const size_t lo = TimeToIndex(t);
  const size_t hi = std::min(lo + 1, op.xs.size() - 1);

  // Fraction of the way between lo and hi.
  const float frac = (t - IndexToTime(lo)) / time_step_;
  return (1.0 - frac) * op.xs[lo] + frac * op.xs[hi];
}

float Log::InterpolateState(size_t iterate, Time t, Dimension dim) const {
  const OperatingPoint& op = operating_points_[iterate];

  // Low and high indices between which to interpolate.
  const size_t lo = TimeToIndex(t);
  const size_t hi = std::min(lo + 1, op.xs.size() - 1);

  // Fraction of the way between lo and hi.
  const float frac = (t - IndexToTime(lo)) / time_step_;
  return (1.0 - frac) * op.xs[lo](dim) + frac * op.xs[hi](dim);
}

VectorXf Log::InterpolateControl(size_t iterate, Time t,
                                 PlayerIndex player) const {
  const OperatingPoint& op = operating_points_[iterate];

  // Low and high indices between which to interpolate.
  const size_t lo = TimeToIndex(t);
  const size_t hi = std::min(lo + 1, op.xs.size() - 1);

  // Fraction of the way between lo and hi.
  const float frac = (t - IndexToTime(lo)) / time_step_;
  return (1.0 - frac) * op.us[lo][player] + frac * op.us[hi][player];
}

float Log::InterpolateControl(size_t iterate, Time t, PlayerIndex player,
                              Dimension dim) const {
  const OperatingPoint& op = operating_points_[iterate];

  // Low and high indices between which to interpolate.
  const size_t lo = TimeToIndex(t);
  const size_t hi = std::min(lo + 1, op.xs.size() - 1);

  // Fraction of the way between lo and hi.
  const float frac = (t - IndexToTime(lo)) / time_step_;
  return (1.0 - frac) * op.us[lo][player](dim) + frac * op.us[hi][player](dim);
}

inline std::vector<MatrixXf> Log::Ps(size_t iterate, size_t time_index) const {
  std::vector<MatrixXf> Ps(strategies_[iterate].size());
  for (PlayerIndex ii = 0; ii < Ps.size(); ii++)
    Ps[ii] = P(iterate, time_index, ii);
  return Ps;
}

inline std::vector<VectorXf> Log::alphas(size_t iterate,
                                         size_t time_index) const {
  std::vector<VectorXf> alphas(strategies_[iterate].size());
  for (PlayerIndex ii = 0; ii < alphas.size(); ii++)
    alphas[ii] = alpha(iterate, time_index, ii);
  return alphas;
}

inline MatrixXf Log::P(size_t iterate, size_t time_index,
                       PlayerIndex player) const {
  return strategies_[iterate][player].Ps[time_index];
}

inline VectorXf Log::alpha(size_t iterate, size_t time_index,
                           PlayerIndex player) const {
  return strategies_[iterate][player].alphas[time_index];
}

}  // namespace ilqgames

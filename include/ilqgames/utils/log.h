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

#ifndef ILQGAMES_UTILS_LOG_H
#define ILQGAMES_UTILS_LOG_H

#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>
#include <ilqgames/utils/uncopyable.h>

#include <math.h>
#include <vector>

namespace ilqgames {

class Log : private Uncopyable {
 public:
  ~Log() {}
  explicit Log(Time time_step) : time_step_(time_step) {}

  // Add a new solver iterate.
  void AddSolverIterate(const OperatingPoint& operating_point,
                        const std::vector<Strategy>& strategies) {
    operating_points_.push_back(operating_point);
    strategies_.push_back(strategies);
  }

  // Accessors.
  VectorXf InterpolateState(size_t iterate, Time t) const;
  float InterpolateState(size_t iterate, Time t, Dimension dim) const;
  VectorXf InterpolateControl(size_t iterate, Time t, PlayerIndex player) const;
  float InterpolateControl(size_t iterate, Time t, PlayerIndex player,
                           Dimension dim) const;

  std::vector<MatrixXf> Ps(size_t iterate, Time t) const;
  std::vector<VectorXf> alphas(size_t iterate, Time t) const;
  MatrixXf P(size_t iterate, Time t, PlayerIndex player) const;
  VectorXf alpha(size_t iterate, Time t, PlayerIndex player) const;

 private:
  // Get index corresponding to the time step immediately before the given time.
  size_t TimeToIndex(Time t) const {
    return static_cast<size_t>(std::max(constants::kSmallNumber, t) /
                               time_step_);
  }

  // Get time stamp corresponding to a particular index.
  Time IndexToTime(size_t idx) const {
    return time_step_ * static_cast<Time>(idx);
  }

  // Time discretization.
  const Time time_step_;

  // Operating points and stratgies, indexed by solver iterate.
  std::vector<OperatingPoint> operating_points_;
  std::vector<std::vector<Strategy>> strategies_;
};  // class Log

}  // namespace ilqgames

#endif

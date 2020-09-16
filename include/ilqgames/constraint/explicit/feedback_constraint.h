/*
 * Copyright (c) 2020, The Regents of the University of California (Regents).
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
// (Time-varying) feedback constraint,
// i.e., 0.5*||u_t^i - gamma(x_t; theta_t^i)||^2 = 0.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_CONSTRAINT_FEEDBACK_CONSTRAINT_H
#define ILQGAMES_CONSTRAINT_FEEDBACK_CONSTRAINT_H

#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/quadratic_constraint_approximation.h>
#include <ilqgames/utils/relative_time_tracker.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <memory>
#include <string>

namespace ilqgames {

class FeedbackConstraint : public RelativeTimeTracker {
 public:
  ~FeedbackConstraint() {}
  FeedbackConstraint(const StrategyRef* strategy_ref,
                     const std::string& name = "")
      : RelativeTimeTracker(name), strategy_(strategy_ref) {
    CHECK_NOTNULL(strategy_);
  }

  // Check if this constraint is satisfied, and optionally return the constraint
  // value, which equals zero if the constraint is satisfied.
  bool IsSatisfied(size_t time_step, const VectorXf& x, const VectorXf& u,
                   float* level) {
    const float value = 0.5 * (u - (*strategy_)(time_step, x)).squaredNorm();
    if (*level) *level = value;

    return std::abs(value) < constants::kSmallNumber;
  }

  // Quadraticize the constraint value. Do *not* keep a running sum since we
  // keep separate multipliers for each constraint.
  void Quadraticize(size_t time_step, const VectorXf& x, const VectorXf& u,
                    QuadraticConstraintApproximation* q) const {
    CHECK_NOTNULL(q);
  }

 private:
  // Strategy of a single player.
  const StrategyRef* strategy_;
};  //\class DynamicConstraint

}  // namespace ilqgames

#endif

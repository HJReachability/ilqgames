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
// (Time-invariant) proximity (inequality) constraint between two vehicles, i.e.
//           g(x) = ||(px1, py1) - (px2, py2)|| - d <= (or >=) 0
//
// NOTE: The `keep_within` argument specifies the sign of the inequality (true
// corresponds to <=).

///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_CONSTRAINT_PROXIMITY_CONSTRAINT_H
#define ILQGAMES_CONSTRAINT_PROXIMITY_CONSTRAINT_H

#include <ilqgames/constraint/time_invariant_constraint.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <memory>
#include <string>

namespace ilqgames {

class ProximityConstraint : public TimeInvariantConstraint {
 public:
  ~ProximityConstraint() {}
  ProximityConstraint(const std::pair<Dimension, Dimension>& dims1,
                      const std::pair<Dimension, Dimension>& dims2,
                      float threshold, bool keep_within, size_t num_time_steps,
                      const std::string& name = "")
      : TimeInvariantConstraint(false, num_time_steps, name),
        px1_idx(dims1.first),
        py1_idx(dims1.second),
        px2_idx(dims2.first),
        py2_idx(dims2.second),
        threshold_(threshold),
        keep_within_(keep_within) {
    CHECK_GT(threshold_, 0.0);
  }

  // Evaluate this constraint value, i.e., g(x).
  float Evaluate(const VectorXf& input) const;

  // Quadraticize the constraint value and its square, each scaled by lambda or
  // mu, respectively (terms in the augmented Lagrangian).
  void Quadraticize(Time t, const VectorXf& input, MatrixXf* hess,
                    VectorXf* grad) const;

 private:
  // Position dimension indices for both players.
  const Dimension px1_idx;
  const Dimension py1_idx;
  const Dimension px2_idx;
  const Dimension py2_idx;

  // Nominal distance threshold.
  const float threshold_;

  // Keep within (or without), i.e., orientation of the inequality.
  const bool keep_within_;
};  // namespace ilqgames

}  // namespace ilqgames

#endif

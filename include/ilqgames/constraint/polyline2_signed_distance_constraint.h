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
// (Time-invariant) inequality constraint encoding
//           g(x) = signed_distance(x, polyline) - d <= (or >=) 0
//
// NOTE: The `keep_left` argument specifies the sign of the inequality (true
// corresponds to <=).

///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_CONSTRAINT_POLYLINE2_SIGNED_DISTANCE_CONSTRAINT_H
#define ILQGAMES_CONSTRAINT_POLYLINE2_SIGNED_DISTANCE_CONSTRAINT_H

#include <ilqgames/constraint/time_invariant_constraint.h>
#include <ilqgames/geometry/line_segment2.h>
#include <ilqgames/geometry/polyline2.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <memory>
#include <string>

namespace ilqgames {

class Polyline2SignedDistanceConstraint : public TimeInvariantConstraint {
 public:
  ~Polyline2SignedDistanceConstraint() {}
  Polyline2SignedDistanceConstraint(const Polyline2& polyline,
                                    const std::pair<Dimension, Dimension>& dims,
                                    float threshold, bool keep_left,
                                    const std::string& name = "")
      : TimeInvariantConstraint(false, name),
        polyline_(polyline),
        xidx_(dims.first),
        yidx_(dims.second),
        threshold_(threshold),
        keep_left_(keep_left) {}

  // Evaluate this constraint value, i.e., g(x).
  float Evaluate(const VectorXf& input) const;

  // Quadraticize the constraint value and its square, each scaled by lambda or
  // mu, respectively (terms in the augmented Lagrangian).
  void Quadraticize(Time t, const VectorXf& input, MatrixXf* hess,
                    VectorXf* grad) const;

 private:
  // Polyline.
  const Polyline2 polyline_;

  // Position dimension indices.
  const Dimension xidx_;
  const Dimension yidx_;

  // Nominal distance threshold.
  const float threshold_;

  // Keep left (or right), i.e., orientation of the inequality.
  const bool keep_left_;
};  // namespace Polyline2SignedDistanceConstraint

}  // namespace ilqgames

#endif

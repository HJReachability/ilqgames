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
// Simple skeleton example intended as a quick-start guide for learning to use
// this repository. This file is extensively commented; however, if you do have
// any questions please don't hesitate to post an issue on the repository
// (https://github.com/HJReachability/ilqgames/issues) or contact the author.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_EXAMPLES_SKELETON_EXAMPLE_H
#define ILQGAMES_EXAMPLES_SKELETON_EXAMPLE_H

#include <ilqgames/solver/solver_params.h>
#include <ilqgames/solver/top_down_renderable_problem.h>

namespace ilqgames {

class SkeletonExample : public TopDownRenderableProblem {
 public:
  ~SkeletonExample() {}
  SkeletonExample() : TopDownRenderableProblem() {}

  // Construct dynamics, initial state, and player costs.
  void ConstructDynamics();
  void ConstructInitialState();
  void ConstructPlayerCosts();

  // Unpack x, y, heading (for each player, potentially) from a given state.
  std::vector<float> Xs(const VectorXf& x) const;
  std::vector<float> Ys(const VectorXf& x) const;
  std::vector<float> Thetas(const VectorXf& x) const;
};  // class SkeletonExample

}  // namespace ilqgames

#endif

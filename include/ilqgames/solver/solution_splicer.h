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
// Splice together existing and new solutions to a receding horizon problem.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_SOLVER_SOLUTION_SPLICER_H
#define ILQGAMES_SOLVER_SOLUTION_SPLICER_H

#include <ilqgames/dynamics/multi_player_integrable_system.h>
#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/solver_log.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

#include <memory>
#include <vector>

namespace ilqgames {

class SolutionSplicer {
 public:
  ~SolutionSplicer() {}
  explicit SolutionSplicer(const SolverLog& log);

  // Splice in a new solution stored in a solver log.
  void Splice(const SolverLog& log);

  // Check if a given time is contained within the current operating point.
  bool ContainsTime(Time t) const {
    return (operating_point_.t0 <= t) &&
           (operating_point_.t0 +
                operating_point_.xs.size() * time::kTimeStep >=
            t);
  }

  // Accessors.
  const std::vector<Strategy>& CurrentStrategies() const { return strategies_; }
  const OperatingPoint& CurrentOperatingPoint() const {
    return operating_point_;
  }

 private:
  // Converged strategies and operating points for all players.
  std::vector<Strategy> strategies_;
  OperatingPoint operating_point_;
};  // class SolutionSplicer

}  // namespace ilqgames

#endif

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
// Static variables shared by all GUI windows.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_GUI_CONTROL_SLIDERS_H
#define ILQGAMES_GUI_CONTROL_SLIDERS_H

#include <ilqgames/utils/solver_log.h>

#include <memory>
#include <vector>

namespace ilqgames {

class ControlSliders {
 public:
  ~ControlSliders() {}
  ControlSliders(const std::vector<
                 std::vector<std::shared_ptr<const ilqgames::SolverLog>>>&
                     logs_for_each_problem)
      : interpolation_time_(0.0),
        solver_iterate_(0),
        log_index_(0),
        max_log_index_(0),
        logs_for_each_problem_(logs_for_each_problem) {
    for (const auto& logs : logs_for_each_problem_) {
      for (const auto& log : logs) CHECK_NOTNULL(log.get());
    }

    // Compute max log index.
    for (const auto& logs : logs_for_each_problem_) {
      if (logs.size() > static_cast<size_t>(max_log_index_) + 1)
        max_log_index_ = logs.size() - 1;
    }
  }

  // Render all the sliders in a separate window.
  void Render();

  // Accessors.
  size_t NumProblems() const { return logs_for_each_problem_.size(); }
  const std::vector<std::vector<std::shared_ptr<const SolverLog>>>&
  LogsForEachProblem() const {
    return logs_for_each_problem_;
  }
  std::vector<std::shared_ptr<const SolverLog>> LogForEachProblem() const {
    std::vector<std::shared_ptr<const SolverLog>> logs(NumProblems());
    for (size_t ii = 0; ii < NumProblems(); ii++)
      logs[ii] = logs_for_each_problem_[ii][LogIndex(ii)];
    return logs;
  }

  Time InterpolationTime(size_t problem_idx) const {
    CHECK_LT(problem_idx, NumProblems());

    const auto& logs = logs_for_each_problem_[problem_idx];
    const int log_idx = LogIndex(problem_idx);
    return std::max(std::min(static_cast<Time>(interpolation_time_),
                             logs[log_idx]->FinalTime()),
                    logs[log_idx]->InitialTime());
  }
  int SolverIterate(size_t problem_idx) const {
    CHECK_LT(problem_idx, NumProblems());

    const auto& logs = logs_for_each_problem_[problem_idx];
    const int log_idx = LogIndex(problem_idx);
    return std::min(solver_iterate_,
                    static_cast<int>(logs[log_idx]->NumIterates() - 1));
  }
  int LogIndex(size_t problem_idx) const {
    CHECK_LT(problem_idx, NumProblems());

    const auto& logs = logs_for_each_problem_[problem_idx];
    return std::min(log_index_, static_cast<int>(logs.size() - 1));
  }
  int MaxLogIndex() const { return max_log_index_; }

 private:
  // Time at which to interpolate each trajectory.
  float interpolation_time_;

  // Solver iterate to display.
  int solver_iterate_;

  // Log index to render for receding horizon problems.
  int log_index_;

  // Keep track of the max number of log indices across all problems.
  int max_log_index_;

  // List of all logs we might want to inspect, indexed by problem, then by
  // receding horizon invocation.
  const std::vector<std::vector<std::shared_ptr<const ilqgames::SolverLog>>>
      logs_for_each_problem_;
};  // class ControlSliders

}  // namespace ilqgames

#endif

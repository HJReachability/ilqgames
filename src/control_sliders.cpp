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

#include <ilqgames/gui/control_sliders.h>
#include <ilqgames/utils/solver_log.h>
#include <ilqgames/utils/types.h>

#include <imgui/imgui.h>
#include <memory>
#include <vector>

namespace ilqgames {

void ControlSliders::Render() {
  ImGui::Begin("Control Sliders");

  // Make a slider to get the desired log index from a receding horizon problem.
  if (!logs_for_each_problem_.empty())
    ImGui::SliderInt("Log Index", &log_index_, 0, max_log_index_);

  // Compute endpoints of sliders to come.
  int max_solver_iterates = 0;
  Time max_final_time = -std::numeric_limits<Time>::infinity();
  Time min_initial_time = std::numeric_limits<Time>::infinity();
  for (size_t ii = 0; ii < logs_for_each_problem_.size(); ii++) {
    const auto& logs = logs_for_each_problem_[ii];

    max_solver_iterates =
        std::max(max_solver_iterates,
                 static_cast<int>(logs[LogIndex(ii)]->NumIterates()) - 1);
    max_final_time = std::max(max_final_time, logs[LogIndex(ii)]->FinalTime());
    min_initial_time =
        std::min(min_initial_time, logs[LogIndex(ii)]->InitialTime());
  }

  // Make a slider to get the desired iterate.
  ImGui::SliderInt("Iterate", &solver_iterate_, 0, max_solver_iterates);

  // Make a slider to get the desired interpolation time.
  ImGui::SliderFloat("Interpolation Time (s)", &interpolation_time_,
                     min_initial_time, max_final_time);

  ImGui::End();
}

}  // namespace ilqgames

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
// Utility for plotting different costs for each player over time. Integrates
// with DearImGui.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/gui/control_sliders.h>
#include <ilqgames/gui/cost_inspector.h>
#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/player_cost_cache.h>
#include <ilqgames/utils/solver_log.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <imgui/imgui.h>
#include <string>
#include <vector>

namespace ilqgames {

void CostInspector::Render() {
  // Do nothing if log is empty.
  if (player_costs_.Log().NumIterates() == 0) return;

  // Set up main window.
  ImGui::Begin("Cost Inspector");

  // Combo box to select player.
  if (ImGui::BeginCombo("Player", std::to_string(selected_player_ + 1).c_str())) {
    for (PlayerIndex ii = 0; ii < player_costs_.NumPlayers(); ii++) {
      const bool is_selected = (selected_player_ == ii);
      if (ImGui::Selectable(std::to_string(ii + 1).c_str(), is_selected))
        selected_player_ = ii;
      if (is_selected) ImGui::SetItemDefaultFocus();
    }

    ImGui::EndCombo();
  }

  // Combo box to select cost.
  if (ImGui::BeginCombo("Cost", selected_cost_name_.c_str())) {
    for (const auto& entry : player_costs_.EvaluatedCosts(selected_player_)) {
      const std::string& cost_name = entry.first;
      const bool is_selected = (selected_cost_name_ == cost_name);
      if (ImGui::Selectable(cost_name.c_str(), is_selected))
        selected_cost_name_ = cost_name;
      if (is_selected) ImGui::SetItemDefaultFocus();
    }

    ImGui::EndCombo();
  }

  // Plot the given cost.
  if (ImGui::BeginChild("Cost over time", ImVec2(0, 0), false)) {
    const std::string label = "Player " + std::to_string(selected_player_ + 1) +
                              ": " + selected_cost_name_;
    if (player_costs_.PlayerHasCost(selected_player_, selected_cost_name_)) {
      const std::vector<float>& values = player_costs_.EvaluatedCost(
          sliders_->SolverIterate(), selected_player_, selected_cost_name_);
      ImGui::PlotLines(label.c_str(), values.data(), values.size(), 0,
                       label.c_str(), FLT_MAX, FLT_MAX,
                       ImGui::GetWindowContentRegionMax());
    }
  }
  ImGui::EndChild();

  // End this window.
  ImGui::End();
}

}  // namespace ilqgames

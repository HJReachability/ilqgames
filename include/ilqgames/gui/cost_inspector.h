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

#ifndef ILQGAMES_GUI_COST_INSPECTOR_H
#define ILQGAMES_GUI_COST_INSPECTOR_H

#include <ilqgames/cost/player_cost.h>
#include <ilqgames/gui/control_sliders.h>
#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/player_cost_cache.h>
#include <ilqgames/utils/solver_log.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <imgui/imgui.h>
#include <string>
#include <vector>

namespace ilqgames {

class CostInspector {
 public:
  ~CostInspector() {}

  // Takes in a log and lists of x/y/heading indices in
  // the state vector.
  CostInspector(const std::shared_ptr<const ControlSliders>& sliders,
                const std::shared_ptr<const SolverLog>& log,
                const std::vector<PlayerCost>& player_costs)
      : sliders_(sliders),
        player_costs_(log, player_costs),
        selected_player_(0),
        selected_cost_name_("<Please select a cost>") {
    CHECK_NOTNULL(sliders_.get());
  }

  // Render the appropriate costs.
  void Render();

 private:
  // Control sliders.
  const std::shared_ptr<const ControlSliders> sliders_;

  // Player cost cache.
  const PlayerCostCache player_costs_;

  // Currently selected player and cost name.
  PlayerIndex selected_player_;
  std::string selected_cost_name_;
};  // class CostInspector

}  // namespace ilqgames

#endif

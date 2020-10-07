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
// Storage utility for inspecting player costs corresponding to a log.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/cost/player_cost.h>
#include <ilqgames/dynamics/multi_player_flat_system.h>
#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/player_cost_cache.h>
#include <ilqgames/utils/solver_log.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <memory>
#include <unordered_map>
#include <vector>

namespace ilqgames {

PlayerCostCache::PlayerCostCache(const std::shared_ptr<const SolverLog>& log,
                                 const std::vector<PlayerCost>& player_costs)
    : log_(log) {
  CHECK_NOTNULL(log.get());

  // Populate costs separately for each player.
  evaluated_player_costs_.resize(player_costs.size());
  for (PlayerIndex ii = 0; ii < player_costs.size(); ii++) {
    const auto& player_cost = player_costs[ii];
    auto& evaluated_costs = evaluated_player_costs_[ii];

    // Cycle through each separate cost.
    // Start with state costs.
    for (const auto& cost : player_cost.StateCosts()) {
      auto e = evaluated_costs.emplace(cost->Name(),
                                       std::vector<std::vector<float>>());
      LOG_IF(WARNING, !e.second)
          << "Player " << ii
          << " has duplicate cost with name: " << cost->Name();

      auto& entry = e.first->second;
      entry.resize(log->NumIterates());
      for (size_t jj = 0; jj < log->NumIterates(); jj++) {
        entry[jj].resize(time::kNumTimeSteps);

        for (size_t kk = 0; kk < time::kNumTimeSteps; kk++) {
          const VectorXf x = log->State(jj, kk);
          entry[jj][kk] = cost->Evaluate(log->IndexToTime(kk), x);
        }
      }
    }

    // Now handle control costs.
    for (const auto& cost_pair : player_cost.ControlCosts()) {
      const auto other_player = cost_pair.first;
      const auto& cost = cost_pair.second;
      auto e = evaluated_costs.emplace(cost->Name(),
                                       std::vector<std::vector<float>>());
      LOG_IF(WARNING, !e.second)
          << "Player " << ii
          << " has duplicate cost with name: " << cost->Name();

      auto& entry = e.first->second;
      entry.resize(log->NumIterates());
      for (size_t jj = 0; jj < log->NumIterates(); jj++) {
        entry[jj].resize(time::kNumTimeSteps);

        for (size_t kk = 0; kk < time::kNumTimeSteps; kk++) {
          entry[jj][kk] = cost->Evaluate(log->IndexToTime(kk),
                                         log->Control(jj, kk, other_player));
        }
      }
    }

    // Handle constraints.
    for (const auto& constraint : player_cost.StateConstraints()) {
      auto e = evaluated_costs.emplace(constraint->Name(),
                                       std::vector<std::vector<float>>());
      LOG_IF(WARNING, !e.second)
          << "Player " << ii
          << " has duplicate constraint with name: " << constraint->Name();

      auto& entry = e.first->second;
      entry.resize(log->NumIterates());
      for (size_t jj = 0; jj < log->NumIterates(); jj++) {
        entry[jj].resize(time::kNumTimeSteps);

        for (size_t kk = 0; kk < time::kNumTimeSteps; kk++) {
          const VectorXf x = log->State(jj, kk);
          entry[jj][kk] = constraint->Evaluate(log->IndexToTime(kk), x);
        }
      }
    }

    // Now handle control constraints.
    for (const auto& constraint_pair : player_cost.ControlConstraints()) {
      const auto other_player = constraint_pair.first;
      const auto& constraint = constraint_pair.second;
      auto e = evaluated_costs.emplace(constraint->Name(),
                                       std::vector<std::vector<float>>());
      LOG_IF(WARNING, !e.second)
          << "Player " << ii
          << " has duplicate constraint with name: " << constraint->Name();

      auto& entry = e.first->second;
      entry.resize(log->NumIterates());
      for (size_t jj = 0; jj < log->NumIterates(); jj++) {
        entry[jj].resize(time::kNumTimeSteps);

        for (size_t kk = 0; kk < time::kNumTimeSteps; kk++) {
          entry[jj][kk] = constraint->Evaluate(
              log->IndexToTime(kk), log->Control(jj, kk, other_player));
        }
      }
    }
  }
}

float PlayerCostCache::Interpolate(size_t iterate, Time t, PlayerIndex player,
                                   const std::string& name) const {
  CHECK_LT(iterate, log_->NumIterates());
  CHECK_LT(player, evaluated_player_costs_.size());

  // Access the approprate time-indexed list of costs.
  const auto& costs = evaluated_player_costs_[player].at(name)[iterate];

  // Interpolate this list.
  const size_t lo = log_->TimeToIndex(t);
  const size_t hi = std::min(lo + 1, time::kNumTimeSteps - 1);

  const float frac = (t - log_->IndexToTime(lo)) / time::kTimeStep;
  return (1.0 - frac) * costs[lo] + frac * costs[hi];
}

}  // namespace ilqgames

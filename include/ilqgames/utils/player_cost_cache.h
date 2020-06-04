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

#ifndef ILQGAMES_UTILS_PLAYER_COST_CACHE_H
#define ILQGAMES_UTILS_PLAYER_COST_CACHE_H

#include <ilqgames/cost/player_cost.h>
#include <ilqgames/dynamics/multi_player_flat_system.h>
#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/solver_log.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <memory>
#include <unordered_map>
#include <vector>

namespace ilqgames {

class PlayerCostCache {
 public:
  ~PlayerCostCache() {}
  PlayerCostCache(const std::shared_ptr<const SolverLog>& log,
                  const std::vector<PlayerCost>& player_costs);

  // Interpolate the given cost at the specified iterate and time.
  float Interpolate(size_t iterate, Time t, PlayerIndex player,
                    const std::string& name) const;

  // Accessors.
  const SolverLog& Log() const { return *log_; }
  size_t NumPlayers() const { return evaluated_player_costs_.size(); }
  size_t NumCosts(PlayerIndex player) const {
    return evaluated_player_costs_[player].size();
  }
  bool PlayerHasCost(PlayerIndex player, const std::string& name) const {
    return evaluated_player_costs_[player].count(name) > 0;
  }
  const std::unordered_map<std::string, std::vector<std::vector<float>>>&
  EvaluatedCosts(PlayerIndex player) const {
    return evaluated_player_costs_[player];
  }
  const std::vector<float>& EvaluatedCost(size_t iterate, PlayerIndex player,
                                          const std::string& name) const {
    CHECK(PlayerHasCost(player, name));
    CHECK_LT(iterate, evaluated_player_costs_[player].at(name).size());

    return evaluated_player_costs_[player].at(name)[iterate];
  }

 private:
  // Log. Currently only used for converting between times and time steps.
  const std::shared_ptr<const SolverLog> log_;

  // Player costs (indexed by player and string ID) evaluated every iterate
  // and time step.
  std::vector<std::unordered_map<std::string, std::vector<std::vector<float>>>>
      evaluated_player_costs_;
};  // class PlayerCostCache

}  // namespace ilqgames

#endif

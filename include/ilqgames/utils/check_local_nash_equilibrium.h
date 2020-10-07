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
// Check whether or not a particular set of strategies is a local Nash
// equilibrium.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_UTILS_CHECK_LOCAL_NASH_EQUILIBRIUM_H
#define ILQGAMES_UTILS_CHECK_LOCAL_NASH_EQUILIBRIUM_H

#include <ilqgames/cost/player_cost.h>
#include <ilqgames/dynamics/multi_player_flat_system.h>
#include <ilqgames/dynamics/multi_player_integrable_system.h>
#include <ilqgames/solver/problem.h>
#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

#include <vector>

namespace ilqgames {

// Check if this set of strategies is a local Nash equilibrium by randomly
// changing each player's strategy with a number of small perturbations.
bool NumericalCheckLocalNashEquilibrium(
    const std::vector<PlayerCost>& player_costs,
    const std::vector<Strategy>& strategies,
    const OperatingPoint& operating_point,
    const MultiPlayerIntegrableSystem& dynamics, const VectorXf& x0,
    float max_perturbation, bool open_loop = false);
bool NumericalCheckLocalNashEquilibrium(const Problem& problem,
                                        float max_perturbation,
                                        bool open_loop = false);

// Check sufficient conditions for local Nash equilibrium, i.e., Q_i, R_ij all
// positive semidefinite for each player. Optionally takes in a pointer to flat
// dynamics; if provided, the PSD check is performed after changing cost
// coordinates.
bool CheckSufficientLocalNashEquilibrium(
    const std::vector<PlayerCost>& player_costs,
    const OperatingPoint& operating_point,
    const std::shared_ptr<const MultiPlayerIntegrableSystem>& dynamics =
        nullptr);
bool CheckSufficientLocalNashEquilibrium(const Problem& problem);

}  // namespace ilqgames

#endif

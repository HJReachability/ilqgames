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
// Base class for all multi-player *integrable* dynamical systems.
// Supports (discrete-time) linearization and integration.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/dynamics/multi_player_integrable_system.h>
#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

#include <vector>

namespace ilqgames {

bool MultiPlayerIntegrableSystem::integrate_using_euler_ = false;

VectorXf MultiPlayerIntegrableSystem::Integrate(
    Time t0, Time t, const VectorXf& x0, const OperatingPoint& operating_point,
    const std::vector<Strategy>& strategies) const {
  CHECK_GE(t, t0);
  CHECK_GE(t0, operating_point.t0);
  CHECK_EQ(strategies.size(), NumPlayers());

  std::vector<VectorXf> us(NumPlayers());

  // Compute current timestep and final timestep.
  const Time relative_t0 = t0 - operating_point.t0;
  const size_t current_timestep =
      static_cast<size_t>(relative_t0 / time::kTimeStep);

  const Time relative_t = t - operating_point.t0;
  const size_t final_timestep =
      static_cast<size_t>(relative_t / time::kTimeStep);

  // Handle case where 't0' is after 'operating_point.t0' by integrating from
  // 't0' to the next discrete timestep.
  VectorXf x(x0);
  if (t0 > operating_point.t0)
    x = IntegrateToNextTimeStep(t0, x0, operating_point, strategies);

  // Integrate forward step by step up to timestep including t.
  x = Integrate(current_timestep + 1, final_timestep, x, operating_point,
                strategies);

  // Integrate forward from this timestep to t.
  return IntegrateFromPriorTimeStep(t, x, operating_point, strategies);
}

VectorXf MultiPlayerIntegrableSystem::Integrate(
    size_t initial_timestep, size_t final_timestep, const VectorXf& x0,
    const OperatingPoint& operating_point,
    const std::vector<Strategy>& strategies) const {
  VectorXf x(x0);
  std::vector<VectorXf> us(NumPlayers());
  for (size_t kk = initial_timestep; kk < final_timestep; kk++) {
    const Time t = operating_point.t0 + kk * time::kTimeStep;

    // Populate controls for all players.
    for (PlayerIndex ii = 0; ii < NumPlayers(); ii++)
      us[ii] = strategies[ii](kk, x - operating_point.xs[kk],
                              operating_point.us[kk][ii]);

    x = Integrate(t, time::kTimeStep, x, us);
  }

  return x;
}

VectorXf MultiPlayerIntegrableSystem::IntegrateToNextTimeStep(
    Time t0, const VectorXf& x0, const OperatingPoint& operating_point,
    const std::vector<Strategy>& strategies) const {
  CHECK_GE(t0, operating_point.t0);

  // Compute remaining time this timestep.
  const Time relative_t0 = t0 - operating_point.t0;
  const size_t current_timestep = static_cast<size_t>(
      (relative_t0 +
       constants::kSmallNumber)  // Add to avoid inadvertently subtracting 1.
      / time::kTimeStep);
  const Time remaining_time_this_step =
      time::kTimeStep * (current_timestep + 1) - relative_t0;
  CHECK_LT(remaining_time_this_step, time::kTimeStep + constants::kSmallNumber);
  CHECK_LT(current_timestep, operating_point.xs.size());

  // Interpolate x0_ref.
  const float frac = remaining_time_this_step / time::kTimeStep;
  const VectorXf x0_ref =
      (current_timestep + 1 < operating_point.xs.size())
          ? frac * operating_point.xs[current_timestep] +
                (1.0 - frac) * operating_point.xs[current_timestep + 1]
          : operating_point.xs.back();

  // Populate controls for each player.
  std::vector<VectorXf> us(NumPlayers());
  for (PlayerIndex ii = 0; ii < NumPlayers(); ii++)
    us[ii] = strategies[ii](current_timestep, x0 - x0_ref,
                            operating_point.us[current_timestep][ii]);

  return Integrate(t0, remaining_time_this_step, x0, us);
}

VectorXf MultiPlayerIntegrableSystem::IntegrateFromPriorTimeStep(
    Time t, const VectorXf& x0, const OperatingPoint& operating_point,
    const std::vector<Strategy>& strategies) const {
  // Compute time until next timestep.
  const Time relative_t = t - operating_point.t0;
  const size_t current_timestep =
      static_cast<size_t>(relative_t / time::kTimeStep);
  const Time remaining_time_until_t =
      relative_t - time::kTimeStep * current_timestep;
  CHECK_LT(current_timestep, operating_point.xs.size()) << t;
  CHECK_LT(remaining_time_until_t, time::kTimeStep);

  // Populate controls for each player.
  std::vector<VectorXf> us(NumPlayers());
  for (PlayerIndex ii = 0; ii < NumPlayers(); ii++) {
    us[ii] = strategies[ii](current_timestep,
                            x0 - operating_point.xs[current_timestep],
                            operating_point.us[current_timestep][ii]);
  }

  return Integrate(operating_point.t0 + time::kTimeStep * current_timestep,
                   remaining_time_until_t, x0, us);
}

VectorXf MultiPlayerIntegrableSystem::Integrate(
    Time t0, Time time_interval, const Eigen::Ref<VectorXf>& x0,
    const std::vector<Eigen::Ref<VectorXf>>& us) const {
  std::vector<VectorXf> eval_us(us.size());
  std::transform(us.begin(), us.end(), eval_us.begin(),
                 [](const Eigen::Ref<VectorXf>& u) { return u.eval(); });

  return Integrate(t0, time_interval, x0.eval(), eval_us);
};

}  // namespace ilqgames

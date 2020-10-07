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

#ifndef ILQGAMES_DYNAMICS_MULTI_PLAYER_INTEGRABLE_SYSTEM_H
#define ILQGAMES_DYNAMICS_MULTI_PLAYER_INTEGRABLE_SYSTEM_H

#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

#include <vector>

namespace ilqgames {

class MultiPlayerIntegrableSystem {
 public:
  virtual ~MultiPlayerIntegrableSystem() {}

  // Integrate these dynamics forward in time.
  // Options include integration for a single timestep, between arbitrary times,
  // and within a single timestep.
  virtual VectorXf Integrate(Time t0, Time time_interval, const VectorXf& x0,
                             const std::vector<VectorXf>& us) const = 0;
  VectorXf Integrate(Time t0, Time t, const VectorXf& x0,
                     const OperatingPoint& operating_point,
                     const std::vector<Strategy>& strategies) const;
  VectorXf Integrate(size_t initial_timestep, size_t final_timestep,
                     const VectorXf& x0, const OperatingPoint& operating_point,
                     const std::vector<Strategy>& strategies) const;
  VectorXf IntegrateToNextTimeStep(
      Time t0, const VectorXf& x0, const OperatingPoint& operating_point,
      const std::vector<Strategy>& strategies) const;
  VectorXf IntegrateFromPriorTimeStep(
      Time t, const VectorXf& x0, const OperatingPoint& operating_point,
      const std::vector<Strategy>& strategies) const;

  // Make a utility version of the above that operates on Eigen::Refs.
  VectorXf Integrate(Time t0, Time time_interval,
                     const Eigen::Ref<VectorXf>& x0,
                     const std::vector<Eigen::Ref<VectorXf>>& us) const;

  // Can this system be treated as linear for the purposes of LQ solves?
  // For example, linear systems and feedback linearizable systems should
  // return true here.
  virtual bool TreatAsLinear() const { return false; }

  // Stitch between two states of the system. By default, just takes the
  // first one but concatenated systems, e.g., can interpret the first one
  // as best for ego and the second as best for other players.
  virtual VectorXf Stitch(const VectorXf& x_ego,
                          const VectorXf& x_others) const {
    return x_ego;
  }

  // Integrate using single step Euler or not, see below for more extensive
  // description.
  static void IntegrateUsingEuler() { integrate_using_euler_ = true; }
  static void IntegrateUsingRK4() { integrate_using_euler_ = false; }
  static bool IntegrationUsesEuler() { return integrate_using_euler_; }

  // Getters.
  Dimension XDim() const { return xdim_; }
  Dimension TotalUDim() const {
    Dimension total = 0;
    for (PlayerIndex ii = 0; ii < NumPlayers(); ii++) total += UDim(ii);
    return total;
  }
  virtual Dimension UDim(PlayerIndex player_idx) const = 0;
  virtual PlayerIndex NumPlayers() const = 0;
  virtual std::vector<Dimension> PositionDimensions() const = 0;

  // Distance metric between two states. By default, just the *squared* 2-norm.
  virtual float DistanceBetween(const VectorXf& x0, const VectorXf& x1) const {
    return (x0 - x1).squaredNorm();
  }

 protected:
  MultiPlayerIntegrableSystem(Dimension xdim) : xdim_(xdim) {}

  // State dimension.
  const Dimension xdim_;

  // Whether to use single Euler during integration. Typically this is false but
  // it is typically used either for testing (we only derive Nash typically in
  // this case) or for speed.
  static bool integrate_using_euler_;
};  //\class MultiPlayerIntegrableSystem

}  // namespace ilqgames

#endif

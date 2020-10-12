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
// Multi-player dynamical system comprised of several single player subsystems.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_DYNAMICS_CONCATENATED_FLAT_SYSTEM_H
#define ILQGAMES_DYNAMICS_CONCATENATED_FLAT_SYSTEM_H

#include <ilqgames/dynamics/multi_player_flat_system.h>
#include <ilqgames/dynamics/single_player_flat_system.h>
#include <ilqgames/utils/linear_dynamics_approximation.h>
#include <ilqgames/utils/types.h>

namespace ilqgames {

using FlatSubsystemList =
    std::vector<std::shared_ptr<SinglePlayerFlatSystem>>;

class ConcatenatedFlatSystem : public MultiPlayerFlatSystem {
 public:
  ~ConcatenatedFlatSystem() {}
  ConcatenatedFlatSystem(const FlatSubsystemList& subsystems);

  // Compute time derivative of state.
  VectorXf Evaluate(const VectorXf& x, const std::vector<VectorXf>& us) const;

  // Discrete time approximation of the underlying linearized system.
  void ComputeLinearizedSystem() const;

  // Utilities for feedback linearization.
  MatrixXf InverseDecouplingMatrix(const VectorXf& x) const;
  VectorXf AffineTerm(const VectorXf& x) const;
  VectorXf LinearizingControl(const VectorXf& x, const VectorXf& v,
                              PlayerIndex player) const;
  std::vector<VectorXf> LinearizingControls(
      const VectorXf& x, const std::vector<VectorXf>& vs) const;
  VectorXf ToLinearSystemState(const VectorXf& x) const;
  VectorXf ToLinearSystemState(const VectorXf& x,
                               PlayerIndex subsystem_idx) const;
  VectorXf FromLinearSystemState(const VectorXf& xi) const;
  VectorXf FromLinearSystemState(const VectorXf& xi,
                                 PlayerIndex subsystem_idx) const;
  bool IsLinearSystemStateSingular(const VectorXf& xi) const;

  // Get the subset of this full state corresponding to the given subsystem.
  VectorXf SubsystemStates(const VectorXf& x, PlayerIndex subsystem_idx) const;

  // Deprecated utilities for changing cost coordinates.
  void ChangeCostCoordinates(const VectorXf& xi,
                             std::vector<QuadraticCostApproximation>* q) const;
  void ChangeControlCostCoordinates(
      const VectorXf& xi, std::vector<QuadraticCostApproximation>* q) const;

  // Distance metric between two states.
  float DistanceBetween(const VectorXf& x0, const VectorXf& x1) const;

  // Getters.
  const FlatSubsystemList& Subsystems() const { return subsystems_; }
  PlayerIndex NumPlayers() const { return subsystems_.size(); }
  Dimension SubsystemXDim(PlayerIndex player_idx) const {
    return subsystems_[player_idx]->XDim();
  }
  Dimension SubsystemStartDim(PlayerIndex player_idx) const {
    return subsystem_start_dims_[player_idx];
  }
  Dimension UDim(PlayerIndex player_idx) const {
    return subsystems_[player_idx]->UDim();
  }
  std::vector<Dimension> PositionDimensions() const;

 private:
  // List of subsystems, each of which controls the affects of a single player.
  const FlatSubsystemList subsystems_;

  // Cumulative sum of dimensions of each subsystem.
  std::vector<Dimension> subsystem_start_dims_;
};  // namespace ilqgames

}  // namespace ilqgames

#endif

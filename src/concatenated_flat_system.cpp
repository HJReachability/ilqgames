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

#include <ilqgames/dynamics/concatenated_flat_system.h>
#include <ilqgames/utils/linear_dynamics_approximation.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>

namespace ilqgames {

ConcatenatedFlatSystem::ConcatenatedFlatSystem(
    const FlatSubsystemList& subsystems, Time time_step)
    : MultiPlayerFlatSystem(std::accumulate(
          subsystems.begin(), subsystems.end(), 0,
          [](Dimension total,
             const std::shared_ptr<SinglePlayerFlatSystem>& subsystem) {
            CHECK_NOTNULL(subsystem.get());
            return total + subsystem->XDim();
          }), time_step),
      subsystems_(subsystems) {}

VectorXf ConcatenatedFlatSystem::Evaluate(
    const VectorXf& x, const std::vector<VectorXf>& us) const {
  CHECK_EQ(us.size(), NumPlayers());

  // Populate 'xdot' one subsystem at a time.
  VectorXf xdot(xdim_);
  Dimension dims_so_far = 0;
  for (size_t ii = 0; ii < NumPlayers(); ii++) {
    const auto& subsystem = subsystems_[ii];
    xdot.segment(dims_so_far, subsystem->XDim()) = subsystem->Evaluate(
        t, x.segment(dims_so_far, subsystem->XDim()), us[ii]);
    dims_so_far += subsystem->XDim();
  }

  return xdot;
}

LinearDynamicsApproximation ConcatenatedFlatSystem::LinearizedSystem(
    Time time_step) const {
  // Populate a block-diagonal A, as well as Bs.
  LinearDynamicsApproximation linearization(*this);

  Dimension dims_so_far = 0;
  for (size_t ii = 0; ii < NumPlayers(); ii++) {
    const auto& subsystem = subsystems_[ii];
    const Dimension xdim = subsystem->XDim();
    const Dimension udim = subsystem->UDim();
    subsystem->LinearizedSystem(
        time_step,
        linearization.A.block(dims_so_far, dims_so_far, xdim, xdim),
        linearization.Bs[ii].block(dims_so_far, 0, xdim, udim));

    dims_so_far += xdim;  
  }

  return linearization;
}

MatrixXf ConcatenatedFlatSystem::InverseDecouplingMatrix(const VectorXf& x) const { 
  // Populate a block-diagonal inverse decoupling matrix M. 
  MatrixXf M = MatrixXf::Zero(TotalUDim(), TotalUDim());

  Dimension x_dims_so_far = 0;
  Dimension u_dims_so_far = 0;
  for (size_t ii = 0; ii < NumPlayers(); ii++) {
    const auto& subsystem = subsystems_[ii];
    const Dimension xdim = subsystem->XDim();
    const Dimension udim = subsystem->UDim();
    M.block(u_dims_so_far, u_dims_so_far, udim, udim) =
      subsystem->InverseDecouplingMatrix(x.segment(x_dims_so_far,x_dim));

    x_dims_so_far += xdim;  
    u_dims_so_far += udim;
  }

  return M;
}

VectorXf ConcatenatedFlatSystem::AffineTerm(const VectorXf& x) const{
  // Populate the affine term m for each subsystem. 
  VectorXf m(TotalUDim());

  Dimension x_dims_so_far = 0;
  Dimension u_dims_so_far = 0;
  for (size_t ii = 0; ii < NumPlayers(); ii++) {
    const auto& subsystem = subsystems_[ii];
    const Dimension xdim = subsystem->XDim();
    const Dimension udim = subsystem->UDim();
    m.segment(u_dims_so_far, udim) =
      subsystem->AffineTerm(x.segment(x_dims_so_far,x_dim));

    x_dims_so_far += xdim;  
    u_dims_so_far += udim;
  }

  return m;
}

VectorXf ConcatenatedFlatSystem::LinearizingControl(const VectorXf& x,
                            const VectorXf& v) const{
  VectorXf u(TotalUDim());

  Dimension x_dims_so_far = 0;
  Dimension u_dims_so_far = 0;
  for (size_t ii = 0; ii < NumPlayers(); ii++) {
    const auto& subsystem = subsystems_[ii];
    const Dimension xdim = subsystem->XDim();
    const Dimension udim = subsystem->UDim();
    u.segment(u_dims_so_far, udim) =
      subsystem->LinearizingControl(x.segment(x_dims_so_far,x_dim),
                                    v.segment(u_dims_so_far,u_dim));

    x_dims_so_far += xdim;  
    u_dims_so_far += udim;
  }

  return u; 
}

VectorXf ConcatenatedFlatSystem::ToLinearSystemState(const VectorXf& x) const{
  VectorXf xi(TotalXDim());

  Dimension x_dims_so_far = 0;
  for (size_t ii = 0; ii < NumPlayers(); ii++) {
    const auto& subsystem = subsystems_[ii];
    const Dimension xdim = subsystem->XDim();
    xi.segment(x_dims_so_far, xdim) =
      subsystem->ToLinearSystemState(x.segment(x_dims_so_far,x_dim));

    x_dims_so_far += xdim;
  }

  return xi;   
}

VectorXf ConcatenatedFlatSystem::FromLinearSystemState(const VectorXf& xi) const{
  VectorXf x(TotalXDim());

  Dimension x_dims_so_far = 0;
  for (size_t ii = 0; ii < NumPlayers(); ii++) {
    const auto& subsystem = subsystems_[ii];
    const Dimension xdim = subsystem->XDim();
    x.segment(x_dims_so_far, xdim) =
      subsystem->FromLinearSystemState(xi.segment(x_dims_so_far,x_dim));

    x_dims_so_far += xdim;
  }

  return x;   
}

// Gradient and hessian of map from xi to x.
void ConcatenatedFlatSystem::GradientAndHessianXi(const VectorXf& xi, Eigen::Ref<VectorXf> grad,
                          Eigen::Ref<MatrixXf> hess) const{

}

}  // namespace ilqgames

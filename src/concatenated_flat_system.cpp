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
    : MultiPlayerFlatSystem(
          std::accumulate(
              subsystems.begin(), subsystems.end(), 0,
              [](Dimension total,
                 const std::shared_ptr<SinglePlayerFlatSystem>& subsystem) {
                CHECK_NOTNULL(subsystem.get());
                return total + subsystem->XDim();
              }),
          time_step),
      subsystems_(subsystems) {}

VectorXf ConcatenatedFlatSystem::Evaluate(
    const VectorXf& x, const std::vector<VectorXf>& us) const {
  CHECK_EQ(us.size(), NumPlayers());

  // Populate 'xdot' one subsystem at a time.
  VectorXf xdot(xdim_);
  Dimension dims_so_far = 0;
  for (size_t ii = 0; ii < NumPlayers(); ii++) {
    const auto& subsystem = subsystems_[ii];
    xdot.segment(dims_so_far, subsystem->XDim()) =
        subsystem->Evaluate(x.segment(dims_so_far, subsystem->XDim()), us[ii]);
    dims_so_far += subsystem->XDim();
  }

  return xdot;
}

void ConcatenatedFlatSystem::ComputeLinearizedSystem() const {
  // Populate a block-diagonal A, as well as Bs.
  LinearDynamicsApproximation linearization(*this);

  Dimension dims_so_far = 0;
  for (size_t ii = 0; ii < NumPlayers(); ii++) {
    const auto& subsystem = subsystems_[ii];
    const Dimension xdim = subsystem->XDim();
    const Dimension udim = subsystem->UDim();
    subsystem->LinearizedSystem(
        time_step_, linearization.A.block(dims_so_far, dims_so_far, xdim, xdim),
        linearization.Bs[ii].block(dims_so_far, 0, xdim, udim));

    dims_so_far += xdim;
  }

  discrete_linear_system_.reset(new LinearDynamicsApproximation(linearization));

  // Reconstruct the continuous system.
  linearization.A -= MatrixXf::Identity(xdim_, xdim_);
  linearization.A /= time_step_;
  for (size_t ii = 0; ii < NumPlayers(); ii++)
    linearization.Bs[ii] /= time_step_;
  continuous_linear_system_.reset(
      new LinearDynamicsApproximation(linearization));
}

MatrixXf ConcatenatedFlatSystem::InverseDecouplingMatrix(
    const VectorXf& x) const {
  // Populate a block-diagonal inverse decoupling matrix M.
  MatrixXf M = MatrixXf::Zero(TotalUDim(), TotalUDim());

  Dimension x_dims_so_far = 0;
  Dimension u_dims_so_far = 0;
  for (size_t ii = 0; ii < NumPlayers(); ii++) {
    const auto& subsystem = subsystems_[ii];
    const Dimension xdim = subsystem->XDim();
    const Dimension udim = subsystem->UDim();
    M.block(u_dims_so_far, u_dims_so_far, udim, udim) =
        subsystem->InverseDecouplingMatrix(x.segment(x_dims_so_far, xdim));

    x_dims_so_far += xdim;
    u_dims_so_far += udim;
  }

  return M;
}

VectorXf ConcatenatedFlatSystem::AffineTerm(const VectorXf& x) const {
  // Populate the affine term m for each subsystem.
  VectorXf m(TotalUDim());

  Dimension x_dims_so_far = 0;
  Dimension u_dims_so_far = 0;
  for (size_t ii = 0; ii < NumPlayers(); ii++) {
    const auto& subsystem = subsystems_[ii];
    const Dimension xdim = subsystem->XDim();
    const Dimension udim = subsystem->UDim();
    m.segment(u_dims_so_far, udim) =
        subsystem->AffineTerm(x.segment(x_dims_so_far, xdim));

    x_dims_so_far += xdim;
    u_dims_so_far += udim;
  }

  return m;
}

VectorXf ConcatenatedFlatSystem::LinearizingControl(const VectorXf& x,
                                                    const VectorXf& v) const {
  VectorXf u(TotalUDim());

  Dimension x_dims_so_far = 0;
  Dimension u_dims_so_far = 0;
  for (size_t ii = 0; ii < NumPlayers(); ii++) {
    const auto& subsystem = subsystems_[ii];
    const Dimension xdim = subsystem->XDim();
    const Dimension udim = subsystem->UDim();
    u.segment(u_dims_so_far, udim) = subsystem->LinearizingControl(
        x.segment(x_dims_so_far, xdim), v.segment(u_dims_so_far, udim));

    x_dims_so_far += xdim;
    u_dims_so_far += udim;
  }

  return u;
}

VectorXf ConcatenatedFlatSystem::ToLinearSystemState(const VectorXf& x) const {
  VectorXf xi(xdim_);

  Dimension x_dims_so_far = 0;
  for (size_t ii = 0; ii < NumPlayers(); ii++) {
    const auto& subsystem = subsystems_[ii];
    const Dimension xdim = subsystem->XDim();
    xi.segment(x_dims_so_far, xdim) =
        subsystem->ToLinearSystemState(x.segment(x_dims_so_far, xdim));

    x_dims_so_far += xdim;
  }

  return xi;
}

VectorXf ConcatenatedFlatSystem::FromLinearSystemState(
    const VectorXf& xi) const {
  VectorXf x(xdim_);

  Dimension x_dims_so_far = 0;
  for (size_t ii = 0; ii < NumPlayers(); ii++) {
    const auto& subsystem = subsystems_[ii];
    const Dimension xdim = subsystem->XDim();
    x.segment(x_dims_so_far, xdim) =
        subsystem->FromLinearSystemState(xi.segment(x_dims_so_far, xdim));

    x_dims_so_far += xdim;
  }

  return x;
}

// Gradient and hessian of map from xi to x.
// void ConcatenatedFlatSystem::GradientAndHessianXi(const VectorXf& xi,
//                                                   VectorXf* grad,
//                                                   MatrixXf* hess) const {
//   // Populate a gradient and hessian.
//   CHECK_NOTNULL(grad);
//   CHECK_NOTNULL(hess);

//   // Initialize with zeros.
//   *grad = VectorXf::Zero(xi.size());
//   *hess = MatrixXf::Zero(xi.size(), xi.size());

//   Dimension x_dims_so_far = 0;
//   for (PlayerIndex ii = 0; ii < NumPlayers(); ii++) {
//     const auto& subsystem = subsystems_[ii];
//     const Dimension xdim = subsystem->XDim();
//     subsystem->GradientAndHessianXi(
//         xi.segment(x_dims_so_far, xdim), grad->segment(x_dims_so_far, xdim),
//         hess->block(x_dims_so_far, x_dims_so_far, xdim, xdim));

//     x_dims_so_far += xdim;
//   }
// }

void ConcatenatedFlatSystem::ChangeCostCoordinates(
    const VectorXf& xi, const std::vector<VectorXf>& vs,
    std::vector<QuadraticCostApproximation>* q) const {
  CHECK_NOTNULL(q);

  std::vector<std::vector<VectorXf>> first_partials;
  std::vector<std::vector<MatrixXf>> second_partials;
  Dimension xi_dims_so_far = 0;
  for(size_t ii=0; ii < NumPlayers(); ii++){
    const auto& subsystem_ii = subsystems_[ii];
    const Dimension xdim = subsystem_ii->XDim();
    subsystem_ii->Partial(xi.segment(xi_dims_so_far,xdim),&first_partials[ii],&second_partials[ii]);
    xi_dims_so_far += xdim;
  }

  // For loop for hessian.
  // Vector that is going to contain all the cost hessians for each player.
  std::vector<MatrixXf> hess_xs(q->size(), MatrixXf::Zero(xdim_, xdim_));
  Dimension rows_so_far = 0;

  // Iterating over primary player indexes.
  for (PlayerIndex pp = 0; pp < NumPlayers(); pp++) {
    Dimension cols_so_far = rows_so_far;
    // Iterating over that player's number of dimensions.
    for (Dimension ii = 0; ii < SubsystemXDim(pp); ii++) {
      const Dimension ii_rows = ii + rows_so_far;

      // Iterating over secondary player indexes.
      for (PlayerIndex qq = pp; qq < NumPlayers(); qq++) {
        // Iterating over that player's number of dimensions.
        for (Dimension jj = 0; jj < SubsystemXDim(qq); jj++) {
          const Dimension jj_cols = jj + cols_so_far;

          // Iterating over each player's cost.
          for (PlayerIndex rr = 0; rr < NumPlayers(); rr++) {
            const MatrixXf& Q = (*q)[rr].Q;
            const VectorXf& l = (*q)[rr].l;
            MatrixXf& xhess = hess_xs[rr];
            for (Dimension kk = 0; kk < SubsystemXDim(pp); kk++) {
              if (pp == qq) {
                xhess(ii_rows, jj_cols) +=
                    l(kk + rows_so_far) * second_partials[pp][kk](ii,jj);
              }

              float tmp = 0.0;
              for (Dimension ll = 0; ll < SubsystemXDim(qq); ll++) {
                tmp += Q(kk + rows_so_far, ll + cols_so_far) *
                       first_partials[qq][ll](jj);
              }

              xhess(ii_rows, jj_cols) +=
                  tmp * first_partials[pp][kk](ii);
            }
          }

          // Increment columns so far to track next subsystem.
          cols_so_far += SubsystemXDim(qq);
        }
      }

      // Increment rows so far to track next subsystem.
      rows_so_far += SubsystemXDim(pp);
    }
  }

  // For loop for gradient.
  // Vector that is going to contain all the cost gradients for each player.
  std::vector<VectorXf> grad_xs(q->size(), VectorXf::Zero(xdim_));
  rows_so_far = 0;

  // Iterating over primary player indexes.
  for (PlayerIndex pp = 0; pp < NumPlayers(); pp++) {
    // Iterating over that player's number of dimensions.
    for (Dimension ii = 0; ii < SubsystemXDim(pp); ii++) {
      const VectorXf& l = (*q)[ii].l;

      // Iterating over each player's cost.
      for (Dimension jj = 0; jj < SubsystemXDim(pp); jj++) {
        grad_xs[pp](ii) += l(ii+rows_so_far) * first_partials[pp][jj](ii);
      }

    }
    // Increment rows so far to track next subsystem.
    rows_so_far += SubsystemXDim(pp);
  }

  // Now modify the cost hessians, i.e. 'Rs'.
  // NOTE: this depends only on the decoupling matrix.
  const VectorXf x = FromLinearSystemState(xi);
  const Dimension total_u_dim = TotalUDim();
  std::vector<MatrixXf> M_invs(NumPlayers());

  Dimension x_dims_so_far = 0;
  for(size_t ii=0; ii < NumPlayers(); ii++){
    const auto& subsystem_ii = subsystems_[ii];
    const Dimension xdim = subsystem_ii->XDim();
    M_invs[ii] = subsystem_ii->InverseDecouplingMatrix(x.segment(x_dims_so_far, xdim));
    x_dims_so_far += xdim;
  }

  for (size_t ii = 0; ii < NumPlayers(); ii++) {
    for (auto& element : (*q)[ii].Rs) {
      element.second =
        M_invs[element.first].transpose() * element.second * M_invs[element.first]; 
    }
  }

}

}  // namespace ilqgames

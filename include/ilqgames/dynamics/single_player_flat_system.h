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
// Base class for all single-player flat (square) systems. Supports (discrete-
// time) linearization.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_DYNAMICS_FLAT_SYSTEM_H
#define ILQGAMES_DYNAMICS_FLAT_SYSTEM_H

#include <ilqgames/utils/types.h>

namespace ilqgames {

class SinglePlayerFlatSystem {
 public:
  virtual ~SinglePlayerFlatSystem() {}

  // Compute time derivative of state.
  virtual VectorXf Evaluate(const VectorXf& x, const VectorXf& u) const = 0;

  // Discrete time approximation of the underlying linearized system.
  virtual void LinearizedSystem(Eigen::Ref<MatrixXf> A,
                                Eigen::Ref<MatrixXf> B) const = 0;

  // Utilities for feedback linearization.
  virtual MatrixXf InverseDecouplingMatrix(const VectorXf& x) const = 0;
  virtual VectorXf AffineTerm(const VectorXf& x) const = 0;
  VectorXf LinearizingControl(const VectorXf& x, const VectorXf& v) const {
    return InverseDecouplingMatrix(x) * (v - AffineTerm(x));
  }
  virtual VectorXf ToLinearSystemState(const VectorXf& x) const = 0;
  virtual VectorXf FromLinearSystemState(const VectorXf& xi) const = 0;
  virtual bool IsLinearSystemStateSingular(const VectorXf& xi) const = 0;

  // Partial derivatives of map from xi to x.
  virtual void Partial(const VectorXf& xi, std::vector<VectorXf>* grads,
                       std::vector<MatrixXf>* hesses) const = 0;

  // Distance metric on the state space. By default, just the *squared* 2-norm.
  virtual float DistanceBetween(const VectorXf& x0, const VectorXf& x1) const {
    return (x0 - x1).squaredNorm();
  }

  // Getters.
  Dimension XDim() const { return xdim_; }
  Dimension UDim() const { return udim_; }
  virtual std::vector<Dimension> PositionDimensions() const = 0;

 protected:
  SinglePlayerFlatSystem(Dimension xdim, Dimension udim)
      : xdim_(xdim), udim_(udim) {}

  // Dimensions.
  const Dimension xdim_;
  const Dimension udim_;
};  //\class SinglePlayerDynamicalSystem

}  // namespace ilqgames

#endif

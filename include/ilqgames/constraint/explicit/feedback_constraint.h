/*
 * Copyright (c) 2020, The Regents of the University of California (Regents).
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
// (Time-varying) feedback constraint,
// i.e., 0.5*||u_t^i - gamma(x_t; theta_t^i)||^2 = 0.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_CONSTRAINT_FEEDBACK_CONSTRAINT_H
#define ILQGAMES_CONSTRAINT_FEEDBACK_CONSTRAINT_H

#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/quadratic_constraint_approximation.h>
#include <ilqgames/utils/relative_time_tracker.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <memory>
#include <string>

namespace ilqgames {

class FeedbackConstraint : public RelativeTimeTracker {
 public:
  ~FeedbackConstraint() {}
  FeedbackConstraint(const StrategyRef* strategy_ref,
                     const std::string& name = "")
      : RelativeTimeTracker(name), strategy_ref_(strategy_ref) {
    CHECK_NOTNULL(strategy_ref_);
  }

  // Check if this constraint is satisfied, and optionally return the constraint
  // value, which equals zero if the constraint is satisfied.
  bool IsSatisfied(size_t time_step, const VectorXf& x, const VectorXf& u,
                   float* level) {
    const float value =
        0.5 * (u - (*strategy_ref_)(time_step, x)).squaredNorm();
    if (*level) *level = value;

    return std::abs(value) < constants::kSmallNumber;
  }

  // Quadraticize the constraint value. Do *not* keep a running sum since we
  // keep separate multipliers for each constraint.
  void Quadraticize(size_t time_step, const VectorXf& x, const VectorXf& u,
                    Eigen::Ref<MatrixXf> hess_x, Eigen::Ref<MatrixXf> hess_u,
                    Eigen::Ref<MatrixXf> hess_strategy,
                    Eigen::Ref<MatrixXf> hess_xu, Eigen::Ref<MatrixXf> hess_ux,
                    Eigen::Ref<MatrixXf> hess_xstrategy,
                    Eigen::Ref<MatrixXf> hess_strategyx,
                    Eigen::Ref<MatrixXf> hess_ustrategy,
                    Eigen::Ref<MatrixXf> hess_strategyu,
                    Eigen::Ref<VectorXf> grad_x, Eigen::Ref<VectorXf> grad_u,
                    Eigen::Ref<VectorXf> grad_strategy) const {
    // NOTE: assuming that all the dimensions are correct, just because checking
    // would be a lot of unnecessary operations, but eventually these should be
    // factored into DCHECKs.

    // Compute mismatch vector.
    const VectorXf error = u - (*strategy_ref_)(time_step, x);

    // Unpack P and alpha.
    const auto& P = strategy_refs_->Ps[time_step];
    const auto& alpha = strategy_refs_->alphas[time_step];

    // (1) Handle x terms.
    hess_x = -P.transpose() * P;
    grad_x = -P.transpose() * error;

    // (2) Handle u terms.
    hess_u.setIdentity();
    grad_u = error;

    // (3) Handle strategy terms.
    const size_t num_Ps = strategy_refs_->SizeP();
    const size_t num_alphas = strategy_refs_->SizeAlpha();

    // P Hessian.
    for (size_t kk = 0; kk < P.cols(); kk++) {
      for (size_t jj = kk; jj < P.cols(); jj++) {
        for (size_t ii = 0; ii < P.rows(); ii++) {
          const size_t row = ii + kk * P.rows();
          const size_t col = ii + jj * P.rows();

          hess_strategy(row, col) = x(jj) * x(kk);
          if (jj == kk) hess_strategy(row, col) -= error(ii);

          hess_strategy(col, row) = hess_strategy(row, col);
        }
      }
    }

    // P gradient.
    for (size_t jj = 0; jj < P.cols(); jj++) {
      const size_t offset =
          jj * P.rows();  // Eigen::Map uses column-major storage by default.
      for (size_t ii = 0; ii < P.rows(); ii++)
        grad_strategy[offset + ii] =
            (u(ii) - P.row(ii) * x - alpha(ii)) * (-x(jj));
    }

    // Alpha Hessian.
    hess_strategy.block(num_Ps, num_Ps, num_alphas, num_alphas).setIdentity();

    // P alpha cross terms. Assumes already set to zero so only accessing the
    // nonzero elements.
    // Top-right and bottom-left blocks.
    Eigen::Ref<MatrixXf> hess_tr =
        hess_strategy.topRightCorner(num_Ps, num_alphas);
    Eigen::Ref<MatrixXf> hess_bl =
        hess_strategy.bottomLeftCorner(num_alphas, num_Ps);
    for (size_t jj = 0; jj < Ps.cols(); jj++) {
      const size_t offset = jj * P.rows();
      for (size_t ii = 0; ii < Ps.rows(); ii++) {
        const size_t P_idx = offset + ii;  // Alpha idx is just ii.
        hess_tr(P_idx, ii) = alpha(ii) * x(jj);
        hess_bl(ii, P_idx) = hess_tr(P_idx, ii);
      }
    }

    // Alpha gradient.
    grad_strategy.tail(num_alphas) = -error;

    // (4) Handle xu terms.
    hess_ux = P;
    hess_xu = hess_ux.transpose();

    // (5) Handle xfeedback terms (xP and xalpha separately).
    for (size_t jj = 0; jj < P.cols(); jj++) {
      for (size_t ii = 0; ii < P.rows(); ii++) {
        for (size_t kk = 0; kk < x.size(); kk++) {
          const size_t P_idx = ii + jj * P.rows();  // X idx is just kk.

          hess_xfeedback(kk, P_idx) = P(ii, kk) * x(jj);
          if (jj == kk) hess_xfeedback(kk, P_idx) -= error(ii);

          hess_feedbackx(P_idx, kk) = hess_xfeedback(kk, P_idx);
        }
      }
    }

    Eigen::Ref<MatrixXf> hess_xalpha = hess_xfeedback.rightCols(num_alphas);
    Eigen::Ref<MatrixXf> hess_alphax = hess_feedbackx.bottomRows(num_alphas);
    hess_xalpha = P.transpose();
    hess_alphax = hess_xalpha.transpose();

    // (6) Handle ufeedback terms (uP and ualpha separately again).
    for (size_t jj = 0; jj < P.cols(); jj++) {
      const size_t offset = jj * P.rows();
      for (size_t ii = 0; ii < P.rows(); ii++) {
        const size_t P_idx = ii + jj * P.rows();  // Alpha idx is just ii.
        hess_ufeedback(ii, P_idx) = -x(jj);
        hess_feedbacku(P_idx, ii) = hess_ufeedback(ii, P_idx);
      }
    }

    Eigen::Ref<MatrixXf> hess_ualpha = hess_ufeedback.rightCols(num_alphas);
    Eigen::Ref<MatrixXf> hess_alphau = hess_feedbacku.bottomRows(num_alphas);
    hess_ualpha.SetIdentity() *= -1.0;
    hess_alphau.SetIdentity() *= -1.0;
  }

 private:
  // Strategy of a single player.
  const StrategyRef* strategy_ref_;
};  //\class DynamicConstraint

}  // namespace ilqgames

#endif

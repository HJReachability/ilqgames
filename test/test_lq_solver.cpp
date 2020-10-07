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
// Test SolveLQGame by comparing to an implementation of Lyapunov iterations
// and ensuring that the solutions agree on a two-player time-invariant
// long-horizon example.
//
// Adapted from original Python implementation of Lyapunov iterations written
// by Eric Mazumdar (2018). Algorithm may be found at:
// https://link.springer.com/chapter/10.1007/978-1-4612-4274-1_17
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/cost/player_cost.h>
#include <ilqgames/cost/quadratic_cost.h>
#include <ilqgames/dynamics/concatenated_dynamical_system.h>
#include <ilqgames/dynamics/multi_player_dynamical_system.h>
#include <ilqgames/dynamics/multi_player_integrable_system.h>
#include <ilqgames/dynamics/single_player_dynamical_system.h>
#include <ilqgames/solver/lq_feedback_solver.h>
#include <ilqgames/solver/lq_open_loop_solver.h>
#include <ilqgames/utils/check_local_nash_equilibrium.h>
#include <ilqgames/utils/linear_dynamics_approximation.h>
#include <ilqgames/utils/quadratic_cost_approximation.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

#include <gtest/gtest.h>
#include <math.h>

using namespace ilqgames;

namespace {

// Solve two-player infinite horizon (time-invariant) LQ game by Lyapunov
// iterations.
void SolveLyapunovIterations(const MatrixXf& A, const MatrixXf& B1,
                             const MatrixXf& B2, const MatrixXf& Q1,
                             const MatrixXf& Q2, const MatrixXf& R11,
                             const MatrixXf& R12, const MatrixXf& R21,
                             const MatrixXf& R22, Eigen::Ref<MatrixXf> P1,
                             Eigen::Ref<MatrixXf> P2) {
  // Number of iterations.
  constexpr size_t kNumIterations = 100;

  // Initialize Zs to Qs.
  MatrixXf Z1 = Q1;
  MatrixXf Z2 = Q2;

  // Initialize Ps.
  P1 = (R11 + B1.transpose() * Z1 * B1)
           .householderQr()
           .solve(B1.transpose() * Z1 * A);
  P2 = (R22 + B2.transpose() * Z2 * B2)
           .householderQr()
           .solve(B2.transpose() * Z2 * A);

  for (size_t ii = 0; ii < kNumIterations; ii++) {
    const MatrixXf old_P1 = P1;
    const MatrixXf old_P2 = P2;

    P1 = (R11 + B1.transpose() * Z1 * B1)
             .householderQr()
             .solve(B1.transpose() * Z1 * (A - B2 * old_P2));
    P2 = (R22 + B2.transpose() * Z2 * B2)
             .householderQr()
             .solve(B2.transpose() * Z2 * (A - B1 * old_P1));

    Z1 = (A - B1 * P1 - B2 * P2).transpose() * Z1 * (A - B1 * P1 - B2 * P2) +
         P1.transpose() * R11 * P1 + P2.transpose() * R12 * P2 + Q1;
    Z2 = (A - B1 * P1 - B2 * P2).transpose() * Z2 * (A - B1 * P1 - B2 * P2) +
         P1.transpose() * R21 * P1 + P2.transpose() * R22 * P2 + Q2;
  }
}

// Utility class for single player system.
class SinglePlayerUtilityDynamics : public SinglePlayerDynamicalSystem {
 public:
  ~SinglePlayerUtilityDynamics() {}
  SinglePlayerUtilityDynamics() : SinglePlayerDynamicalSystem(2, 2) {
    CHECK_EQ(xdim_, udim_);
    A_ = MatrixXf::Identity(xdim_, xdim_);
    B_ = time::kTimeStep * 0.41 * MatrixXf::Identity(xdim_, xdim_);
    A_(0, 1) = time::kTimeStep;
  }

  VectorXf Evaluate(Time t, const VectorXf& x, const VectorXf& u) const {
    return (A_ - MatrixXf::Identity(xdim_, xdim_)) * x / time::kTimeStep +
           B_ * u / time::kTimeStep;
  }

  void Linearize(Time t, const VectorXf& x, const VectorXf& u,
                 Eigen::Ref<MatrixXf> A, Eigen::Ref<MatrixXf> B) const {
    A = A_;
    B = B_;
  }

  std::vector<Dimension> PositionDimensions() const { return {0}; }

  const MatrixXf& A() const { return A_; }
  const MatrixXf& B() const { return B_; }

 private:
  MatrixXf A_, B_;
};  // class SinglePlayerUtilityDynamics

// Utility class for time-invariant linear system.
class TwoPlayerPointMass1D : public MultiPlayerDynamicalSystem {
 public:
  ~TwoPlayerPointMass1D() {}
  TwoPlayerPointMass1D()
      : MultiPlayerDynamicalSystem(2), A_(2, 2), B1_(2), B2_(2) {
    A_ = MatrixXf::Zero(2, 2);
    A_(0, 1) = 1.0;

    B1_(0) = 0.05;
    B1_(1) = 1.0;

    B2_(0) = 0.032;
    B2_(1) = 0.11;
  }

  // Getters.
  Dimension UDim(PlayerIndex player_index) const { return 1; }
  PlayerIndex NumPlayers() const { return 2; }

  // Time derivative of state.
  VectorXf Evaluate(Time t, const VectorXf& x,
                    const std::vector<VectorXf>& us) const {
    return A_ * x + B1_ * us[0] + B2_ * us[1];
  }

  // Discrete-time Jacobian linearization.
  LinearDynamicsApproximation Linearize(Time t, const VectorXf& x,
                                        const std::vector<VectorXf>& us) const {
    LinearDynamicsApproximation linearization(*this);

    linearization.A += A_ * time::kTimeStep;
    linearization.Bs[0] = B1_ * time::kTimeStep;
    linearization.Bs[1] = B2_ * time::kTimeStep;
    return linearization;
  }

  // Position dimensions.
  std::vector<Dimension> PositionDimensions() const { return {0}; }

 private:
  // Continuous-time dynamics.
  MatrixXf A_;
  VectorXf B1_, B2_;
};  // class TwoPlayerPointMass1D

// Provide a default constructor to solvers.
template <typename T>
class ProvideDefaultConstructor : public T {
 public:
  ProvideDefaultConstructor()
      : T(std::make_shared<TwoPlayerPointMass1D>(), time::kNumTimeSteps) {}
};  // class ProvideDefaultConstructor

}  // anonymous namespace

template <typename T>
class LQSolverTest : public ::testing::Test {
 protected:
  void SetUp() {
    dynamics_.reset(new TwoPlayerPointMass1D);
    x0_ = VectorXf::Ones(2);

    // Set linearization and quadraticizations.
    linearization_ = dynamics_->Linearize(
        0.0, VectorXf::Zero(2), {VectorXf::Zero(1), VectorXf::Zero(1)});

    // Set a zero operating point.
    operating_point_.reset(
        new OperatingPoint(time::kNumTimeSteps, dynamics_->NumPlayers(), 0.0));
    for (size_t kk = 0; kk < time::kNumTimeSteps; kk++) {
      operating_point_->xs[kk] =
          MatrixXf::Zero(dynamics_->XDim(), dynamics_->XDim());
      for (PlayerIndex ii = 0; ii < dynamics_->NumPlayers(); ii++)
        operating_point_->us[kk][ii] = VectorXf::Zero(dynamics_->UDim(ii));
    }

    // Set up corresponding player costs.
    ConstructCostsWithNominal(0.0);

    // Solve LQ game.
    QuadraticizeAndSolve();
  }

  // Reset with nonzero nominal state and control.
  void ConstructCostsWithNominal(float nominal = 0.0) {
    player_costs_.clear();
    constexpr float kRelativeCostScaling = 0.1;

    PlayerCost player1_cost;
    player1_cost.AddStateCost(
        std::make_shared<QuadraticCost>(1.0, -1, nominal));
    player1_cost.AddControlCost(
        0, std::make_shared<QuadraticCost>(1.0, -1, nominal));
    player1_cost.AddControlCost(
        1, std::make_shared<QuadraticCost>(kRelativeCostScaling, -1, nominal));
    player_costs_.push_back(player1_cost);

    PlayerCost player2_cost;
    player2_cost.AddStateCost(
        std::make_shared<QuadraticCost>(kRelativeCostScaling, -1, nominal));
    player2_cost.AddControlCost(
        0, std::make_shared<QuadraticCost>(kRelativeCostScaling, -1, nominal));
    player2_cost.AddControlCost(
        1, std::make_shared<QuadraticCost>(1.0, -1, nominal));
    player_costs_.push_back(player2_cost);
  }

  // Quadraticize and solve.
  void QuadraticizeAndSolve() {
    quadraticizations_.clear();
    quadraticizations_.push_back(player_costs_[0].Quadraticize(
        0.0, operating_point_->xs[0], operating_point_->us[0]));
    quadraticizations_.push_back(player_costs_[1].Quadraticize(
        0.0, operating_point_->xs[0], operating_point_->us[0]));

    lq_solution_ =
        lq_solver_.Solve(std::vector<LinearDynamicsApproximation>(
                             time::kNumTimeSteps, linearization_),
                         std::vector<std::vector<QuadraticCostApproximation>>(
                             time::kNumTimeSteps, quadraticizations_),
                         x0_);
  }

  // Dynamics.
  std::unique_ptr<TwoPlayerPointMass1D> dynamics_;

  // Operating point.
  std::unique_ptr<OperatingPoint> operating_point_;

  // Initial state.
  VectorXf x0_;

  // Player costs.
  std::vector<PlayerCost> player_costs_;

  // Linearization and quadraticization.
  LinearDynamicsApproximation linearization_;
  std::vector<QuadraticCostApproximation> quadraticizations_;

  // Core LQ solver.
  ProvideDefaultConstructor<T> lq_solver_;

  // Solution to LQ game.
  std::vector<Strategy> lq_solution_;
};  // class LQGameTest

class LQFeedbackSolverTest : public LQSolverTest<LQFeedbackSolver> {};
class LQOpenLoopSolverTest : public LQSolverTest<LQOpenLoopSolver> {};

TEST_F(LQFeedbackSolverTest, MatchesLyapunovIterations) {
  const MatrixXf& A = linearization_.A;
  const MatrixXf& B1 = linearization_.Bs[0];
  const MatrixXf& B2 = linearization_.Bs[1];

  const MatrixXf& Q1 = quadraticizations_[0].state.hess;
  const MatrixXf& Q2 = quadraticizations_[1].state.hess;
  const VectorXf& l1 = quadraticizations_[0].state.grad;
  const VectorXf& l2 = quadraticizations_[1].state.grad;

  const MatrixXf& R11 = quadraticizations_[0].control.at(0).hess;
  const MatrixXf& R12 = quadraticizations_[0].control.at(1).hess;
  const MatrixXf& R21 = quadraticizations_[1].control.at(0).hess;
  const MatrixXf& R22 = quadraticizations_[1].control.at(1).hess;

  // Solve with Lyapunov iterations.
  MatrixXf P1(1, 2);
  MatrixXf P2(1, 2);
  SolveLyapunovIterations(A, B1, B2, Q1, Q2, R11, R12, R21, R22, P1, P2);

  // Check that the answers are close.
  EXPECT_LT((P1 - lq_solution_[0].Ps[0]).cwiseAbs().maxCoeff(),
            constants::kSmallNumber);
  EXPECT_LT((P2 - lq_solution_[1].Ps[0]).cwiseAbs().maxCoeff(),
            constants::kSmallNumber);
}

TEST_F(LQFeedbackSolverTest, NashEquilibrium) {
  // Make sure this is a feedback Nash and not an open loop Nash.
  constexpr float kMaxPerturbation = 0.1;
  EXPECT_TRUE(NumericalCheckLocalNashEquilibrium(player_costs_, lq_solution_,
                                                 *operating_point_, *dynamics_,
                                                 x0_, kMaxPerturbation));
  EXPECT_FALSE(NumericalCheckLocalNashEquilibrium(player_costs_, lq_solution_,
                                                  *operating_point_, *dynamics_,
                                                  x0_, kMaxPerturbation, true));
}

TEST_F(LQFeedbackSolverTest, NashEquilibriumWithLinearCostTerms) {
  // Reset with nonzero nominal values for state and control.
  ConstructCostsWithNominal(0.5);

  // Solve LQ game.
  QuadraticizeAndSolve();

  // Make sure this is a feedback Nash and not an open loop Nash.
  constexpr float kMaxPerturbation = 0.1;
  EXPECT_TRUE(NumericalCheckLocalNashEquilibrium(player_costs_, lq_solution_,
                                                 *operating_point_, *dynamics_,
                                                 x0_, kMaxPerturbation));
  EXPECT_FALSE(NumericalCheckLocalNashEquilibrium(player_costs_, lq_solution_,
                                                  *operating_point_, *dynamics_,
                                                  x0_, kMaxPerturbation, true));
}

TEST_F(LQOpenLoopSolverTest, NashEquilibrium) {
  // Reset with nonzero nominal values for state and control.
  ConstructCostsWithNominal(0.5);

  // Solve LQ game.
  QuadraticizeAndSolve();

  // const MatrixXf& A = linearization_.A;
  // const MatrixXf& B1 = linearization_.Bs[0];
  // const MatrixXf& B2 = linearization_.Bs[1];

  // const MatrixXf& Q1 = quadraticizations_[0].state.hess;
  // const MatrixXf& Q2 = quadraticizations_[1].state.hess;
  // const VectorXf& l1 = quadraticizations_[0].state.grad;
  // const VectorXf& l2 = quadraticizations_[1].state.grad;

  // const MatrixXf& R11 = quadraticizations_[0].control.at(0).hess;
  // const MatrixXf& R12 = quadraticizations_[0].control.at(1).hess;
  // const MatrixXf& R21 = quadraticizations_[1].control.at(0).hess;
  // const MatrixXf& R22 = quadraticizations_[1].control.at(1).hess;

  // std::cout << "A: " << A << std::endl;
  // std::cout << "B1: " << B1 << std::endl;
  // std::cout << "B2: " << B2 << std::endl;
  // std::cout << "Q1: " << Q1 << std::endl;
  // std::cout << "Q2: " << Q2 << std::endl;
  // std::cout << "l1: " << l1.transpose() << std::endl;
  // std::cout << "l2: " << l2.transpose() << std::endl;
  // std::cout << "R1: " << R11 << std::endl;
  // std::cout << "R2: " << R22 << std::endl;
  // std::cout << "r1: " << quadraticizations_[0].control.at(0).grad.transpose()
  //           << std::endl;
  // std::cout << "r2: " << quadraticizations_[1].control.at(1).grad.transpose()
  //           << std::endl;

  // Make sure this is an open loop Nash.
  constexpr float kMaxPerturbation = 0.1;
  EXPECT_TRUE(NumericalCheckLocalNashEquilibrium(player_costs_, lq_solution_,
                                                 *operating_point_, *dynamics_,
                                                 x0_, kMaxPerturbation, true));
}

TEST(SinglePlayerOpenLoopFeedback, TestSameSolution) {
  // Double integrator in discrete time.
  const SinglePlayerUtilityDynamics single;
  LinearDynamicsApproximation lin;
  lin.A = single.A();
  lin.Bs.push_back(single.B());

  const std::vector<LinearDynamicsApproximation> big_lin(time::kNumTimeSteps,
                                                         lin);

  // Cost structure to regulate to the origin.
  QuadraticCostApproximation quad(single.UDim());
  quad.state.hess = MatrixXf::Identity(single.XDim(), single.XDim());
  quad.control.emplace(0, SingleCostApproximation(
                              MatrixXf::Identity(single.UDim(), single.UDim()),
                              VectorXf::Zero(single.UDim())));

  const std::vector<std::vector<QuadraticCostApproximation>> big_quad(
      time::kNumTimeSteps, {quad});

  // Solve open-loop and feedback separately.
  const std::shared_ptr<ConcatenatedDynamicalSystem> dyn(
      new ConcatenatedDynamicalSystem(
          {std::make_shared<SinglePlayerUtilityDynamics>()}));
  LQOpenLoopSolver ol(dyn, time::kNumTimeSteps);
  LQFeedbackSolver fb(dyn, time::kNumTimeSteps);

  const VectorXf x0 = VectorXf::Ones(single.XDim());
  const std::vector<Strategy> ol_strategy = ol.Solve(big_lin, big_quad, x0);
  const std::vector<Strategy> fb_strategy = fb.Solve(big_lin, big_quad, x0);

  // Check that these are the same open-loop strategy, but only at the first
  // time step because the state costs in open-loop are treated as pertaining to
  // the next state. Over a long time horizon, these differences (which only
  // really change the overall cost at the last time step) should become
  // insignificant, at least when considering the first time step.
  CHECK_EQ(fb_strategy.size(), 1);
  CHECK_EQ(ol_strategy.size(), 1);

  const VectorXf u_ol = ol_strategy[0](0, VectorXf::Zero(2), VectorXf::Zero(2));
  const VectorXf u_fb = fb_strategy[0](0, x0, VectorXf::Zero(2));

  constexpr float kMaxRelativeError = 0.01;
  CHECK_LT((u_ol - u_fb).cwiseAbs().maxCoeff(),
           kMaxRelativeError * u_fb.cwiseAbs().maxCoeff());
}

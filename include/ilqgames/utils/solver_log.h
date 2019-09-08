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
// Container to store solver logs.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_UTILS_LOG_H
#define ILQGAMES_UTILS_LOG_H

#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>
#include <ilqgames/utils/uncopyable.h>

#include <math.h>
#include <vector>

namespace ilqgames {

class SolverLog : private Uncopyable {
 public:
  ~SolverLog() {}
  explicit SolverLog(Time time_step) : time_step_(time_step) {}

  // Add a new solver iterate.
  void AddSolverIterate(const OperatingPoint& operating_point,
                        const std::vector<Strategy>& strategies,
                        const std::vector<float>& total_costs,
                        Time cumulative_runtime) {
    operating_points_.push_back(operating_point);
    strategies_.push_back(strategies);
    total_player_costs_.push_back(total_costs);
    cumulative_runtimes_.push_back(cumulative_runtime);
  }

  // Accessors.
  Time TimeStep() const { return time_step_; }
  Time InitialTime() const {
    return (NumIterates() > 0) ? operating_points_[0].t0 : 0.0;
  }
  Time FinalTime() const {
    return (NumIterates() > 0) ? IndexToTime(operating_points_[0].xs.size() - 1)
                               : 0.0;
  }
  PlayerIndex NumPlayers() const { return strategies_[0].size(); }
  size_t NumIterates() const { return operating_points_.size(); }
  size_t NumTimeSteps() const {
    return static_cast<size_t>((FinalTime() - InitialTime()) / time_step_);
  }

  const std::vector<Strategy>& FinalStrategies() const {
    return strategies_.back();
  }
  const OperatingPoint& FinalOperatingPoint() const {
    return operating_points_.back();
  }

  VectorXf InterpolateState(size_t iterate, Time t) const;
  float InterpolateState(size_t iterate, Time t, Dimension dim) const;
  VectorXf InterpolateControl(size_t iterate, Time t, PlayerIndex player) const;
  float InterpolateControl(size_t iterate, Time t, PlayerIndex player,
                           Dimension dim) const;

  std::vector<MatrixXf> Ps(size_t iterate, size_t time_index) const;
  std::vector<VectorXf> alphas(size_t iterate, size_t time_index) const;
  MatrixXf P(size_t iterate, size_t time_index, PlayerIndex player) const;
  VectorXf alpha(size_t iterate, size_t time_index, PlayerIndex player) const;

  VectorXf State(size_t iterate, size_t time_index) const {
    return operating_points_[iterate].xs[time_index];
  }
  float State(size_t iterate, size_t time_index, Dimension dim) const {
    return operating_points_[iterate].xs[time_index](dim);
  }
  VectorXf Control(size_t iterate, size_t time_index,
                   PlayerIndex player) const {
    return operating_points_[iterate].us[time_index][player];
  }
  float Control(size_t iterate, size_t time_index, PlayerIndex player,
                Dimension dim) const {
    return operating_points_[iterate].us[time_index][player](dim);
  }

  std::vector<MatrixXf> Ps(size_t iterate, Time t) const {
    return Ps(iterate, TimeToIndex(t));
  }
  std::vector<VectorXf> alphas(size_t iterate, Time t) const {
    return alphas(iterate, TimeToIndex(t));
  }
  MatrixXf P(size_t iterate, Time t, PlayerIndex player) const {
    return P(iterate, TimeToIndex(t), player);
  }
  VectorXf alpha(size_t iterate, Time t, PlayerIndex player) const {
    return alpha(iterate, TimeToIndex(t), player);
  }

  // Get index corresponding to the time step immediately before the given time.
  size_t TimeToIndex(Time t) const {
    return static_cast<size_t>(
        std::max(constants::kSmallNumber, t - InitialTime()) / time_step_);
  }

  // Get time stamp corresponding to a particular index.
  Time IndexToTime(size_t idx) const {
    return InitialTime() + time_step_ * static_cast<Time>(idx);
  }

  // Save to disk.
  bool Save(const bool only_last_trajectory = false,
            const std::string& experiment_name = DefaultExperimentName()) const;

 private:
  // Convert current time into a default experiment name for unique log saving.
  static std::string DefaultExperimentName();

  // Time discretization.
  const Time time_step_;

  // Operating points, strategies, total costs, and cumulative runtime indexed
  // by solver iterate.
  std::vector<OperatingPoint> operating_points_;
  std::vector<std::vector<Strategy>> strategies_;
  std::vector<std::vector<float>> total_player_costs_;
  std::vector<Time> cumulative_runtimes_;
};  // class SolverLog

}  // namespace ilqgames

#endif

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

#include <ilqgames/utils/make_directory.h>
#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/solver_log.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>
#include <ilqgames/utils/uncopyable.h>

#include <glog/logging.h>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <regex>

namespace ilqgames {

VectorXf SolverLog::InterpolateState(size_t iterate, Time t) const {
  const OperatingPoint& op = operating_points_[iterate];

  // Low and high indices between which to interpolate.
  const size_t lo = TimeToIndex(t);
  const size_t hi = std::min(lo + 1, op.xs.size() - 1);

  // Fraction of the way between lo and hi.
  const float frac = (t - IndexToTime(lo)) / time::kTimeStep;
  return (1.0 - frac) * op.xs[lo] + frac * op.xs[hi];
}

float SolverLog::InterpolateState(size_t iterate, Time t, Dimension dim) const {
  const OperatingPoint& op = operating_points_[iterate];

  // Low and high indices between which to interpolate.
  const size_t lo = TimeToIndex(t);
  const size_t hi = std::min(lo + 1, op.xs.size() - 1);

  // Fraction of the way between lo and hi.
  const float frac = (t - IndexToTime(lo)) / time::kTimeStep;
  return (1.0 - frac) * op.xs[lo](dim) + frac * op.xs[hi](dim);
}

VectorXf SolverLog::InterpolateControl(size_t iterate, Time t,
                                       PlayerIndex player) const {
  const OperatingPoint& op = operating_points_[iterate];

  // Low and high indices between which to interpolate.
  const size_t lo = TimeToIndex(t);
  const size_t hi = std::min(lo + 1, op.xs.size() - 1);

  // Fraction of the way between lo and hi.
  const float frac = (t - IndexToTime(lo)) / time::kTimeStep;
  return (1.0 - frac) * op.us[lo][player] + frac * op.us[hi][player];
}

float SolverLog::InterpolateControl(size_t iterate, Time t, PlayerIndex player,
                                    Dimension dim) const {
  const OperatingPoint& op = operating_points_[iterate];

  // Low and high indices between which to interpolate.
  const size_t lo = TimeToIndex(t);
  const size_t hi = std::min(lo + 1, op.xs.size() - 1);

  // Fraction of the way between lo and hi.
  const float frac = (t - IndexToTime(lo)) / time::kTimeStep;
  return (1.0 - frac) * op.us[lo][player](dim) + frac * op.us[hi][player](dim);
}

bool SolverLog::Save(bool only_last_trajectory,
                     const std::string& experiment_name) const {
  // Making top-level directory
  const std::string dir_name =
      std::string(ILQGAMES_LOG_DIR) + "/" + experiment_name;
  if (!MakeDirectory(dir_name)) return false;
  LOG(INFO) << "Saving to directory: " << dir_name;

  size_t start = 0;
  if (only_last_trajectory) start = operating_points_.size() - 1;

  for (size_t ii = start; ii < operating_points_.size(); ii++) {
    const auto& op = operating_points_[ii];
    const std::string sub_dir_name = dir_name + "/" + std::to_string(ii);
    if (!MakeDirectory(sub_dir_name)) return false;

    // Dump initial time.
    std::ofstream file;
    file.open(sub_dir_name + "/t0.txt");
    file << op.t0 << std::endl;
    file.close();

    // Dump xs.
    file.open(sub_dir_name + "/xs.txt");
    for (const auto& x : op.xs) {
      file << x.transpose() << std::endl;
    }
    file.close();

    // Dump total costs.
    file.open(sub_dir_name + "/costs.txt");
    for (const auto& c : total_player_costs_[ii]) {
      file << c << std::endl;
    }
    file.close();

    // Dump cumulative runtimes.
    file.open(sub_dir_name + "/cumulative_runtimes.txt");
    file << cumulative_runtimes_[ii] << std::endl;
    file.close();

    // Dump us.
    std::vector<std::ofstream> files(NumPlayers());
    for (size_t jj = 0; jj < files.size(); jj++) {
      files[jj].open(sub_dir_name + "/u" + std::to_string(jj) + ".txt");
    }
    for (size_t kk = 0; kk < op.us.size(); kk++) {
      CHECK_EQ(files.size(), op.us[kk].size());
      for (size_t jj = 0; jj < files.size(); jj++) {
        files[jj] << op.us[kk][jj].transpose() << std::endl;
      }
    }
    for (size_t jj = 0; jj < files.size(); jj++) {
      files[jj].close();
    }
  }

  return true;
}

inline std::vector<MatrixXf> SolverLog::Ps(size_t iterate,
                                           size_t time_index) const {
  std::vector<MatrixXf> Ps(strategies_[iterate].size());
  for (PlayerIndex ii = 0; ii < Ps.size(); ii++)
    Ps[ii] = P(iterate, time_index, ii);
  return Ps;
}

inline std::vector<VectorXf> SolverLog::alphas(size_t iterate,
                                               size_t time_index) const {
  std::vector<VectorXf> alphas(strategies_[iterate].size());
  for (PlayerIndex ii = 0; ii < alphas.size(); ii++)
    alphas[ii] = alpha(iterate, time_index, ii);
  return alphas;
}

inline MatrixXf SolverLog::P(size_t iterate, size_t time_index,
                             PlayerIndex player) const {
  return strategies_[iterate][player].Ps[time_index];
}

inline VectorXf SolverLog::alpha(size_t iterate, size_t time_index,
                                 PlayerIndex player) const {
  return strategies_[iterate][player].alphas[time_index];
}

std::string DefaultExperimentName() {
  const auto date =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::string name = std::string(std::ctime(&date));
  std::transform(name.begin(), name.end(), name.begin(),
                 [](char ch) { return (ch == ' ' || ch == ':') ? '_' : ch; });
  return std::regex_replace(name, std::regex("( |\n)+$"), "");
}

bool SaveLogs(const std::vector<SolverLog>& logs, bool only_last_trajectory,
              const std::string& experiment_name) {
  const std::string dir_name =
      std::string(ILQGAMES_LOG_DIR) + "/" + experiment_name;
  if (!MakeDirectory(dir_name)) return false;

  for (size_t ii = 0; ii < logs.size(); ii++) {
    const auto& log = logs[ii];

    if (!log.Save(only_last_trajectory,
                  experiment_name + "/" + std::to_string(ii)))
      return false;
  }

  return true;
}

bool SaveLogs(const std::vector<std::shared_ptr<const SolverLog>>& logs,
              bool only_last_trajectory, const std::string& experiment_name) {
  const std::string dir_name =
      std::string(ILQGAMES_LOG_DIR) + "/" + experiment_name;
  if (!MakeDirectory(dir_name)) return false;

  for (size_t ii = 0; ii < logs.size(); ii++) {
    const auto& log = logs[ii];

    if (!log->Save(only_last_trajectory,
                   experiment_name + "/" + std::to_string(ii)))
      return false;
  }

  return true;
}

}  // namespace ilqgames

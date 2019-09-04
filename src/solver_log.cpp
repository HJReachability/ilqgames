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

// Convert current time into a default experiment name for unique log saving.
std::string SolverLog::DefaultExperimentName() {
  const auto date =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::string name = std::string(std::ctime(&date));
  std::transform(name.begin(), name.end(), name.begin(),
                 [](char ch) { return (ch == ' ' || ch == ':') ? '_' : ch; });
  return std::regex_replace(name, std::regex("( |\n)+$"), "");
}

VectorXf SolverLog::InterpolateState(size_t iterate, Time t) const {
  const OperatingPoint& op = operating_points_[iterate];

  // Low and high indices between which to interpolate.
  const size_t lo = TimeToIndex(t);
  const size_t hi = std::min(lo + 1, op.xs.size() - 1);

  // Fraction of the way between lo and hi.
  const float frac = (t - IndexToTime(lo)) / time_step_;
  return (1.0 - frac) * op.xs[lo] + frac * op.xs[hi];
}

float SolverLog::InterpolateState(size_t iterate, Time t, Dimension dim) const {
  const OperatingPoint& op = operating_points_[iterate];

  // Low and high indices between which to interpolate.
  const size_t lo = TimeToIndex(t);
  const size_t hi = std::min(lo + 1, op.xs.size() - 1);

  // Fraction of the way between lo and hi.
  const float frac = (t - IndexToTime(lo)) / time_step_;
  return (1.0 - frac) * op.xs[lo](dim) + frac * op.xs[hi](dim);
}

VectorXf SolverLog::InterpolateControl(size_t iterate, Time t,
                                       PlayerIndex player) const {
  const OperatingPoint& op = operating_points_[iterate];

  // Low and high indices between which to interpolate.
  const size_t lo = TimeToIndex(t);
  const size_t hi = std::min(lo + 1, op.xs.size() - 1);

  // Fraction of the way between lo and hi.
  const float frac = (t - IndexToTime(lo)) / time_step_;
  return (1.0 - frac) * op.us[lo][player] + frac * op.us[hi][player];
}

float SolverLog::InterpolateControl(size_t iterate, Time t, PlayerIndex player,
                                    Dimension dim) const {
  const OperatingPoint& op = operating_points_[iterate];

  // Low and high indices between which to interpolate.
  const size_t lo = TimeToIndex(t);
  const size_t hi = std::min(lo + 1, op.xs.size() - 1);

  // Fraction of the way between lo and hi.
  const float frac = (t - IndexToTime(lo)) / time_step_;
  return (1.0 - frac) * op.us[lo][player](dim) + frac * op.us[hi][player](dim);
}

bool SolverLog::Save(const std::string& experiment_name) const {
  auto make_directory = [](const std::string& directory_name) {
    if (mkdir(directory_name.c_str(), 0777) == -1) {
      LOG(ERROR) << "Could not create directory " << directory_name
                 << ". Error msg: " << std::strerror(errno);
      return false;
    }
    return true;
  };  // make_directory

  // Making top-level directory

  const std::string dir_name =
      std::string(ILQGAMES_LOG_DIR) + "/" + experiment_name;
  if (!make_directory(dir_name)) return false;
  LOG(INFO) << "Saving to directory: " << dir_name;

  for (size_t ii = 0; ii < operating_points_.size(); ii++) {
    const auto& op = operating_points_[ii];
    const std::string sub_dir_name = dir_name + "/" + std::to_string(ii);
    if (!make_directory(sub_dir_name)) return false;

    // Dump xs.
    std::ofstream file;
    file.open(sub_dir_name + "/xs.txt");
    for (const auto& x : op.xs) {
      file << x.transpose() << std::endl;
    }
    file.close();

    // Dump cumulative runtimes.
    file.open(sub_dir_name + "/runtimes.txt");
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

}  // namespace ilqgames

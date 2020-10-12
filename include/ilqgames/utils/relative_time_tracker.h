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
// Base class for all named objects which depend upon the initial time and need
// to convert between absolute times and time steps. Examples of derived classes
// are Cost.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_UTILS_RELATIVE_TIME_TRACKER_H
#define ILQGAMES_UTILS_RELATIVE_TIME_TRACKER_H

#include <ilqgames/utils/types.h>

#include <glog/logging.h>

namespace ilqgames {

class RelativeTimeTracker {
 public:
  virtual ~RelativeTimeTracker() {}

  // Access and reset initial time.
  static void ResetInitialTime(Time t0) { initial_time_ = t0; };
  static Time InitialTime() { return initial_time_; }

  // Convert between time step and initial time.
  static Time RelativeTime(size_t kk) {
    return static_cast<Time>(kk) * time::kTimeStep;
  }
  static Time AbsoluteTime(size_t kk) {
    return initial_time_ + static_cast<Time>(kk) * time::kTimeStep;
  }
  static size_t TimeIndex(Time t) {
    CHECK_GE(t, initial_time_);
    return static_cast<size_t>((t - initial_time_) / time::kTimeStep);
  }

  // Access the name of this object.
  const std::string& Name() const { return name_; }

 protected:
  RelativeTimeTracker(const std::string& name) : name_(name) {}

  // Name associated to every cost.
  const std::string name_;

  // Initial time.
  static Time initial_time_;
};  //\class Cost

}  // namespace ilqgames

#endif

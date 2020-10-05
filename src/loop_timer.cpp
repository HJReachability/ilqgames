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
// Keeps track of elapsed time (e.g., during loops) and provides an upper bound
// on the runtime of the next loop. To reduce memory consumption and adapt to
// changing processor activity, computes statistics based on a moving window of
// specified length.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/utils/loop_timer.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <chrono>
#include <list>

namespace ilqgames {

void LoopTimer::Tic() { start_ = std::chrono::high_resolution_clock::now(); }

Time LoopTimer::Toc() {
  // Elapsed time in seconds.
  const Time elapsed = (std::chrono::duration<Time>(
                            std::chrono::high_resolution_clock::now() - start_))
                           .count();

  // Add to queue and pop if queue is too long.
  loop_times_.push_back(elapsed);
  total_time_ += elapsed;

  if (loop_times_.size() > max_samples_) {
    total_time_ -= loop_times_.front();
    loop_times_.pop_front();
  }

  return elapsed;
}

Time LoopTimer::RuntimeUpperBound(float num_stddevs, Time initial_guess) const {
  // Handle not enough data.
  if (loop_times_.size() < 2) return initial_guess;

  // Compute mean and variance.
  const Time mean = total_time_ / static_cast<Time>(loop_times_.size());
  Time variance = 0.0;
  for (const Time entry : loop_times_) {
    const Time diff = entry - mean;
    variance += diff * diff;
  }

  // Unbiased estimator of variance should divide by N - 1, not N.
  variance /= static_cast<Time>(loop_times_.size() - 1);

  // Compute upper bound.
  return mean + num_stddevs * std::sqrt(variance);
}

}  // namespace ilqgames

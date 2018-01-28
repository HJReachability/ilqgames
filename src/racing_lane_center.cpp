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
// Compute a lane center entering and leaving a roundabout.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <vector>

namespace ilqgames {

PointList2 RacingLaneCenter(float X_init, float Y_init,
                            float side_len, float turn_rad,
                            float kNumPointsInArc) {

float root_angle = 180/kNumPointsInArc;
// Radius of roundabout and lane half width.
//setup polyline points from track parameters
//first straight-away, directly in front of initial car locations


  //initialize point list with initial point and end point of first straight-away
  PointList2 points = {Point2(X_init,Y_init), Point2(X_init,side_len)};
 
  
  for (int ii=1; ii<=kNumPointsInArc; ii++){
  //for top turn
    if(ii*root_angle<90){
      points.push_back(Point2(X_init-turn_rad*(1-cos(ii*root_angle)),Y_init+side_len+turn_rad*sin(ii*root_angle)));
    }
    else {
      points.push_back(Point2(X_init-turn_rad+turn_rad*cos(ii*root_angle),Y_init+side_len+turn_rad*sin(ii*root_angle)));    
    }  
  }
  //insert beginning and end of straight-away 2 
  points.push_back(Point2(X_init-2*turn_rad,Y_init+side_len));
  points.push_back(Point2(X_init-2*turn_rad,Y_init));

  for (int ii=1; ii<=kNumPointsInArc; ii++){
  //for top turn
    if(ii*root_angle<90){
      //bottom turn
      points.push_back(Point2(X_init-2*turn_rad+turn_rad*(1-cos(ii*root_angle)),Y_init-turn_rad*sin(ii*root_angle)));
    }
    else {
      //bottom turn
      points.push_back(Point2(X_init-2*turn_rad+turn_rad*cos(ii*root_angle),side_len+Y_init+turn_rad*sin(ii*root_angle)));
    }   
  }
  printf("test");
  return points;
}

}  // namespace ilqgames

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Simulation of distraction-aware planning for a 2D point mass.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Parameters.
u_max = 4.0; % Maximum acceleration/deceleration (m/s/s).
max_velocity_delta = 2.0; % Maximum velocity change (m/s).
min_dist = 1.0; % Closest vehicles should ever get (m).

%% Initial states (p = position, v = velocity).
ph0 = [20.0, -4.0];
vh0 = [-10.0, 0.0];

pr0 = [0.0, 0.0];
vr0 = [10.0, 0.0];

%% Hardcode human trajectory and nominal robot trajectory as straight lines
% at constant speed.
% TODO! For now assuming just moving at initial velocity.

%% Loop over time and figure out maximal T (time to potential unavoidable
% collision), and apply corresponding control.
dt = 0.1;
max_time = 10.0;
times = 0.0:dt:max_time;

ph = ph0; vh = vh0;
pr = pr0; vr = vr0;

for t = times
  % Compute time till states get close in x, y.
  [max_tx, ty] = compute_max_tx_ty(ph, vh, pr, vr);
  T = max(max_tx, ty);

  % Determine robot's ideal speed. This will either be the nominal
  % speed or it will mean slowing down as much as possible.
  % TODO! Make this max velocity change relative to last speed not
  % initial speed. That will help keep things dynamically feasible.
  vr_desired = vr0
  if T = max_tx
    vr_desired(1) = vr_desired(1) - max_velocity_delta;
  end

  % Update human and robot states.
  ph = ph + vh * dt;
  pr = pr + vr_desired * dt;

  % Plot/animate this over time.
  % TODO!
end

%% Utility function to compute max time to "collision" in x and y.
function [tx, ty] = compute_max_tx_ty(ph, vh, pr, vr)
  p_rel = ph - pr;
  v_rel = vh - vr;

  % Adjust v_rel in x-direction so as to maximize time to collision in x.
  if v_rel < 0
    v_rel(1) = v_rel(1) + max_velocity_delta;
  end

  % Time till states get too close in x is either 0 (if already that close)
  % or something more complicated.
  tx = 0.0;
  if abs(p_rel(1)) > min_dist
    tx = (v_rel(1) / u_max) * sign(p_rel(1)) + ...
         sqrt(v_rel(1)*v_rel(1) + u_max(abs(p_rel(1)) - min_dist)) / u_max;
  end

  % Same for time to get close in y.
  ty = 0.0;
  if abs(p_rel(2)) > min_dist
    ty = (v_rel(2) / u_max) * sign(p_rel(2)) + ...
         sqrt(v_rel(2)*v_rel(2) + u_max(abs(p_rel(2)) - min_dist)) / u_max;
  end
end

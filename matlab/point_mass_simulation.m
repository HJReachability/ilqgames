%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Simulation of distraction-aware planning for a 2D point mass.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Parameters.
u_max = 4.0; % Maximum acceleration/deceleration (m/s/s).
max_velocity_delta = 5.0; % Maximum velocity change (m/s).
min_dist = 1.0; % Closest vehicles should ever get (m).
gif_output = 'out.gif';

%% Initial states (p = position, v = velocity).
ph0 = [60.0, -4.0];
vh0 = [-10.0, 0.0];

pr0 = [0.0, 0.0];
vr0 = [10.0, 0.0];

%% Time definition
dt = 0.1;
max_time = 5.0;
times = 0.0:dt:max_time;


%% Hardcode human trajectory and nominal robot trajectory as straight lines
% at constant speed.
% TODO! For now assuming just moving at initial velocity.
u_h = zeros(2, length(times));
half_T = floor(length(times) / 2) + 3;
k = 5;
u_h(2,half_T-k:half_T) = 4.0;
u_h(2,half_T+1:half_T+k+2) = -4.0;


%% Loop over time and figure out maximal T (time to potential unavoidable
% collision), and apply corresponding control.

ph = ph0; vh = vh0;
pr = pr0; vr = vr0;

fgh = figure();
axh = axes('Parent',fgh); 
scatter(ph(2), ph(1), 'ro');
scatter(pr(2), pr(1), 'bo');
axis([-8 4 0 100]);

for i=1:length(times)
  %t = times(i);
  % Compute time till states get close in x, y.
  p_rel = ph - pr; 
  vr_desired = vr0;

  if p_rel(1) >= 0
      [max_tx, ty] = compute_max_tx_ty(ph, vh, pr, vr, max_velocity_delta, ...
          min_dist, u_max);
      T = max(max_tx, ty);

      % Determine robot's ideal speed. This will either be the nominal
      % speed or it will mean slowing down as much as possible.
      % TODO! Make this max velocity change relative to last speed not
      % initial speed. That will help keep things dynamically feasible.
      if max_tx > ty
        vr_desired(1) = vr_desired(1) - max_velocity_delta;
      end
  end

  vh
  vr_desired
  %u_h(:,i)'
  vh = vh + u_h(:,i)' * dt;
  
  % Update human and robot states.
  ph = ph + vh * dt;
  pr = pr + vr_desired * dt;
  
  %vr_desired
  %vh0
  
  %ph
  % Plot/animate this over time.
  % TODO!
  pause(dt);
  hold all; 
  scatter(ph(2), ph(1), 'ro');
  scatter(pr(2), pr(1), 'bo');
  
  if ~isempty(gif_output)
    frame = getframe(fgh);
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256); 
    if i == 1
      imwrite(imind, cm, gif_output, 'gif', 'DelayTime', dt, 'LoopCount', inf);
    else
      imwrite(imind, cm, gif_output, 'gif', 'WriteMode', 'append', 'DelayTime', dt);
    end
  end

  %title(['t = '  int2str(times(i))]);
  title(['t = '  int2str(times(i)) ', ' sprintf('v_r = %f', vr_desired(1))]);  
end

%% Utility function to compute max time to "collision" in x and y.
function [tx, ty] = compute_max_tx_ty(ph, vh, pr, vr, max_velocity_delta, ...
    min_dist, u_max)
  p_rel = ph - pr;
  v_rel = vh - vr;

  % Adjust v_rel in x-direction so as to maximize time to collision in x.
  v_rel(1) = v_rel(1) + max_velocity_delta;

  % Time till states get too close in x is either 0 (if already that close)
  % or something more complicated.
  tx = 0.0;
  if abs(p_rel(1)) > min_dist
    tx = (v_rel(1) / u_max) * sign(p_rel(1)) + ...
         sqrt(v_rel(1)*v_rel(1) + u_max * (abs(p_rel(1)) - min_dist)) / u_max;
  end

  % Same for time to get close in y.
  ty = 0.0;
  if abs(p_rel(2)) > min_dist
    ty = (v_rel(2) / u_max) * sign(p_rel(2)) + ...
         sqrt(v_rel(2)*v_rel(2) + u_max * (abs(p_rel(2)) - min_dist)) / u_max;
  end
  
  %tx
  %ty
end

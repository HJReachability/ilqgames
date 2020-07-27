collision_avoidance_example();

function collision_avoidance_example()
close all;

%% Compute ILQ trajectory for same problem with different parameters and overlay plots.
regularization_vals = linspace(0.75, 1.5, 5);

figure;
set(gca, 'FontSize', 24');
title(sprintf('Sensitivity to Regularization', nominal_scale), ...
      'Interpreter', 'latex');
xlabel('$p_x$ (m)', 'Interpreter', 'latex');
ylabel('$p_y$ (m)', 'Interpreter', 'latex');
xlim([-5.5, 5.5]);
ylim([-5.5, 5.5]);

hold on;
ii = 1;
for regularization = regularization_vals
  [ilq_traj, values] = run_ilqgames("three_player_collision_avoidance_reachability_example", "", ...
                                    regularization, x0_flag);
  if max(abs(values(1) - values(2)), abs(values(1) - values(3))) > 0.01
    disp(char("Error! Values do not match: " + values(1) + ", " + values(2) + ", " + values(3)));
  end

  pe(ii) = plot(ilq_traj(:, 1), ilq_traj(:, 2), 'x-', ...
                'color', colormap(regularization, regularization_vals, false), ...
                'DisplayName', sprintf('$\\epsilon = %1.2f$', regularization));
  plot(ilq_traj(:, 6), ilq_traj(:, 7), 'x:', 'color', ...
       colormap(regularization, regularization_vals, false));
  plot(ilq_traj(:, 11), ilq_traj(:, 12), 'x--', 'color', ...
       colormap(regularization, regularization_vals, false));

  ii = ii + 1;
end

hold off;
l2 = legend(pe, 'Location', 'NorthEast');

set(l1, 'Interpreter', 'latex');
set(l2, 'Interpreter', 'latex');
end

%% Simple red-blue colormap.
function color = colormap(val, opts, reverse)
  r = (val - opts(1)) / (opts(end) - opts(1));
  color = [r, 0.25, 1.0 - r];

  if reverse
    color = [1.0 - r, 0.25, r];
  end
end

%% Compute ILQ trajectory for given example.
function [traj, values] = run_ilqgames(exec, extra_suffix, regularization, x0_flag)
  experiment_name = exec + "_" + control_penalty + x0_flag;
  experiment_arg = " --experiment_name='" + experiment_name + extra_suffix + "'";
  save_flag = "--save" + extra_suffix;

  if ~experiment_already_run(char(experiment_name + extra_suffix))
    %% Stitch together the command for the executable.
    instruction = "../bin/" + exec + " --noviz " + save_flag + ...
                  " --last_traj" + experiment_arg + ...
                  " --convergence_tolerance=0.01 --trust_region_size=0.1" + ...
                  " --control_penalty=" + regularization + ...
                  " --initial_alpha_scaling=0.01 " + x0_flag;
    system(char(instruction));
  end

  log_folder = "../logs/";
  cd(char(log_folder + experiment_name + extra_suffix));
  dirs = dir;
  last_iterate = "blah";
  for ii = 1:size(dirs)
      if dirs(ii).name(1) ~= '.'
          last_iterate = dirs(ii).name;
          break;
      end
  end
  cd('../../matlab');

  traj = load(log_folder + experiment_name + extra_suffix + "/" + ...
              last_iterate + "/xs.txt");
  values = load(log_folder + experiment_name + extra_suffix + "/" + ...
                last_iterate + "/costs.txt");
  for ii = 1:length(values)
    values(ii) = log(values(ii)) / scale;
  end
end

function exists = experiment_already_run(folder_name)
  %% Get experiment folder names
  cd('../logs');
  dirs = dir;
  exists = false;
  for ii = 1:length(dirs)
    if strcmp(folder_name, dirs(ii).name)
      exists = true;
      break;
    end
  end
  cd('../matlab');
end

test_region_size_vs_alpha_scaling('three_player_flat_intersection');
% plot_convergence('Thu_Sep__5_11_07_29_2019');
% hold on
% plot_convergence('Thu_Sep__5_11_09_55_2019');

% Norm of the difference between trajectories.
function y = trajectory_d(traj1,traj2)
d = traj1 - traj2;
y = norm(d); 
end

% Plot a trajectory (assuming it is a concatenation of a
% 6D + 6D + 4D system).
function plot_trajectory(trajectory)
    xy_indeces = [1,2,7,8,13,14];
    x = trajectory(:,1);
    y = trajectory(:,2);
    plot(x,y,'r');
    hold on
    x = trajectory(:,7);
    y = trajectory(:,8);
    plot(x,y,'b');
    hold on
    x = trajectory(:,13);
    y = trajectory(:,14);
    plot(x,y,'g');
end

% Plot trajectory for a certain experiment
function plot_experiment_traj(experiment_folder)
    cd(experiment_folder);
    dirs = struct2cell(dir);
    cd(dirs{1,3});
    traj = load("xs.txt");
    plot_trajectory(traj);
    cd('..');
    cd('..');
end

% This script plots ||traj_final - traj_iter||
function plot_convergence(experiment_name)
    cd(experiment_name);
    dirs = dir;
    % Record final trajectory for comparison
    num_iters = length(dirs) - 3;
    cd(num2str(num_iters));
    traj_f = load("xs.txt");
    cd('..');
    convergence_value = zeros(1,num_iters);
    for i=1:num_iters
       cd(num2str(i));
       traj_i = load("xs.txt");
       convergence_value(i) = trajectory_d(traj_i,traj_f);
       cd('..')
    end
    plot(convergence_value);
    cd('..');
end

function test_region_size_vs_alpha_scaling(exec1)
    folder = "../bin/";
    flag1 = " -trust_region_size=";
    flag2 = " -initial_alpha_scaling=";
    experiment_arg = "-experiment_name=";
    n = 3;
    m = 3;
    range_for_tregion = linspace(10,100,n);
    range_for_alphasc = linspace(0.1,1.0,m);
    j = 1;
    for tr_size=range_for_tregion
       for alpha_scaling=range_for_alphasc
           string_alpha_sc = string(num2str(alpha_scaling));
           string_tr_size  = string(num2str(tr_size));
           tmp =exec1+"_trsize_" + num2str(tr_size) + ...
                                               "_alphasc_" + num2str(alpha_scaling);
           exp_folder_name = "'"+tmp + "'";
           experiment_name = experiment_arg + exp_folder_name;
           instruction = folder + exec1 + flag1 + string_tr_size + ...
                         flag2 + string_alpha_sc + " -viz=false -save=true -last_traj=true " + ...
                         experiment_name;
           system(char(instruction));
           subplot(n,m,j);
           plot_experiment_traj(char(tmp));
           j = j+1;
       end
    end
end
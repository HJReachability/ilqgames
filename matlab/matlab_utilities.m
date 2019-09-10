% 1 or 0
rng(1)

% Test region size vs. alpha scaling
%test_region_size_vs_alpha_scaling('three_player_intersection',5,6);
sensitivity_to_params('ideal_intersection_trajectory',...
    {'three_player_flat_intersection','three_player_intersection'},310);

% Plotting convergence of two different experiments
% plot_convergence('Sat_Sep__7_15_37_49_2019');
% hold on
% plot_convergence('Sat_Sep__7_15_38_38_2019');

% Norm of the difference between trajectories.
function y = trajectory_d(traj1,traj2)
xy_indeces = [1,2,7,8,13,14];
d = traj1(:,xy_indeces) - traj2(:,xy_indeces);
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
    axis equal
end

% Plot trajectory for a certain experiment
function runtime = plot_experiment_traj(experiment_folder)
    cd(experiment_folder);
    dirs = struct2cell(dir);
    cd(dirs{1,3});
    traj = load("xs.txt");
    runtime = load("runtimes.txt");
    plot_trajectory(traj);
    cd('..');
    cd('..');
end

% This script plots ||traj_final - traj_iter||
function plot_convergence(experiment_name)
    cd('../logs')
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
    cd('../matlab')
end

% This script plots ||traj_ideal - traj_iter||
function [y,runtime] = convergence_val(experiment_folder,ideal_trajectory)
    cd('../logs')
    cd(experiment_folder);
    dirs = struct2cell(dir);
    cd(dirs{1,3});
    traj = load("xs.txt");
    runtime = load("runtimes.txt");
    y = trajectory_d(traj,ideal_trajectory);
    cd('..');
    cd('../../matlab')
end

function y=check_if_folder_exists(folder_name)
    % Get experiment folder names
    cd('../logs')
    dirs = dir;
    y = false;
    for i=1:length(dirs)
        if strcmp(folder_name,dirs(i).name)
            y=true;
            break;
        end
    end
    cd('../matlab');
end

function test_region_size_vs_alpha_scaling(exec1,n,m)
    prefix = "../logs";
    folder = "../bin/";
    flag1 = " -trust_region_size=";
    flag2 = " -initial_alpha_scaling=";
    experiment_arg = "-experiment_name=";
    
    range_for_tregion = [1.0,2.0,3.0,4.0,5.0];%linspace(0.0,20.0,n);
    range_for_alphasc = linspace(0.1,0.75,m);
    j = 1;
    for tr_size=range_for_tregion
       for alpha_scaling=range_for_alphasc
           
           % Convert parameters loop parameters to strings for names
           string_alpha_sc = string(num2str(alpha_scaling));
           string_tr_size  = string(num2str(tr_size));
           tmp =exec1+"_trsize_" + num2str(tr_size) + ...
                                               "_alphasc_" + num2str(alpha_scaling);
           exp_folder_name = "'"+tmp + "'";
           experiment_name = experiment_arg + exp_folder_name;
           
           exists = check_if_folder_exists(char(tmp));
           if ~exists
               % Stitch together the command for the executable.
               instruction = folder + exec1 + flag1 + string_tr_size + ...
                             flag2 + string_alpha_sc + " -viz=false -save=true -last_traj=true " + ...
                             experiment_name;
               system(char(instruction));
           end
           subplot(n,m,j);
           runtime = plot_experiment_traj(char(prefix+"/"+tmp));
           title("as=" + string_alpha_sc + ",tr=" + string_tr_size + "(" +num2str(runtime)+ ")");
           j = j+1;
       end
    end
    cd('../matlab')
end

% Sensitivity to parameter choices
function sensitivity_to_params(ideal_traj,execs,num_samples)
    vals = zeros([length(execs), num_samples]);
    alphas = zeros([length(execs), num_samples]);
    sizes = zeros([length(execs), num_samples]);
    times = zeros([length(execs), num_samples]);
    % Load ideal trajectory
    cd('../logs');
    cd(ideal_traj);
    dirs = struct2cell(dir);
    cd(dirs{1,3});
    ideal = load("xs.txt");
    cd('..');
    cd('../../matlab')

    prefix = "../logs";
    folder = "../bin/";
    flag1 = " -trust_region_size=";
    flag2 = " -initial_alpha_scaling=";
    experiment_arg = "-experiment_name=";
    
    range_for_tregion = [1.0,10.0];%linspace(0.0,20.0,n);
    range_for_alphasc = [0.1,0.75];
    
    for i=1:num_samples
       tr_size = random('unif',range_for_tregion(1),range_for_tregion(2));
       alpha_scaling = random('unif',range_for_alphasc(1),range_for_alphasc(2));
           
       % Convert parameters loop parameters to strings for names
       string_alpha_sc = string(num2str(alpha_scaling));
       string_tr_size  = string(num2str(tr_size));
       for j=1:length(execs)
           tmp =execs{j}+"_trsize_" + num2str(tr_size) + ...
                                               "_alphasc_" + num2str(alpha_scaling);
           exp_folder_name = "'" + tmp + "'";
           experiment_name = experiment_arg + exp_folder_name;

           exists = check_if_folder_exists(char(tmp));
           if ~exists
               % Stitch together the command for the executable.
               instruction = folder + execs{j} + flag1 + string_tr_size + ...
                             flag2 + string_alpha_sc + " -viz=false -save=true -last_traj=true " + ...
                             experiment_name;
               system(char(instruction));
           end
           [conv_val,runtime] = convergence_val(char(prefix+"/"+tmp),ideal);
           vals(j,i) = conv_val;
           alphas(j,i) = alpha_scaling; 
           sizes(j,i) = tr_size;
           times(j,i) = runtime;
       end
       %title("as=" + string_alpha_sc + ",tr=" + string_tr_size + "(" +num2str(runtime)+ ")");
    end
    
    threshold = 100;
    vals_min = min(min(vals));
    vals_max = max(max(vals));
    for i=1:num_samples
        for j=1:length(execs)
           if isnan(vals(j,i))
                vals(j,i) = vals_max;
                fprintf("crap" + num2str(i) + num2str(j) + "\n")
           end
        end
    end
    figure(1)
    hist(vals(1,:))
    title("Flat algorithm.")
    xlabel("Similarity to desired (x,y) traj (closer to 0 is better)")
    ylabel("Number of trajectories  (random samples from (alpha,tr))")
    figure(2) 
    hist(vals(2,:))
    title("Non-Flat algorithm.")
    xlabel("Similarity to desired (x,y) traj (closer to 0 is better)")
    ylabel("Number of trajectories  (random samples from (alpha,tr))")
    
    true_times = cell(1,length(execs));
    for j=1:length(execs)
        figure(2+j)
        for i=1:num_samples
           if vals(j,i) > threshold
               vals(j,i) = 1;
           elseif vals(j,i) < threshold
               vals(j,i) = 0;
               true_times{j} = [true_times{j},times(j,i)];
           end
           % red = [1 0 0]
           % blue = [0 0 1]
           plot(alphas(j,i),sizes(j,i),'*','color',[vals(j,i) 0 1-vals(j,i)]);
           hold on 
        end
        if j==1
            title("Flat Algorithm " + "("+mean(true_times{j})+")");
        else
            title("Non-Flat Algorithm " + "("+mean(true_times{j})+")");
        end
        xlabel("initial alpha scaling");
        ylabel("trust region size");
    end
    
    cd('../matlab')
end
% 1 or 0
rng(0)

% Test region size vs. alpha scaling
% test_region_size_vs_alpha_scaling('three_player_intersection',5,6);
% sensitivity_to_params('ideal_intersection_trajectory',...
%    {'three_player_flat_intersection','three_player_intersection'},324,true,0.1);

sensitivity_to_params('highway_merging');

% Plotting convergence of two different experiments
% plot_convergence('Sat_Sep__7_15_37_49_2019');
% hold on
% plot_convergence('Sat_Sep__7_15_38_38_2019');

% Vecnorm
function y=vecnorm(A)
   A = A.^2;
   norm2 = sum(A,2);
   y = sqrt(norm2);
end

% Norm of the difference between trajectories.
function y = trajectory_d(traj1,traj2)
xy_indeces = [1,2;7,8;13,14];
biggest = 0; 
for i=1:length(xy_indeces)
   for j=1:length(traj2)
      tmp = min(vecnorm(traj1(:,xy_indeces(i,:)) - traj2(j,xy_indeces(i,:))));
      if biggest < tmp
         biggest = tmp; 
      end
   end
end
% d = traj1(:,xy_indeces) - traj2(:,xy_indeces);
% d = max(abs(d),[],2);
% y = norm(d,Inf);
y=biggest;
end

% Plot a trajectory (assuming it is a concatenation of a
% 6D + 6D + 4D system).
function plot_trajectory(trajectory)
    % Right now, it's assuming there are 3 players.
    xy_indices = [1,2,7,8,13,14,19,20,25,26,31,32];
    traj_length = size(trajectory, 1);
    num_dots = 5;
    x_min = -15;
    x_max = 15;
    y_min = -35;
    y_max = 75;
    
    % Player 1's trajectory:
    x = trajectory(:,1);
    y = trajectory(:,2);
    plot(x,y,'b', 'MarkerSize', 2); 
    hold on
    plot(trajectory(1,1), trajectory(1,2), 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
    for i = 1:num_dots
        plot(trajectory(round(traj_length*i/num_dots),1), trajectory(round(traj_length*i/num_dots),2), ...
        'bo', 'MarkerSize', 5, 'MarkerFaceColor', 'b');
    end
    xlim([x_min x_max]);
    ylim([y_min y_max]);
    xlabel('x (m)');
    ylabel('y (m)');
    
    hold on
    
    % Player 2's trajectory:
    x = trajectory(:,7);
    y = trajectory(:,8);
    plot(x,y,'b', 'MarkerSize', 2);
    hold on
    plot(trajectory(1,7), trajectory(1,8), 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
    for i = 1:num_dots
        plot(trajectory(round(traj_length*i/num_dots),7), ...
            trajectory(round(traj_length*i/num_dots),8), ...
            'bo', 'MarkerSize', 5, 'MarkerFaceColor', 'b');
    end
    xlim([x_min x_max]);
    ylim([y_min y_max]);
    xlabel('x (m)');
    ylabel('y (m)');
    
    hold on
    
    % Player 3's trajectory:
    x = trajectory(:,13);
    y = trajectory(:,14);
    plot(x,y,'b', 'MarkerSize', 2); 
    hold on
    plot(trajectory(1,13), trajectory(1,14), 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
    for i = 1:num_dots
        plot(trajectory(round(traj_length*i/num_dots),13), ...
            trajectory(round(traj_length*i/num_dots),14), ...
            'bo', 'MarkerSize', 5, 'MarkerFaceColor', 'b');
    end
    xlim([x_min x_max]);
    ylim([y_min y_max]);
    xlabel('x (m)');
    ylabel('y (m)');
    
    hold on
    
    % Player 4's trajectory:
    x = trajectory(:,19);
    y = trajectory(:,20);
    plot(x,y,'r', 'MarkerSize', 2);
    hold on
    plot(trajectory(1,19), trajectory(1,20), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    for i = 1:num_dots
        plot(trajectory(round(traj_length*i/num_dots),19), ...
            trajectory(round(traj_length*i/num_dots),20), ...
            'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'r');
    end
    xlim([x_min x_max]);
    ylim([y_min y_max]);
    xlabel('x (m)');
    ylabel('y (m)');
    
    hold on
    
    % Player 5's trajectory:
    x = trajectory(:,25);
    y = trajectory(:,26);
    plot(x,y,'b', 'MarkerSize', 2); 
    hold on
    plot(trajectory(1,25), trajectory(1,26), 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
    for i = 1:num_dots
        plot(trajectory(round(traj_length*i/num_dots),25), ...
            trajectory(round(traj_length*i/num_dots),26), ...
            'bo', 'MarkerSize', 5, 'MarkerFaceColor', 'b');
    end
    xlim([x_min x_max]);
    ylim([y_min y_max]);
    xlabel('x (m)');
    ylabel('y (m)');
    
    hold on
    
    % Player 6's trajectory:
    x = trajectory(:,31);
    y = trajectory(:,32);
    plot(x,y,'b', 'MarkerSize', 2);
    hold on
    plot(trajectory(1,31), trajectory(1,32), 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
    for i = 1:num_dots
        plot(trajectory(round(traj_length*i/num_dots),31), ...
            trajectory(round(traj_length*i/num_dots),32), ...
            'bo', 'MarkerSize', 5, 'MarkerFaceColor', 'b');
    end
    xlim([x_min x_max]);
    ylim([y_min y_max]);
    xlabel('x (m)');
    ylabel('y (m)');
    
    hold on
    
%     axis equal
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
%     if(y < 10.0 && y > 5.0)
%        hold off
%        plot_trajectory(traj);
%        title(num2str(y));
%        pause(1.0);
%     end
    cd('..');
    cd('../../matlab')
end

function y=check_if_folder_exists(folder_name)
    % Get experiment folder names
    cd('/Users/chihyuan_chiu/Desktop/Research___Shankar/Robotics___Research/ilqgames/logs');
    dirs = dir;
    y = false;
    for i=1:length(dirs)
        if strcmp(folder_name,dirs(i).name)
            y=true;
            break;
        end
    end
%     pwd
    cd('/Users/chihyuan_chiu/Desktop/Research___Shankar/Robotics___Research/ilqgames/matlab');
end

function test_region_size_vs_alpha_scaling(exec)
    prefix = "../logs";
    folder = "../bin/";
    flag_adversarial_time = " -adversarial_time=";
    experiment_arg = "-experiment_name=";
    
    num_figs = 3;
    range_for_adversarial_time = linspace(0, 2, num_figs);
    j = 1;
    for adversarial_time=range_for_adversarial_time
        % Convert parameters loop parameters to strings for names
        string_adversarial_time  = string(num2str(adversarial_time));
        tmp = exec+"_adversarial_time_" + num2str(adversarial_time);
        exp_folder_name = "'"+tmp + "'";
        experiment_name = experiment_arg + exp_folder_name;
           
        exists = check_if_folder_exists(char(tmp));
        
%         if exists
%             system('cd /Users/chihyuan_chiu/Desktop/Research___Shankar/Robotics___Research/ilqgames/logs');
%             system('rm -rf *');
%             system('cd /Users/chihyuan_chiu/Desktop/Research___Shankar/Robotics___Research/ilqgames/matlab');
% %            system('/Users/chihyuan_chiu/Desktop/Research___Shankar/Robotics___Research/ilqgames/logs rm -rf *');
%         end
        
        if ~exists
            % Stitch together the command for the executable.
            instruction = folder + exec + flag_adversarial_time + string_adversarial_time ...
                          + " -initial_alpha_scaling=1.0 -trust_region_size=1.0 " ...
                          + " -viz=false -save=true -last_traj=true " + experiment_name;
%             instruction = folder + exec + flag_adversarial_time + string_adversarial_time ...
%                           + " -viz=false -save=true -last_traj=true " + ...
%                           experiment_name;
            system(char(instruction));
        end

%         instruction = folder + exec + flag_adversarial_time + string_adversarial_time ...
%                           + " -trust_region_size=10 -viz=false -save=true -last_traj=true " + ...
%                           experiment_name;
%         system(char(instruction));
        
        subplot(1,num_figs,j);

        runtime = plot_experiment_traj(char(prefix+"/"+tmp));
        title("Adversarial time = " + string_adversarial_time + " s");
%         title("adv_time=" + string_adversarial_time + "(" +num2str(runtime)+ ")");
        j = j+1;
    end
    cd('../matlab')
end

% Sensitivity to parameter choices
function sensitivity_to_params(experiment_name)
    
    figure;
    
    % Call test_region_size_vs_alpha_scaling
    
    test_region_size_vs_alpha_scaling(experiment_name);
    
end

function detailed_vs_histogram(vals,execs,alphas,sizes,tol)
    [counts1,bins1] = hist(vals(1,:),30);
    [counts2,bins2] = hist(vals(2,:),30);
    if max(bins1) > max(bins2)
        bins = bins1;
    else
        bins = bins2;
    end
    barh(bins,counts1,'r');
    hold on;
    barh(bins,-counts2);
    xlabel("Similarity to desired (x,y) traj (closer to 0 is better)")
    ylabel("Number of trajectories  (random samples from (alpha,tr))")
    xrange = xlim;
    if(abs(xrange(1)) > abs(xrange(2)))
       new_range = [xrange(1),-xrange(1)];
    else
       new_range = [-xrange(2),xrange(2)]; 
    end
    xlim(new_range);
    ylim([0,60]);
    
    yy = 6.5;
    plot(new_range,[yy,yy],'k--');
    yy = 3.75;
    plot(new_range,[yy,yy],'k--');
    
    % Find good example
    [~,argood] = min(vals(2,:));
    tmp =execs{2}+"_trsize_" + num2str(sizes(argood)) + ...
                                               "_alphasc_" + num2str(alphas(argood)) + "_tol_" + num2str(tol);
    cd('../logs')
    cd(char(tmp))
    dirs = struct2cell(dir);
    cd(dirs{1,3});
    traj = load("xs.txt");
    axes('Position',[.2 .22 .2 .2])
    box on
    plot_trajectory(traj)
    cd('../../../matlab')
    % Find bad example 
    [~,argbad] = max(vals(2,:) > 50);
    tmp =execs{2}+"_trsize_" + num2str(sizes(argbad)) + ...
                                               "_alphasc_" + num2str(alphas(argbad)) + "_tol_" + num2str(tol);
    cd('../logs')
    cd(char(tmp))
    dirs = struct2cell(dir);
    cd(dirs{1,3});
    traj = load("xs.txt");
    axes('Position',[.2 .7 .2 .2])
    box on
    plot_trajectory(traj)
    cd('../../../matlab')
    % Find mid example (hardcoded)
    % [~,argmid]=max((vals(2,:) < bins(ceil(length(bins)/2)) + 5.0) & (vals(2,:) > bins(floor(length(bins)/2)) - 5.0));
    [~,argmid]=max((vals(2,:) < 5.5) & (vals(2,:) > 4.5));
    tmp =execs{2}+"_trsize_" + num2str(sizes(argmid)) + ...
                                               "_alphasc_" + num2str(alphas(argmid)) + "_tol_" + num2str(tol);
    cd('../logs')
    cd(char(tmp))
    dirs = struct2cell(dir);
    cd(dirs{1,3});
    traj = load("xs.txt");
    axes('Position',[.2 .45 .2 .2])
    box on
    plot_trajectory(traj)
    cd('../../../matlab')
    
end


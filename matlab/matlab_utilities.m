% 1 or 0
rng(0)

% Test region size vs. alpha scaling
%test_region_size_vs_alpha_scaling('three_player_intersection',5,6);
sensitivity_to_params('ideal_intersection_trajectory',...
   {'three_player_flat_intersection','three_player_intersection'},324,true,0.1);

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
function sensitivity_to_params(ideal_traj,execs,num_samples,randornot,tol)
    vals = zeros([length(execs), num_samples]);
    alphas = zeros([length(execs), num_samples]);
    sizes = zeros([length(execs), num_samples]);
    times = zeros([length(execs), num_samples]);
    % This threshold sets which points in the scatterplot are successes vs
    % failures
    threshold = 1.0;
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

    tr_min = 1.0; tr_max = 10.0;
    al_min = 0.1; al_max = 0.75;
    range_for_tregion = [tr_min,tr_max];%linspace(0.0,20.0,n);
    range_for_alphasc = [al_min,al_max];

    if ~randornot
        trr = linspace(tr_min,tr_max,ceil(sqrt(num_samples)));
        alp = linspace(al_min,al_max,ceil(sqrt(num_samples)));
        [all_points_tr,all_points_al] = ...
            meshgrid(trr,alp);
        alltr_size = all_points_tr(:);
        allalpha_scaling = all_points_al(:);
    end

    for i=1:num_samples
       if randornot
        tr_size = random('unif',range_for_tregion(1),range_for_tregion(2));
        alpha_scaling = random('unif',range_for_alphasc(1),range_for_alphasc(2));
       else
        tr_size = alltr_size(i);
        alpha_scaling = allalpha_scaling(i);
       end

       % Convert parameters loop parameters to strings for names
       string_alpha_sc = string(num2str(alpha_scaling));
       string_tr_size  = string(num2str(tr_size));
       for j=1:length(execs)
           tmp =execs{j}+"_trsize_" + num2str(tr_size) + ...
                                               "_alphasc_" + num2str(alpha_scaling) + "_tol_" + num2str(tol);
           exp_folder_name = "'" + tmp + "'";
           experiment_name = experiment_arg + exp_folder_name;

           exists = check_if_folder_exists(char(tmp));
           if ~exists
               % Stitch together the command for the executable.
               instruction = folder + execs{j} + flag1 + string_tr_size + ...
                             flag2 + string_alpha_sc + " -convergence_tolerance="+num2str(tol)+ ...
                             " -viz=false -save=true -last_traj=true " + experiment_name;
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

    vals_min = min(min(vals));
    vals_max = max(max(vals));
    for i=1:num_samples
        for j=1:length(execs)
           if isnan(vals(j,i))
                vals(j,i) = vals_max-0.05;
                fprintf("crap" + num2str(i) + num2str(j) + "\n")
           end
        end
    end

    figure(1)
    detailed_vs_histogram(vals,execs,alphas(1,:),sizes(1,:),tol);
%     title("Flat algorithm.")
%     figure(2)
%     detailed_histogram(vals(2,:),alphas(2,:),sizes(2,:));
%     title("Non-Flat algorithm.")
%     figure(1)

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
            title("("+execs{j}+")"+"Flat Algorithm " + "("+mean(true_times{j})+")");
        else
            title("("+execs{j}+")"+"Non-Flat Algorithm " + "("+mean(true_times{j})+")");
        end
        xlabel("initial alpha scaling");
        ylabel("trust region size");
    end

    cd('../matlab')
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
function [data, tau, extraOuts] = ...
    runningSumHJIPDE_solve(data0, tau, schemeData, compMethod, extraArgs)
% [data, tau, extraOuts] = ...
%   HJIPDE_solve(data0, tau, schemeData, minWith, extraargs)
%     Solves HJIPDE with initial conditions data0, at times tau, and with
%     parameters schemeData and extraArgs
%
% ----- How to use this function -----
%
% Inputs:
%   data0      - initial value function
%   tau        - list of computation times
%   schemeData - problem parameters passed into the Hamiltonian function
%                  .grid: grid (required!)
%   compMethod - set to 'set' or 'none' to compute reachable set (not tube)
%              - set to 'zero' to do min with zero
%              - set to 'none' to compute reachable set (not tube)
%              - set to 'minVOverTime' to do min with previous data
%              - set to 'minVWithTarget' to do min with original data
%              - set to 'maxVOverTime' to do max over time
%   extraArgs  - this structure can be used to leverage other additional
%                functionalities within this function. Its subfields are:
%     .obstacles:  a single obstacle or a list of obstacles with time
%                  stamps tau (obstacles must have same time stamp as the
%                  solution)
%     .keepLast:  Only keep data from latest time stamp and delete previous
%                 datas
%     .compRegion: unused for now (meant to limit computation region)
%     .visualize:  set to true to visualize reachable set
%     .RS_level:  level set of reachable set to visualize (defaults to 0)
%     .plotData:   information required to plot the data (need to fill in)
%     .deleteLastPlot:
%         set to true to delete previous plot before displaying next one
%     .fig_num:   List if you want to plot on a specific figure number
%     .fig_filename:
%         provide this to save the figures (requires export_fig package)
%     .stopInit:   stop the computation once the reachable set includes the
%                  initial state
%     .stopSetInclude:
%         stops computation when reachable set includes this set
%     .stopSetIntersect:
%         stops computation when reachable set intersects this set
%     .stopLevel:  level of the stopSet to check the inclusion for. Default
%                  level is zero.
%     .targets:    a single target or a list of targets with time
%                  stamps tau (targets must have same time stamp as the
%                  solution). This functionality is mainly useful when the
%                  targets are time-varying, in case of variational
%                  inequality for example; data0 can be used to
%                  specify the target otherwise.
%     .stopConverge:
%         set to true to stop the computation when it converges
%     .convergeThreshold:
%         Max change in each iteration allowed when checking convergence
%
%     .SDModFunc, .SDModParams:
%         Function for modifying scheme data every time step given by tau.
%         Currently this is only used to switch between using optimal control at
%         every grid point and using maximal control for the SPP project when
%         computing FRS using centralized controller
%
%     .save_filename, .saveFrequency:
%         file name under which temporary data is saved at some frequency in
%         terms of the number of time steps
%
% Outputs:
%   data - solution corresponding to grid g and time vector tau
%   tau  - list of computation times (redundant)
%   extraOuts - This structure can be used to pass on extra outputs, for
%               example:
%      .stoptau: time at which the reachable set contains the initial
%                state; tau and data vectors only contain the data till
%                stoptau time.
%      .hT:      figure handle
%

%% Default parameters
if numel(tau) < 2
    error('Time vector must have at least two elements!')
end

if nargin < 4
    compMethod = 'zero';
end

if nargin < 5
    extraArgs = [];
end

extraOuts = [];
quiet = false;
low_memory = false;
keepLast = false;
flip_output = false;

small = 1e-4;
g = schemeData.grid;
gDim = g.dim;
clns = repmat({':'}, 1, gDim);

%% Extract the information from extraargs
% Quiet mode
if isfield(extraArgs, 'quiet') && extraArgs.quiet
    fprintf('HJIPDE_solve running in quiet mode...\n')
    quiet = true;
end

% Low memory mode
if isfield(extraArgs, 'low_memory') && extraArgs.low_memory
    fprintf('HJIPDE_solve running in low memory mode...\n')
    low_memory = true;
    
    % Save the output in reverse order
    if isfield(extraArgs, 'flip_output') && extraArgs.flip_output
        flip_output = true;
    end
    
end

if isfield(extraArgs, 'keepLast') && extraArgs.keepLast
    keepLast = true;
end



% Extract the information about obstacles
obsMode = 'none';
if isfield(extraArgs, 'obstacles')
    obstacles = extraArgs.obstacles;
    
    if numDims(obstacles) == gDim
        obsMode = 'static';
        obstacle_i = obstacles;
    elseif numDims(obstacles) == gDim + 1
        obsMode = 'time-varying';
        obstacle_i = obstacles(clns{:}, 1);
    else
        error('Inconsistent obstacle dimensions!')
    end
    data0 = max(data0, -obstacle_i);
end

% Extract the information about targets
if isfield(extraArgs, 'targets')
    targets = extraArgs.targets;
end

% Check validity of stopInit if needed
if isfield(extraArgs, 'stopInit')
    if ~isvector(extraArgs.stopInit) || gDim ~= length(extraArgs.stopInit)
        error('stopInit must be a vector of length g.dim!')
    end
end

% Check validity of stopSet if needed
if isfield(extraArgs, 'stopSet') % For backwards compatibility
    extraArgs.stopSetInclude = extraArgs.stopSet;
end

if isfield(extraArgs,'stopSetInclude') || isfield(extraArgs,'stopSetIntersect')
    if isfield(extraArgs,'stopSetInclude')
        stopSet = extraArgs.stopSetInclude;
    else
        stopSet = extraArgs.stopSetIntersect;
    end
    
    if numDims(stopSet) ~= gDim || any(size(stopSet) ~= g.N')
        error('Inconsistent stopSet dimensions!')
    end
    
    % Extract set of indices at which stopSet is negative
    setInds = find(stopSet(:) < 0);
    
    % Check validity of stopLevel if needed
    if isfield(extraArgs, 'stopLevel')
        stopLevel = extraArgs.stopLevel;
    else
        stopLevel = 0;
    end
end

%% Visualization
%if isfield(extraArgs, 'visualize') && (extraArgs.visualize )
if (isfield(extraArgs, 'visualize') && extraArgs.visualize)...
        || (isfield(extraArgs, 'visualizeLevelSet') && extraArgs.visualizeLevelSet)...
        || (isfield(extraArgs, 'visualizeValueFunction') && extraArgs.visualizeValueFunction)
    
    timeCount = 0;
    
    if isfield(extraArgs, 'makeVideo') && extraArgs.makeVideo
        if ~isfield(extraArgs, 'video_filename')
            extraArgs.video_filename = [datestr(now,'YYYYMMDD_hhmmss') '.mp4'];
        end
        
        vout = VideoWriter(extraArgs.video_filename,'MPEG-4');
        vout.Quality = 100;
        if isfield(extraArgs, 'frameRate')
            vout.FrameRate = extraArgs.frameRate;
        else
            vout.FrameRate = 30;
        end
        
        try
            vout.open;
        catch
            error('cannot open file for writing')
            return
        end
    end
    
    
    if isfield(extraArgs, 'visualize') && strcmp(extraArgs.visualize,'function')
        extraArgs.visualize = 1;
        extraArgs.visualizeValueFunction = 1;
    
    else
        extraArgs.visualize = 1;
        extraArgs.visualizeLevelSet = 1;
        
        RS_level = 0;
        if isfield(extraArgs, 'RS_level')
            RS_level = extraArgs.RS_level;
        end
        
        if isfield(extraArgs, 'plotColorT')
            L = extraArgs.plotColorT;
        else
            L = 'k';
        end
        
        if isfield(extraArgs, 'plotColorT0')
            L0 = extraArgs.plotColorT0;
        else
            L0 = 'k';
        end
        
        if isfield(extraArgs, 'plotColorTF0')
            LF0 = extraArgs.plotColorTF0;
        else
            LF0 = 'g';
        end
        
        if isfield(extraArgs,'plotColorO')
            G = extraArgs.plotColorO;
        else
            G = 'k';
        end
        
    end
    
    % Extract the information about plotData
    plotDims = ones(gDim, 1);
    projpt = [];
    if isfield(extraArgs, 'plotData')
        % Dimensions to visualize
        % It will be an array of 1s and 0s with 1s means that dimension should
        % be plotted.
        plotDims = extraArgs.plotData.plotDims;
        % Points to project other dimensions at. There should be an entry point
        % corresponding to each 0 in plotDims.
        projpt = extraArgs.plotData.projpt;
    end
    
    deleteLastPlot = false;
    if isfield(extraArgs, 'deleteLastPlot')
        deleteLastPlot = extraArgs.deleteLastPlot;
    end
    
    % Initialize the figure for visualization
    if isfield(extraArgs,'fig_num')
        f = figure(extraArgs.fig_num);
        clf
    else
        f = figure;
        clf
    end
    hold on
    need_light = true;
    
    
    
    if isfield(extraArgs, 'stopInit')
        projectedInit = extraArgs.stopInit(logical(plotDims));
        if nnz(plotDims) == 2
            plot(projectedInit(1), projectedInit(2), 'b*')
        elseif nnz(plotDims) == 3
            plot3(projectedInit(1), projectedInit(2), projectedInit(3), 'b*')
        end
    end
    
    grid on
    
    % Number of dimensions to be plotted and to be projected
    pDims = nnz(plotDims);
    projDims = length(projpt);
    
    % Basic Checks
    if(length(plotDims) ~= gDim || projDims ~= (gDim - pDims))
        error('Mismatch between plot and grid dimensions!');
    end
    
    if (pDims > 4)
        error('Currently plotting up to 3D is supported!');
    end
    
    % Visualize the reachable set
    figure(f)
    viewFunc = 0;
    
    % Project
    if projDims == 0
        gPlot = g;
        dataPlot = data0;
        
        if isfield(extraArgs, 'obstacles')
        % if strcmp(obsMode, 'time-varying')
            obsPlot = obstacle_i;
         end
    else
        [gPlot, dataPlot] = proj(g, data0, ~plotDims, projpt);
        
        if isfield(extraArgs, 'obstacles')
        %if strcmp(obsMode, 'time-varying')
            [~, obsPlot] = proj(g, obstacle_i, ~plotDims, projpt);
        end
    end
    
    
    eAT_visSetIm.sliceDim = gPlot.dim;
    eAT_visSetIm.applyLight = false;
    if isfield(extraArgs, 'lineWidth')
        eAT_visSetIm.LineWidth = extraArgs.lineWidth;
    end
    
    
    if isfield(extraArgs, 'visualizeTarget') && extraArgs.visualizeTarget
        extraOuts.hT0 = visSetIm(gPlot, dataPlot, L0, RS_level, eAT_visSetIm);
    end
    
    if isfield(extraArgs, 'visualizeTargetFunction') &&...
            extraArgs.visualizeTargetFunction
        viewFunc = 1;
        %extraOuts.hT0 = visSetIm(gPlot, dataPlot, L, RS_level, eAT_visSetIm);
        
        extraOuts.hTf0 = surf(gPlot.xs{1}, gPlot.xs{2}, dataPlot);
        extraOuts.hTf0.EdgeColor = 'none';
        
        if isfield(extraArgs,'plotColorTF0')
            extraOuts.hTf0.FaceColor = LF0;
        else
            extraOuts.hTf0.FaceColor = 'b';
        end
        
        if isfield(extraArgs,'plotAlphaTF0')
            extraOuts.hTf0.FaceAlpha = extraArgs.plotAlphaTF;
        else
            extraOuts.hTf0.FaceAlpha= .5;
        end
    end
        
    if isfield(extraArgs, 'visualizeLevelSet') && extraArgs.visualizeLevelSet
        extraOuts.hT = visSetIm(gPlot, dataPlot, L, RS_level, eAT_visSetIm);
        
        if strcmp(obsMode, 'static')
            if all(plotDims)
                extraOuts.hO = visSetIm(g, obstacle_i, G);
            else
                [~, obsPlot] = proj(g, obstacle_i, ~plotDims, projpt);
                extraOuts.hO = visSetIm(gPlot, obsPlot, G);
            end
            if isfield(extraArgs, 'lineWidth')
                extraOuts.hO.LineWidth = extraArgs.lineWidth;
            end
        elseif strcmp(obsMode, 'time-varying')
            eAO_visSetIm.applyLight = false;
                if isfield(extraArgs, 'lineWidth')
                    eAO_visSetIm.LineWidth = extraArgs.lineWidth;
                end
                extraOuts.hO = visSetIm(gPlot, obsPlot, G, 0, eAO_visSetIm);
            end
        end
    
    if isfield(extraArgs, 'visualizeValueFunction') && ...
            extraArgs.visualizeValueFunction
        viewFunc = 1;
        extraOuts.hTf = surf(gPlot.xs{1}, gPlot.xs{2}, dataPlot);
        extraOuts.hTf.EdgeColor = 'none';
        
        if isfield(extraArgs,'plotColorTF')
            extraOuts.hTf.FaceColor = extraArgs.plotColorTF;
        else
            extraOuts.hTf.FaceColor = 'b';
        end
        
        if isfield(extraArgs,'plotAlphaTF')
            extraOuts.hTf.FaceAlpha = extraArgs.plotAlphaTF;
        else
            extraOuts.hTf.FaceAlpha= .5;
        end
        
        
        %visSetIm(gPlot, dataPlot, 'r', RS_level, eAT_visSetIm);
        
        
        %end
        
    end
    
    if isfield(extraArgs, 'obstacles') && ...
            isfield(extraArgs, 'visualizeObstaclesFunction') &&...
            extraArgs.visualizeObstaclesFunction
        viewFunc = 1;
        %if strcmp(obsMode, 'time-varying')
        extraOuts.hOf =  surf(gPlot.xs{1}, gPlot.xs{2}, -obsPlot);
        %visSetIm(gPlot, obsPlot, 'k', 0, eAO_visSetIm);
        extraOuts.hOf.EdgeColor = 'none';
        if isfield(extraArgs,'plotColorOF')
            extraOuts.hOf.FaceColor = extraArgs.plotColorOF;
        else
            extraOuts.hOf.FaceColor = 'r';
        end
        
        if isfield(extraArgs,'plotAlphaOF')
            extraOuts.hOf.FaceAlpha = extraArgs.plotAlphaOF;
        else
            extraOuts.hOf.FaceAlpha = .5;
        end
    end
    
    if pDims >2 || viewFunc || isfield(extraArgs, 'viewAngle')
        if isfield(extraArgs, 'viewAngle')
            view(extraArgs.viewAngle)
        else
            view(30,10)
        end
    end
    
    
    c = camlight;
    
    if isfield(extraArgs, 'camlightPosition')
        c.Position = extraArgs.camlightPosition;
    else
        c.Position = [-30 -30 -30];
    end
    
    if isfield(extraArgs, 'viewGrid') && ~extraArgs.viewGrid
        grid off
    end
    
    if isfield(extraArgs, 'viewAxes')
        axis(extraArgs.viewAxes)
    end
    axis square
    
    if isfield(extraArgs, 'xTitle')
        xlabel(extraArgs.xTitle, 'interpreter','latex')
    end
    
    
    if isfield(extraArgs, 'yTitle')
        ylabel(extraArgs.yTitle,'interpreter','latex')
    end
    
    
    if isfield(extraArgs, 'zTitle')
        zlabel(extraArgs.zTitle,'interpreter','latex')
    end
    
    if need_light && (gPlot.dim == 3)
        camlight left
        camlight right
        need_light = false;
    end
    title(['t = ' num2str(0) ' s'])
    set(gcf,'Color','white')
    
    if isfield(extraArgs, 'fontSize')
        set(gca,'FontSize',extraArgs.fontSize)
    end
    
    if isfield(extraArgs, 'lineWidth')
         set(gca,'LineWidth',extraArgs.lineWidth)
    end
        
    drawnow;
    
    if isfield(extraArgs, 'makeVideo') && extraArgs.makeVideo
        current_frame = getframe(gcf); %gca does just the plot
        writeVideo(vout,current_frame);
    end
    
    if isfield(extraArgs, 'fig_filename')
        export_fig(sprintf('%s%d', extraArgs.fig_filename, 0), '-png')
    end
end


%% Extract cdynamical system if needed
if isfield(schemeData, 'dynSys') && ~isfield(schemeData, 'hamFunc')
    schemeData.hamFunc = @genericHam;
    schemeData.partialFunc = @genericPartial;
end

stopConverge = false;
if isfield(extraArgs, 'stopConverge')
    stopConverge = extraArgs.stopConverge;
    if isfield(extraArgs, 'convergeThreshold')
        convergeThreshold = extraArgs.convergeThreshold;
    else
        convergeThreshold = 1e-5;
    end
end

%% SchemeFunc and SchemeData
schemeFunc = @termLaxFriedrichs;
% Extract accuracy parameter o/w set default accuracy
accuracy = 'veryHigh';
if isfield(schemeData, 'accuracy')
    accuracy = schemeData.accuracy;
end

%% Numerical approximation functions
dissType = 'global';
[schemeData.dissFunc, integratorFunc, schemeData.derivFunc] = ...
    getNumericalFuncs(dissType, accuracy);

if strcmp(compMethod, 'minWithZero') || strcmp(compMethod, 'zero')
    schemeFunc = @termRestrictUpdate;
    schemeData.innerFunc = @termLaxFriedrichs;
    schemeData.innerData = schemeData;
    schemeData.innerData = schemeData;
    schemeData.positive = 0;
end

%% Time integration
integratorOptions = odeCFLset('factorCFL', 0.8, 'singleStep', 'on');

startTime = cputime;

%% Stochastic additive terms
if isfield(extraArgs, 'addGaussianNoiseStandardDeviation')
    % We are taking all the previous scheme terms and adding noise to it
    % Save all the previous terms as the deterministic component in detFunc
    detFunc = schemeFunc;
    detData = schemeData;
      % The full computation scheme will include this added term so clear
      % out the schemeFunc so we can pack everything back in later with the
      % new stuff
    clear schemeFunc schemeData;

    % Create the Hessian term corresponding to white noise diffusion
    stochasticFunc = @termTraceHessian;
    stochasticData.grid = g;
    stochasticData.L = extraArgs.addGaussianNoiseStandardDeviation';
    stochasticData.R = extraArgs.addGaussianNoiseStandardDeviation;
    stochasticData.hessianFunc = @hessianSecond;

    % Add the (saved) deterministic terms and the (new) stochastic term
    % together into the complete scheme
    schemeFunc = @termSum;
    schemeData.innerFunc = { detFunc; stochasticFunc };
    schemeData.innerData = { detData; stochasticData };
end

%% Initialize PDE solution
data0size = size(data0);

if numDims(data0) == gDim
    % New computation
    if keepLast
        data = data0;
    elseif low_memory
        data = single(data0);
    else
        data = zeros([data0size(1:gDim) length(tau)]);
        data(clns{:}, 1) = data0;
    end
    
    istart = 2;
elseif numDims(data0) == gDim + 1
    % Continue an old computation
    if keepLast
        data = data0(clns{:}, data0size(end));
    elseif low_memory
        data = single(data0(clns{:}, data0size(end)));
    else
        data = zeros([data0size(1:gDim) length(tau)]);
        data(clns{:}, 1:data0size(end)) = data0;
    end
    
    % Start at custom starting index if specified
    if isfield(extraArgs, 'istart')
        istart = extraArgs.istart;
    else
        istart = data0size(end)+1;
    end
else
    error('Inconsistent initial condition dimension!')
end

for i = istart:length(tau)
    if ~quiet
        fprintf('tau(i) = %f\n', tau(i))
    end
    %% Variable schemeData
    if isfield(extraArgs, 'SDModFunc')
        if isfield(extraArgs, 'SDModParams')
            paramsIn = extraArgs.SDModParams;
        else
            paramsIn = [];
        end
        
        schemeData = extraArgs.SDModFunc(schemeData, i, tau, data, obstacles, ...
            paramsIn);
    end
    
    if keepLast
        y0 = data;
    elseif low_memory
        if flip_output
            y0 = data(clns{:}, 1);
        else
            y0 = data(clns{:}, size(data, g.dim+1));
        end
        
    else
        y0 = data(clns{:}, i-1);
    end
    y = y0(:);
    
    tNow = tau(i-1);
    
    %% Main integration loop to get to the next tau(i)
    while tNow < tau(i) - small
        % Save previous data if needed
        if strcmp(compMethod, 'minVOverTime') || ...
                strcmp(compMethod, 'maxVOverTime')
            yLast = y;
        end
        
        if ~quiet
            fprintf('  Computing [%f %f]...\n', tNow, tau(i))
        end
        
        [tNow, y] = feval(integratorFunc, schemeFunc, [tNow tau(i)], y, ...
            integratorOptions, schemeData);
        
        if any(isnan(y))
            keyboard
        end
        
        
        %Tube Computations
        %   compMethod - set to 'set' or 'none' to compute reachable set (not tube)
        %              - set to 'zero' to do min with zero
        %              - set to 'none' to compute reachable set (not tube)
        %              - set to 'minVOverTime' to do min with previous data
        %              - set to 'minVWithTarget' to do min with original data
        %              - set to 'maxVOverTime' to do max over time
        if strcmp(compMethod, 'minVOverTime') %Min over Time
            y = min(y, yLast);
        elseif strcmp(compMethod, 'maxVOverTime')
            y = max(y, yLast);
        elseif strcmp(compMethod, 'minVWithTarget')%Min with data0
            y = min(y,data0(:));
        elseif strcmp(compMethod, 'maxVWithTarget')
            y = max(y,data0(:));
            
        end
        
        % Min with targets
        if isfield(extraArgs, 'targets')
            if numDims(targets) == gDim
                y = min(y, targets(:));
            else
                target_i = targets(clns{:}, i);
                y = min(y, target_i(:));
            end
        end
        
        % "Mask" using obstacles
        if isfield(extraArgs, 'obstacles')
            if strcmp(obsMode, 'time-varying')
                obstacle_i = obstacles(clns{:}, i);
            end
            y = max(y, -obstacle_i(:));
        end
    end
    
    if stopConverge
        change = max(abs(y - y0(:)));
        if ~quiet
            fprintf('Max change since last iteration: %f\n', change)
        end
    end
    
    % Reshape value function
    data_i = reshape(y, g.shape);
    if keepLast
        data = data_i;
    elseif low_memory
        if flip_output
            data = cat(g.dim+1, reshape(y, g.shape), data);
        else
            data = cat(g.dim+1, data, reshape(y, g.shape));
        end
        
    else
        data(clns{:}, i) = data_i;
    end
    
    %% If commanded, stop the reachable set computation once it contains
    % the initial state.
    if isfield(extraArgs, 'stopInit')
        initValue = eval_u(g, data_i, extraArgs.stopInit);
        if ~isnan(initValue) && initValue <= 0
            extraOuts.stoptau = tau(i);
            tau(i+1:end) = [];
            
            if ~low_memory && ~keepLast
                data(clns{:}, i+1:size(data, gDim+1)) = [];
            end
            break
        end
    end
    
    %% Stop computation if reachable set contains a "stopSet"
    if exist('stopSet', 'var')
        dataInds = find(data_i(:) <= stopLevel);
        
        if isfield(extraArgs, 'stopSetInclude')
            stopSetFun = @all;
        else
            stopSetFun = @any;
        end
        
        if stopSetFun(ismember(setInds, dataInds))
            extraOuts.stoptau = tau(i);
            tau(i+1:end) = [];
            
            if ~low_memory && ~keepLast
                data(clns{:}, i+1:size(data, gDim+1)) = [];
            end
            break
        end
    end
    
    fprintf('Change in value: %f\n', change);
    
    if stopConverge && change < convergeThreshold
        extraOuts.stoptau = tau(i);
        tau(i+1:end) = [];
        
        if ~low_memory && ~keepLast
            data(clns{:}, i+1:size(data, gDim+1)) = [];
        end
        break
    end
    
    %% If commanded, visualize the level set.
    
    if isfield(extraArgs, 'visualize') && extraArgs.visualize
        timeCount = timeCount + 1;
        % Number of dimensions to be plotted and to be projected
        pDims = nnz(plotDims);
        projDims = length(projpt);
        
        % Basic Checks
        if(length(plotDims) ~= gDim || projDims ~= (gDim - pDims))
            error('Mismatch between plot and grid dimensions!');
        end
        
        if (pDims > 4)
            error('Currently plotting up to 3D is supported!');
        end
        
        % Visualize the reachable set
        %fig_handle=figure(f);
        
        % Delete last plot
        if deleteLastPlot
            if isfield(extraOuts, 'hT')
                if iscell(extraOuts.hT)
                    for hi = 1:length(extraOuts.hT)
                        delete(extraOuts.hT{hi})
                    end
                else
                    delete(extraOuts.hT);
                end
            end
            
            if isfield(extraOuts, 'hTf')
                if iscell(extraOuts.hTf)
                    for hi = 1:length(extraOuts.hTf)
                        delete(extraOuts.hTf{hi})
                    end
                else
                    delete(extraOuts.hTf);
                end
            end
            
            if isfield(extraOuts, 'hO') && strcmp(obsMode, 'time-varying')
                delete(extraOuts.hO);
            end
            
            if isfield(extraOuts, 'hOf') && strcmp(obsMode, 'time-varying')
                delete(extraOuts.hOf);
            end
        end
        
        % Project
        if projDims == 0
            gPlot = g;
            dataPlot = data_i;
            
            if strcmp(obsMode, 'time-varying')
                obsPlot = obstacle_i;
            end
        else
            [gPlot, dataPlot] = proj(g, data_i, ~plotDims, projpt);
            
            if strcmp(obsMode, 'time-varying')
                [~, obsPlot] = proj(g, obstacle_i, ~plotDims, projpt);
            end
        end
        
        eAT_visSetIm.sliceDim = gPlot.dim;
        eAT_visSetIm.applyLight = false;
        
        if isfield(extraArgs, 'lineWidth')
         eAT_visSetIm.LineWidth = extraArgs.lineWidth;
        end
        
        if isfield(extraArgs, 'visualizeLevelSet') && extraArgs.visualizeLevelSet
            extraOuts.hT = visSetIm(gPlot, dataPlot, L, RS_level, eAT_visSetIm);
            
            if strcmp(obsMode, 'time-varying')
                eAO_visSetIm.applyLight = false;
                
                if isfield(extraArgs, 'lineWidth')
                    eAO_visSetIm.LineWidth = extraArgs.lineWidth;
                end
                extraOuts.hO = visSetIm(gPlot, obsPlot, G, 0, eAO_visSetIm);
            end
        end
        
        if isfield(extraArgs, 'visualizeValueFunction') && ...
                extraArgs.visualizeValueFunction
            
            
            %        set(0,'CurrentFigure',fig_handle);
            extraOuts.hTf = surf(gPlot.xs{1}, gPlot.xs{2}, dataPlot);
            
            extraOuts.hTf.EdgeColor = 'none';
            extraOuts.hTf.FaceLighting = 'phong';
            
            if isfield(extraArgs,'plotColorTF')
                extraOuts.hTf.FaceColor = extraArgs.plotColorTF;
            else
                extraOuts.hTf.FaceColor = 'b';
            end
            
            if isfield(extraArgs,'plotAlphaTF')
                extraOuts.hTf.FaceAlpha = extraArgs.plotAlphaTF;
            else
                extraOuts.hTf.FaceAlpha= .5;
            end
            
            
            %visSetIm(gPlot, dataPlot, 'r', RS_level, eAT_visSetIm);
            
            
            
            
            
        end
        
        if strcmp(obsMode, 'time-varying') && ...
                isfield(extraArgs, 'visualizeObstaclesFunction') &&...
                extraArgs.visualizeObstaclesFunction
            eAO_visSetIm.applyLight = false;
            extraOuts.hOf =  surf(gPlot.xs{1}, gPlot.xs{2}, -obsPlot);
            %visSetIm(gPlot, obsPlot, 'k', 0, eAO_visSetIm);
            extraOuts.hOf.EdgeColor = 'none';
            extraOuts.hOf.FaceLighting = 'phong';
            if isfield(extraArgs,'plotColorOF')
                extraOuts.hOf.FaceColor = extraArgs.plotColorOF;
            else
                extraOuts.hOf.FaceColor = 'r';
            end
            
            if isfield(extraArgs,'plotAlphaOF')
                extraOuts.hOf.FaceAlpha = extraArgs.plotAlphaOF;
            else
                extraOuts.hOf.FaceAlpha = .5;
            end
        end
        
        
        if isfield(extraArgs, 'viewAxes')
            axis(extraArgs.viewAxes)
        end
        axis square
        
        if isfield(extraArgs, 'xTitle')
            xlabel(extraArgs.xTitle, 'interpreter','latex')
        end
        
        
        if isfield(extraArgs, 'yTitle')
            ylabel(extraArgs.yTitle, 'interpreter','latex')
        end
        
        
        if isfield(extraArgs, 'zTitle')
            zlabel(extraArgs.zTitle, 'interpreter','latex')
        end
        
        if need_light && (gPlot.dim == 3)
            camlight left
            camlight right
            need_light = false;
        end
        
        if ~isfield(extraArgs, 'dtTime')
            title(['t = ' num2str(tNow) ' s'])
        elseif isfield(extraArgs, 'dtTime') && ...
            floor(extraArgs.dtTime/((tau(end)-tau(1))/length(tau))) ...
            == timeCount
            
            title(['t = ' num2str(tNow) ' s'])
            timeCount = 0;
        end
        drawnow;
        
        
        
        if isfield(extraArgs, 'makeVideo') && extraArgs.makeVideo
            current_frame = getframe(gcf); %gca does just the plot
            writeVideo(vout,current_frame);
        end
        
        if isfield(extraArgs, 'fig_filename')
            export_fig(sprintf('%s%d', extraArgs.fig_filename, i), '-png')
        end
    end
    
    %% Save the results if needed
    if isfield(extraArgs, 'save_filename')
        if mod(i, extraArgs.saveFrequency) == 0
            ilast = i;
            save(extraArgs.save_filename, 'data', 'tau', 'ilast', '-v7.3')
        end
    end
end

endTime = cputime;
if ~quiet;
    fprintf('Total execution time %g seconds\n', endTime - startTime);
end

if isfield(extraArgs, 'makeVideo') && extraArgs.makeVideo
    vout.close
end

end

function [dissFunc, integratorFunc, derivFunc] = ...
    getNumericalFuncs(dissType, accuracy)
% Dissipation
switch(dissType)
    case 'global'
        dissFunc = @artificialDissipationGLF;
    case 'local'
        dissFunc = @artificialDissipationLLF;
    case 'locallocal'
        dissFunc = @artificialDissipationLLLF;
    otherwise
        error('Unknown dissipation function %s', dissFunc);
end

% Accuracy
switch(accuracy)
    case 'low'
        derivFunc = @upwindFirstFirst;
        integratorFunc = @odeCFL1;
    case 'medium'
        derivFunc = @upwindFirstENO2;
        integratorFunc = @odeCFL2;
    case 'high'
        derivFunc = @upwindFirstENO3;
        integratorFunc = @odeCFL3;
    case 'veryHigh'
        derivFunc = @upwindFirstWENO5;
        integratorFunc = @odeCFL3;
    otherwise
        error('Unknown accuracy level %s', accuracy);
end
end
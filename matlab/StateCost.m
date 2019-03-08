classdef StateCost
    %STATECOST A time-varying, state dependent cost. 
    
    methods (Abstract)
        getCost(obj, t, x)
    end
end


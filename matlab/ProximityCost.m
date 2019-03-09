classdef ProximityCost < StateCost
    %PROXIMITYCOST A state-dependent cost that penalizes proximity.
    
    properties
        positionIndices % Indices of the state corresponding to position
        point           % Point from which to compute proximity
        maxDistance     % Maximum distance to penalize
        outsideWeight   % Weight of the quadratic cost beyond the threshold
        
        maxSquaredDistance
    end
    
    methods
        function obj = ProximityCost(...
            positionIndices, point, maxDistance, outsideWeight)
            obj.positionIndices = positionIndices; 
            obj.point = point;
            obj.maxDistance = maxDistance;
            obj.outsideWeight = outsideWeight;
            
            obj.maxSquaredDistance = maxDistance^2;
        end
        
        function cost = getCost(obj, ~, x)
            %GETCOST Computes the proximity cost
            dx = x{obj.positionIndices(1)} - obj.point(1);
            dy = x{obj.positionIndices(2)} - obj.point(2);
            relSquaredDistance = dx.^2 + dy.^2;
            
            if relSquaredDistance < obj.maxSquaredDistance
                cost = -relSquaredDistance;
            else
                outsidePenalty = obj.outsideWeight * (...
                    relSquaredDistance + obj.maxSquaredDistance - ...
                    2.*sqrt(relSquaredDistance * obj.maxSquaredDistance));
                cost = -outsidePenalty - obj.maxSquaredDistance;
            end
        end
    end
end


classdef ObstacleCost < StateCost
    %OBSTACLECOST A cost that penalizes distance from an obstacle center
        
    properties
        positionIndices % Indices of the state corresponding to position
        point           % Point from which to compute proximity
        maxDistance     % Maximum distance to penalize
    end
    
    methods
        function obj = ObstacleCost(positionIndices, point, maxDistance)
            obj.positionIndices = positionIndices;
            obj.point = point;
            obj.maxDistance = maxDistance;
        end
        
        function cost = getCost(obj, ~, x)
            %GETCOST Computes the obstacle cost
            dx = x{obj.positionIndices(1)} - obj.point(1);
            dy = x{obj.positionIndices(2)} - obj.point(2);
            relDistance = sqrt(dx.^2 + dy.^2);
            
            cost = min(relDistance - obj.maxDistance, 0).^2;
        end
    end
end


classdef SemiquadraticCost < StateCost
    %SEMIQUADRATICCOST A state-dependent semiquadratic cost
    
    properties
        dimension     % State dimension to impose the cost on
        threshold     % Value above which to impose the quadratic cost
        orientedRight % Determines which side of threshold to penalize
    end
    
    methods
        function obj = SemiquadraticCost(...
                dimension, threshold, orientedRight)
            obj.dimension = dimension;
            obj.threshold = threshold;
            obj.orientedRight = orientedRight;
        end
        
        function cost = getCost(obj, ~, x)
            %GETCOST Computes the semiquadratic cost
            cost = 0;
            
            if obj.orientedRight
                if x{obj.dimension} > obj.threshold
                    cost = (x{obj.dimension} - obj.threshold).^2;
                end
            else
                if x{obj.dimension} < obj.threshold
                    cost = (x{obj.dimension} - obj.threshold).^2;
                end
            end
        end
    end
end


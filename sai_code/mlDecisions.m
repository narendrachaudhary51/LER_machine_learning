function [x] = mlDecisions(const,y)

minimum = 10000;
constellationIndex = -1;

for i = 1:length(const)
    currentDistance = abs(const(i) - y);
    if(currentDistance < minimum)
        minimum = currentDistance;
        constellationIndex = i;
    end
end
x = const(constellationIndex);
end
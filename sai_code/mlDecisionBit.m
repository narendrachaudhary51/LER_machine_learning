function [x] = mlDecisionBit(const,y)

minimum = 10000;
constellationIndex = -1;

for i = 1:length(const)
    currentDistanceSquared = 0;
    for j = 1:length(const(1,:))
        currentDistanceSquared = currentDistanceSquared + (const(i,j) - y(1,j))^2;
    end
    if(currentDistanceSquared < minimum)
        minimum = currentDistanceSquared;
        constellationIndex = i;
    end
end
x = const(constellationIndex,:);
end
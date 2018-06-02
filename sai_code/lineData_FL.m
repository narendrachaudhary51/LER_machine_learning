function [line1,line2] =lineData_FL(BWs)
line1 = zeros(size(BWs,1),1);
line2 = zeros(size(BWs,1),1);
for i = 1:size(BWs,1)
    
    
    x= find(BWs(i,:),1,'first');
    y= find(BWs(i,:),1,'last');
    if (size(x) ~= 0 )
        line1(i) = x;
    end
    if (size(y) ~= 0)
        line2(i) = y;
    end
    
end

end

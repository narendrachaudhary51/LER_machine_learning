function [line1,line2] =lineData(BWs)
line1 = zeros(size(BWs,1),1);
line2 = zeros(size(BWs,1),1);
for i = 1:size(BWs,1)
    [~,idx]  = find(BWs(i,:));
    %if (size(idx) > 2)
    
    if (size(idx) < 2)
        idx = [0,0];
    else
        [p,q] = kmeans(idx',2);
    
        if(q(1) > q(2))
            line2(i) = floor(q(1)); 
            line1(i) = floor(q(2));
        else
            line1(i) = floor(q(1)); 
            line2(i) = floor(q(2));
        end
    end
    
end

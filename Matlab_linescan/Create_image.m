
function [I line2] = Creat_image(edge, span,M,offset,width,space)
 I = zeros(M,M);
 Pixelwidth = span/M;
 
 for i=1:size(edge,1)
     edge(i,:) = edge(i,:) + offset*Pixelwidth;
     if(rem(i,2) == 0)
         edge(i,:) = edge(i,:) + (i/2)*width*Pixelwidth + (i/2 - 1)*space*Pixelwidth;
     else
         edge(i,:) = edge(i,:) + ((i-1)/2)*width*Pixelwidth + ((i-1)/2)*space*Pixelwidth;
     end
 end
 
 line = edge/Pixelwidth;
 %line = round(line);
 for k = 1:2:size(edge,1)
    for i = 1:M
         for j=1:M
            %if((j*Pixelwidth >= edge(k,i)) && (j*Pixelwidth <= edge(k+1,i)))
            if((j >= line(k,i)) && (j <= line(k+1,i)))
                I(i,j) = 100;
            end    
         end
    end
 end
 
 %figure
 %plot(edge')

for k=1:size(edge,1)
    for i =1:M
        line1(k,i,1) = i;
        line1(k,i,2) = line(k,i);
    end
end


for k=1:2:size(edge,1)
    line2(floor(k/2) + 1,:,1) = [line1(k,:,1) fliplr(line1(k+1,:,1))];
    line2(floor(k/2) + 1,:,2) = [line1(k,:,2) fliplr(line1(k+1,:,2))];
end

 
end

d = 'C:\courses\DirectedStudies\Datasets\noisy_images\';

s = dir('C:\courses\DirectedStudies\Datasets\noisy_images\noisy_images\*.tiff');
I = zeros(1024,64,3);


for i = 1 : 30
    filename = strcat('C:\courses\DirectedStudies\Datasets\noisy_images\noisy_images\',s(i).name);
    I(:,:,i) = imread(filename);
  
end
%%
s = dir('C:\courses\DirectedStudies\Datasets\original_images\original_images\*.tiff');

orig_img = zeros(1024,64,3);

for i = 1 : 3
    filename = strcat('C:\courses\DirectedStudies\Datasets\original_images\original_images\',s(i).name);
    orig_img(:,:,i) = imread(filename);
     figure,colormap(gray);
    imagesc(orig_img(:,:,i));
end

%%

% compute SNR
snr = 20*log(norm(orig_img(:,:,1),'fro')/norm(orig_img(:,:,1)-I(:,:,3),'fro')); % check the formula for the SNR

% gaussian filter
%%

filtered_images = zeros(size(I));

for i = 1:size(I,3)
    %filtered_images(:,:,i) = medfilt2(I(:,:,i));
    %filtered_images(:,:,i) = imgaussfilt(I(:,:,i),1);
    filtered_images(:,:,i) = imgaussfilt(I(:,:,i),[3,1]);
    
end

for i = 1:10
      %figure,imshowpair(filtered_images(:,:,i),I(:,:,i),'montage');
      
      figure,colormap(gray);
      imagesc(filtered_images(:,:,i));
end
%%

snr_after_filter = 20*log(norm(orig_img(:,:,1),'fro')/norm(orig_img(:,:,1)-filtered_images(:,:,3),'fro')); % check the formula for the SNR


%%
edge_detected_images = zeros(size(I));

% edge filter application

for i = 1:10
%[~, threshold] = edge(filtered_images(:,:,i), 'canny');
%fudgeFactor = 1;
edge_detected_images(:,:,i) = edge(I(:,:,i),'canny',[],7);

figure,colormap(gray);
imagesc(edge_detected_images(:,:,i));

   
    % edge_detected_images(:,:,i) = (I(:,:,i),0.5)   
end
%%

% computer ler


%%
%[~, threshold] = edge(orig_img(:,:,1), 'canny',[],3);
%fudgeFactor = 2;
BW_orig = edge(orig_img(:,:,1),'canny',[],5);
%figure, imshowpair(BWs,filtered_images(:,:,1),'montage');
figure,colormap(gray)
imagesc(BW_orig);
% %%
% surf(BWs(:,:));
% 
%%
%step 1 : find ones in each row
 
% [rows, columns] = find(BWs(:,:));
% data_clip = zeros(size(rows,1),2);
% %%
% data_clip = [rows,columns];
% [~,idx] = sort(data_clip(:,1)); % sort just the first column
% sortedmat = data_clip(idx,:);


%%
line_left_orig = zeros(1024,1);
line_right_orig = zeros(1024,1);
for i = 1:1024
    [~,idx]  = find(BWs(i,:));
    %if (size(idx) > 2)
    
    if (size(idx) < 2)
        idx = [10,30];
    else
        [p,q] = kmeans(idx',2);
    
        if(q(1) > q(2))
            line_right_orig(i) = floor(q(1)); 
            line_left_orig(i) = floor(q(2));
        else
            line_left_orig(i) = floor(q(1)); 
            line_right_orig(i) = floor(q(2));
        end
    end
    
end
%%
lineData_images_left = zeros(1024,10);
lineData_images_right = zeros(1024,10);
%%
for i = 1:10
    [lineData_images_left(:,i),lineData_images_right(:,i)] = lineData(edge_detected_images(:,:,i));
end

%%
% predicted image line data
x = load('C:\courses\DirectedStudies\Datasets\predicted_images\predicted.mat');
predicted_Img = round(x.arr*256); 
imagesc(predicted_Img);

%%
% predicted image edge detection

BW_Img = edge(predicted_Img,'canny',[],4);
%figure, imshowpair(BWs,filtered_images(:,:,1),'montage');
figure,colormap(gray)
imagesc(BW_Img);

[line_predict_left,line_predict_right] = lineData(BW_Img);




%%    
i=1:10
std_noisy_images_left =std(lineData_images_left(256:768,i))   
std_noisy_images_right = std(lineData_images_right(256:768,i))
std_orig_image_left = std(line_left_orig(256:768))
std_orig_image_right = std(line_right_orig(256:768))
std_predict_image_left = std(line_predict_left(256:768))
std_predict_image_right = std(line_predict_right(256:768))


%%


    
    

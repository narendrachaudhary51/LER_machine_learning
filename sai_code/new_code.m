d = 'C:\courses\DirectedStudies\Datasets\noisy_images\';

s = dir('C:\courses\DirectedStudies\Datasets\noisy_images\noisy_images\*.tiff');
I = zeros(1024,64,3);


% s(1).name = 'nim_1.2e-09_0.1_1.1e-08_20_20_19_10.tiff';
% s(2).name = 'nim_1.2e-09_0.1_1.1e-08_20_20_19_100.tiff';
% s(3).name = 'nim_1.2e-09_0.1_1.1e-08_20_20_19_10.tiff';
% s(4).name = 'nim_1.2e-09_0.1_1.1e-08_20_20_19_10.tiff';
% s(5).name = 'nim_1.2e-09_0.1_1.1e-08_20_20_19_10.tiff';
% s(6).name = 'nim_1.2e-09_0.1_1.1e-08_20_20_19_10.tiff';
% s(7).name = 'nim_1.2e-09_0.1_1.1e-08_20_20_19_10.tiff';
% s(8).name = 'nim_1.2e-09_0.1_1.1e-08_20_20_19_10.tiff';
% s(9).name = 'nim_1.2e-09_0.1_1.1e-08_20_20_19_10.tiff';
% s(10).name = 'nim_1.2e-09_0.1_1.1e-08_20_20_19_10.tiff';


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
edge_detected_images(:,:,i) = edge(I(:,:,i),'canny',[],4);

figure,colormap(gray);
imagesc(edge_detected_images(:,:,i));

   
    % edge_detected_images(:,:,i) = (I(:,:,i),0.5)   
end
%%

% computer ler


%%
%[~, threshold] = edge(orig_img(:,:,1), 'canny',[],3);
%fudgeFactor = 2;
BW_orig = edge(orig_img(:,:,1),'canny',[],1);
%figure, imshowpair(BWs,filtered_images(:,:,1),'montage');
figure,colormap(gray)
imagesc(BW_orig);
% %%
% surf(BWs(:,:));
% 

%%
lineData_images_left = zeros(1024,10);
lineData_images_right = zeros(1024,10);
%%
for i = 1:10
    [lineData_images_left(:,i),lineData_images_right(:,i)] = lineData_FL(edge_detected_images(:,:,i));
end

%%
% predicted image line data
x = load('C:\courses\DirectedStudies\Datasets\predicted_images\predicted1.mat');
predicted_Img = round(x.arr*256); 
imagesc(predicted_Img);

%%
% predicted image edge detection

BW_Img = edge(predicted_Img,'canny',[],1);
%figure, imshowpair(BWs,filtered_images(:,:,1),'montage');
figure,colormap(gray)
imagesc(BW_Img);

[line_predict_left,line_predict_right] = lineData_FL(BW_Img);
[line_left_orig,line_right_orig] = lineData_FL(BW_orig);




%%    
i=1:10
std_noisy_images_left =std(lineData_images_left(257:768,i))/2   
std_noisy_images_right = std(lineData_images_right(257:768,i))/2
std_orig_image_left = std(line_left_orig(257:768))/2
std_orig_image_right = std(line_right_orig(257:768))/2
std_predict_image_left = std(line_predict_left(257:768))/2
std_predict_image_right = std(line_predict_right(257:768))/2


%%


x = 1:1024;
plot(x,line_left_orig,x,line_predict_left);
title('Graph of predicted vs original for image oim_1.2e-09_0.1_1.1e-08_20_20_19.tiff');
legend('Original','Predicted')





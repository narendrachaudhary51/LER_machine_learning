
s(1).name = 'nim_8e-10_0.3_1e-08_20_20_22_10.tiff';
s(2).name = 'nim_1.2e-09_0.7_4e-08_30_30_17_2.tiff';
s(3).name = 'nim_1.6e-09_0.5_3e-08_20_40_14_100.tiff';



for i = 1 : 3
    filename = strcat('H:\MATLAB\noisy_images\images_voal\',s(i).name);
    Test_noisy(:,:,i) = imread(filename);
end
%%
%s = dir('C:\courses\DirectedStudies\Datasets\original_images\original_images\*.tiff');
s1(1).name = 'oim_8e-10_0.3_1e-08_20_20_22.tiff';
s1(2).name = 'oim_1.2e-09_0.7_4e-08_30_30_17.tiff';
s1(3).name = 'oim_1.6e-09_0.5_3e-08_20_40_14.tiff';

test = zeros(1024,64,3);

for i = 1 : 3
    filename = strcat('H:\MATLAB\noisy_images\images_voal\',s1(i).name);
    test(:,:,i) = imread(filename);
     figure,colormap(gray);
    imagesc(test(:,:,i));
end

% gaussian filter
%%

gaussian_filtered_images = zeros(size(Test_noisy));
filtered_images_1 = zeros(size(Test_noisy));
filtered_images_2 = zeros(size(Test_noisy));

for i = 2:size(Test_noisy,3)
    %filtered_images(:,:,i) = medfilt2(I(:,:,i));
    %filtered_images(:,:,i) = imgaussfilt(I(:,:,i),1);
    gaussian_filtered_images(:,:,i) = imgaussfilt(Test_noisy(:,:,i),[3,1]);
    
end

for i = 2:size(Test_noisy,3)
      %figure,imshowpair(filtered_images(:,:,i),I(:,:,i),'montage');
      
      figure,colormap(gray);
      imagesc(gaussian_filtered_images(:,:,i));
end



%%
edge_detected_images = zeros(size(Test_noisy));

% edge filter application

for i = 1:3
%[~, threshold] = edge(filtered_images(:,:,i), 'canny');
%fudgeFactor = 1;
edge_detected_images(:,:,i) = edge(Test_noisy(:,:,i),'canny',[],4);

figure,colormap(gray);
imagesc(edge_detected_images(:,:,i));

   
    % edge_detected_images(:,:,i) = (I(:,:,i),0.5)   
end
%%

% computer ler


%%
%[~, threshold] = edge(orig_img(:,:,1), 'canny',[],3);
%fudgeFactor = 2;
BW_orig = edge(Test(:,:,2),'canny',[],1);
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
x = load('H:\MATLAB\noisy_images\images_voal\predicted1.mat');
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
std_noisy_images_left =std(lineData_images_left(256:768,i))   
std_noisy_images_right = std(lineData_images_right(256:768,i))
std_orig_image_left = std(line_left_orig(256:768))
std_orig_image_right = std(line_right_orig(256:768))
std_predict_image_left = std(line_predict_left(256:768))
std_predict_image_right = std(line_predict_right(256:768))


%%


x = 1:1024;
plot(x,line_left_orig,x,line_predict_left);
title('Graph of predicted vs original for image oim_1.2e-09_0.1_1.1e-08_20_20_19.tiff');
legend('Original','Predicted')


%%
filename1 = 'C:\courses\DirectedStudies\Datasets\linescans\linescans\linescan_1.2e-09_0.1_1.1e-08_20_20.txt';
M = csvread(filename1);

pr = M(257:768,2);





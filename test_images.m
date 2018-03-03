
s(1).name = 'nim_8e-10_0.3_1e-08_20_20_22_10.tiff';
s(2).name = 'nim_1.2e-09_0.7_4e-08_30_30_17_2.tiff';
s(3).name = 'nim_1.6e-09_0.5_3e-08_20_40_14_100.tiff';

range_factor = 257:768;
%range_factor = 2:1022;



for i = 1 : 3
    filename = strcat('C:\courses\DirectedStudies\Datasets\noisy_images\noisy_images\',s(i).name);
    Test_noisy(:,:,i) = imread(filename);
end
%%
%s = dir('C:\courses\DirectedStudies\Datasets\original_images\original_images\*.tiff');
s1(1).name = 'oim_8e-10_0.3_1e-08_20_20_22.tiff';
s1(2).name = 'oim_1.2e-09_0.7_4e-08_30_30_17.tiff';
s1(3).name = 'oim_1.6e-09_0.5_3e-08_20_40_14.tiff';

test = zeros(1024,64,3);

for i = 1 : 3
    filename = strcat('C:\courses\DirectedStudies\Datasets\original_images\original_images\',s1(i).name);
    test(:,:,i) = imread(filename);
     figure,colormap(gray);
    imagesc(test(:,:,i));
end

% gaussian filter
%%

gaussian_filtered_images = zeros(size(Test_noisy));

for i = 1:size(Test_noisy,3)
    gaussian_filtered_images(:,:,i) = imgaussfilt(Test_noisy(:,:,i),[3,1]);
  
end




%%
original_edge_detected_images = zeros(size(test));
original_lineData_images_left = zeros(1024,3);
original_lineData_images_right = zeros(1024,3);

for i = 1:size(test,3) 
%[~, threshold] = edge(filtered_images(:,:,i), 'canny');
%fudgeFactor = 1;
original_edge_detected_images(:,:,i) = edge(test(:,:,i),'canny',[],1);
[original_lineData_images_left(:,i),original_lineData_images_right(:,i)] = lineData_FL(original_edge_detected_images(:,:,i));


    % edge_detected_images(:,:,i) = (I(:,:,i),0.5)   
end

colormap(gray),imagesc(original_edge_detected_images(:,:,1));
colormap(gray),imagesc(test(:,:,2));

%%
gaussian_edge_detected_images = zeros(size(Test_noisy));
noisy_edge_detected_images = zeros(size(Test_noisy));
% edge filter application

%for i = 1:1
%[~, threshold] = edge(filtered_images(:,:,i), 'canny');
%fudgeFactor = 1;
gaussian_edge_detected_images(:,:,1) = edge(gaussian_filtered_images(:,:,1),'canny',[],4);
gaussian_edge_detected_images(:,:,2) = edge(gaussian_filtered_images(:,:,2),'canny',[],4);
gaussian_edge_detected_images(:,:,3) = edge(gaussian_filtered_images(:,:,3),'canny',[],4);

noisy_edge_detected_images(:,:,1) = edge(Test_noisy(:,:,1),'canny',[],4);
noisy_edge_detected_images(:,:,2) = edge(Test_noisy(:,:,2),'canny',[],4);
noisy_edge_detected_images(:,:,3) = edge(Test_noisy(:,:,3),'canny',[],4);
%% View edge detected images for gaussian denoiser
figure,colormap(gray);
imagesc(gaussian_edge_detected_images(:,:,1));
   
    % edge_detected_images(:,:,i) = (I(:,:,i),0.5)   
%end
%% View edge detected images for noisy images


figure,colormap(gray);
imagesc(noisy_edge_detected_images(:,:,1));
   
%%
gaussian_lineData_images_left = zeros(1024,3);
gaussian_lineData_images_right = zeros(1024,3);
noisy_lineData_images_left = zeros(1024,3);
noisy_lineData_images_right = zeros(1024,3);

%%
for i = 1:size(Test_noisy,3)
    [gaussian_lineData_images_left(:,i),gaussian_lineData_images_right(:,i)] = lineData_FL(gaussian_edge_detected_images(:,:,i));
    [noisy_lineData_images_left(:,i),noisy_lineData_images_right(:,i)] = lineData_FL(noisy_edge_detected_images(:,:,i));
end





%%    
i=1:3;
std_noisy_images_left_gaussian =std(gaussian_lineData_images_left(256:768,i))/2;   
std_noisy_images_right_gaussian = std(gaussian_lineData_images_right(256:768,i))/2;



%%
filename1 = 'C:\courses\DirectedStudies\Datasets\linescans\linescans\linescan_8e-10_0.3_1e-08_20_20.txt';
filename2 = 'C:\courses\DirectedStudies\Datasets\linescans\linescans\linescan_1.2e-09_0.7_4e-08_30_30.txt';
filename3 = 'C:\courses\DirectedStudies\Datasets\linescans\linescans\linescan_1.6e-09_0.5_3e-08_20_40.txt';
%%
M1 = csvread(filename1);
l_1 = M1(1:1024,2);
l_1_int = int16(l_1);
ler1 = std(double(l_1_int(257:768)))/2;
r_1 = M1(1024+1:1024+1024,2);
r_1_int = int16(fliplr(r_1));
%%

M2 = csvread(filename2);
l_2 = M2(1:1024,2);
l_2_int = int16(l_2);
ler2 = std(double(l_2_int(257:768)))/2;
r_2 = M2(1024+1:1024+1024,2);
r_2_int = int16(fliplr(r_2));

%%
M3 = csvread(filename3);
l_3 = M3(1:1024,2);
l_3_int = int16(l_3);
ler3 = std(double(l_3_int(257:768)))/2;
r_3 = M3(1024+1:1024+1024,2);
r_3_int = int16(fliplr(r_3));

%%
mu = 10;


ATV_filtered_images = zeros(size(Test_noisy));
ITV_filtered_images = zeros(size(Test_noisy));
Test_noisy = double(Test_noisy);

for i = 1:size(Test_noisy,3)
    temp = (Test_noisy(:,:,i));
    temp_2 = SB_ATV(temp(:),mu);
    ATV_filtered_images(:,:,i) = reshape(temp_2,[size(Test_noisy,1),size(Test_noisy,2)]);
    temp_2 = SB_ATV(temp(:),mu);
    ITV_filtered_images(:,:,i) = reshape(temp_2,[size(Test_noisy,1),size(Test_noisy,2)]);
end
%%

ATV_edge_detected_images = zeros(size(Test_noisy));
ITV_edge_detected_images = zeros(size(Test_noisy));

% edge filter application

%for i = 1:size(Test_noisy,3)

ATV_edge_detected_images(:,:,1) = edge(ATV_filtered_images(:,:,1),'canny',[],4);
ATV_edge_detected_images(:,:,2) = edge(ATV_filtered_images(:,:,2),'canny',[],4);
ATV_edge_detected_images(:,:,3) = edge(ATV_filtered_images(:,:,3),'canny',[],4);

ITV_edge_detected_images(:,:,1) = edge(ITV_filtered_images(:,:,1),'canny',[],4);
ITV_edge_detected_images(:,:,2) = edge(ITV_filtered_images(:,:,2),'canny',[],4);
ITV_edge_detected_images(:,:,3) = edge(ITV_filtered_images(:,:,3),'canny',[],4);


%% View edge detected images for TV denoiser


figure,colormap(gray);
imagesc(ATV_edge_detected_images(:,:,1));


%end
%%
ATV_lineData_images_left = zeros(1024,size(Test_noisy,3));
ATV_lineData_images_right = zeros(1024,size(Test_noisy,3));
ITV_lineData_images_left = zeros(1024,size(Test_noisy,3));
ITV_lineData_images_right = zeros(1024,size(Test_noisy,3));




%%
for i = 1:size(Test_noisy,3)
    [ATV_lineData_images_left(:,i),ATV_lineData_images_right(:,i)] = lineData_FL(ATV_edge_detected_images(:,:,i));
    [ITV_lineData_images_left(:,i),ITV_lineData_images_right(:,i)] = lineData_FL(ITV_edge_detected_images(:,:,i));
end
%%

i=1:3;


LER_data_left_ATV = std(ATV_lineData_images_left(range_factor,i))/2;
LER_data_right_ATV = std(ATV_lineData_images_right(range_factor,i))/2;
LER_data_left_ITV = std(ITV_lineData_images_left(range_factor,i))/2;
LER_data_right_ITV = std(ITV_lineData_images_right(range_factor,i))/2;


LER_data_left_original = std(original_lineData_images_left(range_factor,i))/2;
LER_data_right_original = std(original_lineData_images_right(range_factor,i))/2;

LER_data_left_noisy = std(noisy_lineData_images_left(range_factor,i))/2;
LER_data_right_noisy = std(noisy_lineData_images_right(range_factor,i))/2;



%%
% predicted image line data
Predicted_images_int = zeros(size(Test_noisy));
Predicted_edge_images = zeros(size(Test_noisy));
lineData_images_left_predicted = zeros(1024,size(Test_noisy,3));
lineData_images_right_predicted = zeros(1024,size(Test_noisy,3));

x = load('C:\courses\DirectedStudies\Datasets\predicted_images\predicted_1stTest.mat');
Predicted_images_int(:,:,1) = x.arr;
x = load('C:\courses\DirectedStudies\Datasets\predicted_images\predicted_2rdTest.mat');
Predicted_images_int(:,:,2) = x.arr;
x = load('C:\courses\DirectedStudies\Datasets\predicted_images\predicted_3rdTest.mat');
Predicted_images_int(:,:,3) = x.arr;

%%
for i = 1:size(Test_noisy,3)
    
    Predicted_edge_images(:,:,i) =  edge(Predicted_images_int(:,:,i),'canny',[],1); 
    [lineData_images_left_predicted(:,i),lineData_images_right_predicted(:,i)] = lineData_FL(Predicted_edge_images(:,:,i));
end
i = 1:3;
LER_data_left_predicted = std(lineData_images_left_predicted(range_factor,i))/2;
LER_data_right_predicted = std(lineData_images_right_predicted(range_factor,i))/2;



%% View edge detected images for Predicted Images


figure,colormap(gray);
imagesc(Predicted_edge_images(:,:,1));

%%
% original image in test :
% Noisy Image in Test_noisy
% line data l_1, l_2, l_3
% predicte images Predicted_images
%%
%MSE_NOISE
range_line = 256:767;
range_edge = 257:768;
pixel_range = 256;
Predicted_images_int = uint8(pixel_range*Predicted_images_int);
test = uint8(test);
Test_noisy = uint8(Test_noisy);
i=3
mse_noisy = mean((uint8(test(:,i)) - Test_noisy(:,i)).^2);
mse_predict = mean(((Predicted_images_int(:,i)) - test(:,i)).^2);
%%
%PSNR_NOISY
psnr_noisy = psnr(test(:,i), Test_noisy(:,i));  
psnr_predict = psnr(Predicted_images_int(:,i),test(:,i));

%%
%Predicted_edge_images
%lineData_images_left_predicted,lineData_images_right_predicted
%original_edge_detected_images
%original_lineData_images_left,original_lineData_images_right
%ATV_lineData_images_left,ATV_lineData_images_right
%noisy_lineData_images_left,noisy_lineData_images_right

ledge_line_sigma_1 = std(double(l_1_int(range_line)))/2;
redge_line_sigma_1 = std(double(r_1_int(range_line)))/2;

ledge_line_sigma_2 = std(double(l_2_int(range_line)))/2;
redge_line_sigma_2 = std(double(r_2_int(range_line)))/2;

ledge_line_sigma_3 = std(double(l_3_int(range_line)))/2;
redge_line_sigma_3 = std(double(r_3_int(range_line)))/2;
    

%%
ledge_orig_sigma  = std(original_lineData_images_left(range_edge,i))/2;
ledge_pred_sigma  = std(lineData_images_left_predicted(range_edge,i))/2;
redge_orig_sigma  = std(original_lineData_images_right(range_edge,i))/2;
redge_pred_sigma  = std(lineData_images_right_predicted(range_edge,i))/2;

%%

lorig_rmse = sqrt(mean((l_1_int(range_line) - int16(original_lineData_images_left(range_edge,i))).^2));
rorig_rmse = sqrt(mean((l_1_int(range_line) - int16(original_lineData_images_right(range_edge,i))).^2));

lpred_rmse = sqrt(mean((l_1_int(range_line) - int16(lineData_images_left_predicted(range_edge,i))).^2));
rpred_rmse = sqrt(mean((l_1_int(range_line) - int16(lineData_images_right_predicted(range_edge,i))).^2));

ldiff_rmse = sqrt(mean((original_lineData_images_left(range_edge,i) - (lineData_images_left_predicted(range_edge,i))).^2));
rdiff_rmse = sqrt(mean((original_lineData_images_right(range_edge,i) - (lineData_images_right_predicted(range_edge,i))).^2));
 
%%
plot(original_lineData_images_left(range_edge,2),range_edge,lineData_images_left_predicted(range_edge,2),range_edge);

%%
%TV
%ATV_lineData_images_left,ATV_lineData_images_right
%ATV_filtered_images
i = 3
psnr_TV_nosiy = psnr(test(:,i), uint8(ATV_filtered_images(:,i)));  

ledge_TV_sigma  = std(ATV_lineData_images_left(range_edge,i))/2;
redge_TV_sigma  = std(ATV_lineData_images_right(range_edge,i))/2;

lline_rmse_TV = sqrt(mean((l_1_int(range_line) - int16(ATV_lineData_images_left(range_edge,i))).^2));
rline_rmse_TV = sqrt(mean((l_1_int(range_line) - int16(ATV_lineData_images_right(range_edge,i))).^2));

lorig_rmse_TV = sqrt(mean((ATV_lineData_images_left(range_edge,i)) - (original_lineData_images_left(range_edge,i))).^2);
rorig_rmse_TV = sqrt(mean((ATV_lineData_images_right(range_edge,i)) - (original_lineData_images_right(range_edge,i))).^2);

%%
%gaussian_filtered_images
%gaussian_lineData_images_right,gaussian_lineData_images_left
i = 3
psnr_Gaussian_nosiy = psnr(test(:,i), uint8(gaussian_filtered_images(:,i)));  

ledge_Gaussian_sigma  = std(gaussian_lineData_images_left(range_edge,i))/2;
redge_Gaussian_sigma  = std(gaussian_lineData_images_right(range_edge,i))/2;

lline_rmse_Gaussian = sqrt(mean((l_1_int(range_line) - int16(gaussian_lineData_images_left(range_edge,i))).^2));
rline_rmse_Gaussian = sqrt(mean((l_1_int(range_line) - int16(gaussian_lineData_images_right(range_edge,i))).^2));

lorig_rmse_Gaussian = sqrt(mean((gaussian_lineData_images_left(range_edge,i)) - (original_lineData_images_left(range_edge,i))).^2);
rorig_rmse_Gaussian = sqrt(mean((gaussian_lineData_images_right(range_edge,i)) - (original_lineData_images_right(range_edge,i))).^2);






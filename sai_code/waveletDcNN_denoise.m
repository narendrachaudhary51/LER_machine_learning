
wavelet_filtered_images = zeros(size(Test_noisy));

for i= 1:3
    x = Test_noisy(:,:,i);
    [thr,sorh,keepapp] = ddencmp('den','wv',x);
    wavelet_filtered_images(:,:,i) = wdencmp('gbl',x,'db3',2,thr,sorh,keepapp);
    figure,colormap(gray),imagesc(wavelet_filtered_images(:,:,i));
end
%%

%for i = 2:3
%[~, threshold] = edge(filtered_images(:,:,i), 'canny');
%fudgeFactor = 1;
edge_detected_images_wave(:,:,1) = edge(wavelet_filtered_images(:,:,1),'canny',[],3);
edge_detected_images_wave(:,:,2) = edge(wavelet_filtered_images(:,:,2),'canny',[],5);
edge_detected_images_wave(:,:,3) = edge(wavelet_filtered_images(:,:,3),'canny',[],2);

figure,colormap(gray);
imagesc(edge_detected_images_wave(:,:,1));

figure,colormap(gray);
imagesc(edge_detected_images_wave(:,:,2));

figure,colormap(gray);
imagesc(edge_detected_images_wave(:,:,3));

   
    % edge_detected_images(:,:,i) = (I(:,:,i),0.5)   
%end
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


%%
filename1 = 'H:\MATLAB\noisy_images\images_voal\linescan_8e-10_0.3_1e-08_20_20.txt';
filename2 = 'H:\MATLAB\noisy_images\images_voal\linescan_1.2e-09_0.7_4e-08_30_30.txt';
filename3 = 'H:\MATLAB\noisy_images\images_voal\linescan_1.6e-09_0.5_3e-08_20_40.txt';
%%
shift = [22 17 14];
M1 = csvread(filename1);
l_1 = M1(1:1024,2);
l_1_int = floor(l_1 - shift(1));
ler1 = std(double(l_1_int(257:768)))/2;
r_1 = M1(1024+1:1024+1024,2);
r_1_int = floor(fliplr(r_1) - shift(1));
%%

M2 = csvread(filename2);
l_2 = M2(1:1024,2);
l_2_int = floor(l_2 - shift(2));
ler2 = std(double(l_2_int(257:768)))/2;
r_2 = M2(1024+1:1024+1024,2);
r_2_int = floor(fliplr(r_2) - shift(2));

%%
M3 = csvread(filename3);
l_3 = M3(1:1024,2);
l_3_int = floor(l_3 - shift(3));
ler3 = std(double(l_3_int(257:768)))/2;
r_3 = M3(1024+1:1024+1024,2);
r_3_int = floor(fliplr(r_3)- shift(3));



%%
lineData_images_left_wave = zeros(1024,10);
lineData_images_right_wave = zeros(1024,10);

for i = 1:3
    [lineData_images_left_wave(:,i),lineData_images_right_wave(:,i)] = lineData_FL(edge_detected_images_wave(:,:,i));
end
%%
range_edge = 14:1013;
range_line = 13:1012;
%%
i=1:3;

LER_data_left_wavelet = std(lineData_images_left_wave(range_edge,i))/2;
LER_data_right_wavelet = std(lineData_images_right_wave(range_edge,i))/2;


psnr_wavelet_nosiy = psnr((wavelet_filtered_images(:,:,3)),test(:,:,3))


lline_rmse_wavelet = sqrt(mean((l_1_int(range_line) - (lineData_images_left_wave(range_edge,i))).^2));
rline_rmse_wavelet = sqrt(mean((l_1_int(range_line) - (lineData_images_right_wave(range_edge,i))).^2));

lorig_rmse_wavelet = sqrt(mean((lineData_images_left_wave(range_edge,i)) - (original_lineData_images_left(range_edge,i))).^2);
rorig_rmse_wavelet = sqrt(mean((lineData_images_right_wave(range_edge,i)) - (original_lineData_images_right(range_edge,i))).^2);


%%
% DNN filter
DcNN_filtered_images = zeros(size(Test_noisy));
net = denoisingNetwork('DnCNN');
tic
 for i= 1:3
    x = Test_noisy(:,:,i);
    DcNN_filtered_images(:,:,i) = denoiseImage(x,net);
    figure,colormap(gray),imagesc(DcNN_filtered_images(:,:,i));
end
time_record = toc;
%%

figure,colormap(gray),imagesc(DcNN_filtered_images(:,:,2));


%%



edge_detected_images_DcNN(:,:,1) = edge(DcNN_filtered_images(:,:,1),'canny',[],1);
edge_detected_images_DcNN(:,:,2) = edge(DcNN_filtered_images(:,:,2),'canny',[],2);
edge_detected_images_DcNN(:,:,3) = edge(DcNN_filtered_images(:,:,3),'canny',[],0.5);


figure,colormap(gray);
imagesc(edge_detected_images_DcNN(:,:,1));

figure,colormap(gray);
imagesc(edge_detected_images_DcNN(:,:,2));

figure,colormap(gray);
imagesc(edge_detected_images_DcNN(:,:,3));

%%
lineData_images_left_DcNN = zeros(1024,10);
lineData_images_right_DcNN = zeros(1024,10);

for i = 1:3
    [lineData_images_left_DcNN(:,i),lineData_images_right_DcNN(:,i)] = lineData_FL(edge_detected_images_DcNN(:,:,i));
end


i=1:3;

LER_data_left_DcNN = std(lineData_images_left_DcNN(range_edge,i))/2;
LER_data_right_DcNN = std(lineData_images_right_DcNN(range_edge,i))/2;
%%
image1 = DcNN_filtered_images(:,:,3)/256;
test1 = test(:,:,3)/256;
figure,colormap(gray)
imagesc(image1)
figure, colormap(gray)
imagesc(test1)

psnr(image1, test1)
%%

lline_rmse_DcNN_1 = sqrt(mean((l_1_int(range_line) - (lineData_images_left_DcNN(range_edge,1))).^2));
rline_rmse_DcNN_1 = sqrt(mean((r_1_int(range_line) - (lineData_images_right_DcNN(range_edge,1))).^2));

lline_rmse_DcNN_2 = sqrt(mean((l_2_int(range_line) - (lineData_images_left_DcNN(range_edge,2))).^2));
rline_rmse_DcNN_2 = sqrt(mean((r_2_int(range_line) - (lineData_images_right_DcNN(range_edge,2))).^2));

lline_rmse_DcNN_3 = sqrt(mean((l_3_int(range_line) - (lineData_images_left_DcNN(range_edge,3))).^2));
rline_rmse_DcNN_3 = sqrt(mean((r_3_int(range_line) - (lineData_images_right_DcNN(range_edge,3))).^2));


i = 1:3

lorig_rmse_DcNN = sqrt(mean((lineData_images_left_DcNN(range_edge,i)) - (original_lineData_images_left(range_edge,i))).^2);
rorig_rmse_DcNN = sqrt(mean((lineData_images_right_DcNN(range_edge,i)) - (original_lineData_images_right(range_edge,i))).^2);

clear all;
sigma = 1.5e-9;     % LER
alpha = 0.75;       % roughness (Hurst) exponent
Xi = 25e-9;         % coorelation length
L = 2.048e-6;
N = 1024;

offset = 20;        % offset of lines in pixels
width = 20;         % width of lines in pixels 
space = 80;         % space of lines in pixels

K = 2* int32(N/(width + space)); %number of edges 

span = 512e-9;      % width of entire image in x-direction
PixelLength = L/N;  % pixel length in y-direction

% create frequencies
qx = zeros(N,1);
for k = 0:N -1
 %    qx(k+1)=(2*pi/N)*(k);
     qx(k+1)= (k - N/2);
end

qx = 2*pi*qx/L;

% Palasantzas PSD
num = (sqrt(pi))*(gamma(alpha + 0.5)/gamma(alpha))*(2*Xi*sigma^2);
denom = (1+(qx.*Xi).^2).^(alpha+0.5);
PSD = num./denom;
% figure
% plot(PSD)
% loglog(PSD(N/2+1:N))

dev = sqrt((1/L)*sum(PSD));
%bias = 2*(gamma(alpha + 3/2)/(sqrt(pi)*gamma(alpha + 1)))*(1/N^(2*alpha/(1+2*alpha)));


% Thoros method, creating K different random PSD
for i = 1:K
    for j = 0:(N/2) - 1
        if(j == 0)
            FFT_edge(i,j + N/2 + 1) = sqrt(PSD(j + N/2 + 1))*randn;
        else
            FFT_edge(i,j + N/2 + 1) = sqrt(PSD(j + N/2 + 1))*(randn + 1i*randn)/sqrt(2);
        end
    end
end


for i = 1:K
    for j=2:N/2
        FFT_edge(i,j) = conj(FFT_edge(i,N - j +2,:));   % Make PSD symmetric because edge is real     
    end
end

for i = 1:K
    FFT_edge(i,1) = sqrt(PSD(i,1,:))*randn;
end
% figure
% plot(abs(FFT_edge(5,:)).^2)

% x = zeros(N,1);
% for n = 0:N-1
%     for j = -N/2:1:(N/2 - 1)
%         x(n+1) =  x(n+1) + (1/sqrt(L))*(rPSD_edge(5,j+ (N/2) +1))*exp(1i*2*pi*j*n/N);
%     end
% end
% figure
% plot(real(x))

for i=1:K
   rdeviations(i) =  sqrt((1/L)*sum(abs(FFT_edge(i,:)).^2));
end
rdev = mean(rdeviations);

% take inverse fourier transform
for i = 1:K
    edge(i,:) = ifft(fftshift((1/sqrt(L))*N*FFT_edge(i,:)),'symmetric');
end
rStd = mean(std(edge,0,2));
% figure
% plot(edge(5,:))

% edge = edge(:,N/2 - N/(4*Z) +1 : N/2 + N/(4*Z));
% N = N/(2*Z);
% L = L/(2*Z);

% estimate the correct PSD
for i=1:K
    %PSD_short(i,:) = ((L/N^2)*abs((fftshift(fft(edge(i,:))).^2)));   % reverse process
    PSD_short(i,:) = ((L/N^2)*abs(fftshift(fft(edge(i,:)))).^2);
end

% figure
% plot(PSD_short(5,:))

for i=1:K
   short_deviations(i) =  sqrt((1/L)*sum(PSD_short(i,:)));
end
sdev = mean(short_deviations);

% PSD_short = pmtm(edge','centered','eigen');
% figure
% plot(mean(PSD_short',1))

Mean = mean(mean(edge,2));
sStd = mean(std(edge,0,2));
[dev rdev rStd sdev sStd]

[Im linescan] = Create_image(edge,span,N,offset,width,space);

figure
imshow(Im)
imwrite(Im,'linescan3.tiff')
% linescan = [line1; flipud(line2); line1(1,:)];
dlmwrite('linescan3.txt',reshape(linescan(1,:,:),[2*N 2]));
for i = 2:int32(K/2)
     dlmwrite('linescan3.txt',reshape(linescan(i,:,:),[2*N 2]),'-append');
end

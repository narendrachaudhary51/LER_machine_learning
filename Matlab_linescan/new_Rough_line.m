clear all;

clear all;
sigma = 1.5e-9;     % LER
alpha = 0.75;       % roughness (Hurst) exponent
Xi = 25e-9;         % coorelation length
Z = 4;
L = Z*2e-6;
N = Z*256;

span = 64e-9;      % width of entire image in x-direction
PixelLength = L/N;  % pixel length in y-direction

% create frequencies
qx = zeros(N+1,1);
for k = 0:N
 %    qx(k+1)=(2*pi/N)*(k);
     qx(k+1)= (k - N/2);
end

qx = 2*pi*qx/L;
num = (sqrt(pi))*(gamma(alpha + 0.5)/gamma(alpha))*(2*Xi*sigma^2);
denom = (1+(qx.*Xi).^2).^(alpha+0.5);
PSD = num./denom;
% figure
% plot(PSD)
dev = sqrt((1/L)*sum(PSD));

%y = ifft(fftshift((1/sqrt(L))*N*sqrt(PSD)),'symmetric');
N = N+1
for j = 1:(N+1)/2
    if(j == 1 || j == (N+1)/2)
        rPSD_edge(j) = sqrt(PSD(j))*randn;
    else
        rPSD_edge(j) = sqrt(PSD(j))*(randn + 1i*randn)/sqrt(2);
    end
    rPSD_edge(N - j + 1) = conj(rPSD_edge(j));
end
figure
plot(abs(rPSD_edge))


edge = ifft(fftshift((1/sqrt(L))*N*rPSD_edge),'symmetric');

figure
plot(edge)

edge = edge((N+1)/2 - (N-1)/(4) + 1 : (N+1)/2 + (N-1)/(4));
% figure
% plot(edge)
L = L/Z;
PSD_short = ((L/N^2)*abs((fftshift(fft(edge)).^2)));


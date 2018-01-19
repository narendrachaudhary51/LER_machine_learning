function [I, pLER, rLER, K] = Data_generation(sigma, alpha, Xi, offset, width, space)
    N = 1024;                       % Number of pixels
    L = 2.048e-6;                   % Length in Y-direction
    span = 512e-9;      % width of entire image in x-direction
    K = 2* int32(N/(width + space)); %number of edges 
    PixelLength = L/N;  % pixel length in y-direction
    
    % create frequencies
    qx = zeros(N,1);
    for i = 0:N -1
         qx(i+1)= (i - N/2);
    end
    qx = 2*pi*qx/L;
    
    % Palasantzas PSD
    num = (sqrt(pi))*(gamma(alpha + 0.5)/gamma(alpha))*(2*Xi*sigma^2);
    denom = (1+(qx.*Xi).^2).^(alpha+0.5);
    PSD = num./denom;
    
    pLER = sqrt((1/L)*sum(PSD));
    
    % Thoros method, creating K different random PSD
    FFT_edge = zeros(K,N);
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
    
    % take inverse fourier transform
    edge = zeros(K,N);
    for i = 1:K
        edge(i,:) = ifft(fftshift((1/sqrt(L))*N*FFT_edge(i,:)),'symmetric');
    end
    
    rLER = mean(std(edge,0,2));         %random LER
    [I, linescan] = Create_image(edge,span,N,offset,width,space);
    
    filename = ['C:\Users\narendra\Documents\LER\Aritfical_surface\linescans\linescan_' num2str(sigma) '_' num2str(alpha) '_' num2str(Xi) '_' num2str(width) '_' num2str(space) '.txt']
    
    dlmwrite(filename,reshape(linescan(1,:,:),[2*N 2]));
    for i = 2:int32(K/2)
        dlmwrite(filename,reshape(linescan(i,:,:),[2*N 2]),'-append');
    end
end
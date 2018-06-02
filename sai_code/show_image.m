clear all;
%I1 = imread('C:\Users\narendra\Documents\LER\CNN_course\Experiments\noisy_lines\nline_28_23.tiff');
%I2 = imread('C:\Users\narendra\Documents\LER\CNN_course\Experiments\original_lines\line28.tiff');
%I = imread('rough_curve.tiff');
%surf(I)
count = 0;
%offset = 30;
%A = zeros(10000,9);
for sigma = 4e-10:2e-10:18e-10
    for alpha = 0.1:0.1:0.9
        for Xi = 6e-9: 1e-9: 40e-9
            for width = 20:10:30
                for s = 0:1
                    space = width*2^s;
                    count = count + 1;
                    %[Im, pLER, rLER, K] = Data_generation(sigma, alpha, Xi, offset, width, space);
                    %A(count,:) = [sigma, alpha, Xi, offset, width, space, pLER, rLER, K];
                end
            end
        end
    end
end
%xlswrite('C:\Users\narendra\Documents\LER\Aritfical_surface\linescans\data.xlsx',A)
%[Im, pLER, rLER, K] = Data_generation(0.9e-9, 0.1, 6e-9, 20, 30, 60);
%imshow(Im);
%[pLER, rLER]
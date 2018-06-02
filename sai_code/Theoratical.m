clear all;
close all;
snr_dB = -1:0.01:1;
snr = 10.^(snr_dB);

QPSK_Pe = 2*qfunc(sqrt(snr)) - qfunc(sqrt(snr)).^2;

QAM16_Pe = 3*qfunc(sqrt(snr./5)) - (9/4)*qfunc(sqrt(snr./5)).^2;

PSK8_Pe = 2*qfunc(sqrt(2*snr*sin(pi/8)^2));
figure
plot(snr,PSK8_Pe)
xlabel('SNR')
ylabel('Probability of Error')
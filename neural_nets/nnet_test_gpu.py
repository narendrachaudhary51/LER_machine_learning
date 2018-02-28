import numpy as np
import matplotlib as plt
from keras.models import Sequential, load_model
from PIL import Image
import timeit,time
import math
import pandas as pd

from skimage import measure
from scipy import ndimage as ndi
from skimage import feature

# load the model
path = '/scratch/user/narendra5/LER_machine_learning/'
model = load_model(path + 'models/' + 'full_run1_epoch_4.h5')
model.summary()

sigmas = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
Xis = [10, 30, 40]
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
widths = [20, 30]
noises = [2, 3, 4, 5, 10, 20, 30, 50, 100, 200]

testsize = len(sigmas)*len(Xis)*len(alphas)*len(widths)*len(noises)*2
print('Testsize: ', testsize)
#X_test = np.zeros([testsize, 1024, 64])
#y_test = np.zeros([testsize, 1024, 64])

df = pd.DataFrame(columns = ['noise', 'sigma', 'alpha', 'Xi', 'width', 'space', 'MSE_noise','MSE_pred', \
                             'PSNR_noise','PSNR_pred', 'Pred_time', 'O_leftline_sigma', 'i_leftline_sigma', \
                             'O_rightline_sigma', 'i_rightline_sigma', 'ledge_orig_sigma', 'ledge_pred_sigma', \
                             'redge_orig_sigma', 'redge_pred_sigma', 'lorig_rmse', 'lpred_rmse', 'rorig_rmse', \
                             'rpred_rmse', 'ldiff_rmse','rdiff_rmse'])

count = 0
for sigma in sigmas:
    for alpha in alphas:
        for Xi in Xis:
            for width in widths:
                for s in range(2):
                    for noise in noises:
                        space = math.floor(width*2**s)
                        shift = math.floor(-25 + (width + space/2 + Xi + alpha*10 + sigma*10)%16) 
                        
                        original_file = path + 'original_images/oim_' + "{:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '.tiff'
                        noisy_file = path + 'noisy_images/nim_' + "{0:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '_' + str(noise) + '.tiff'
                        linescan_file = path + 'linescans/linescan_' + "{:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '.txt'
                        linescan = []
                        with open(linescan_file,'r') as f:
                            for line in f:
                                a, b = line.split(',')
                                linescan.append(float(b))
                        
                        linescan = linescan[:2048]
                        
                        leftline = np.array(linescan[:1024]) 
                        rightline = linescan[1024:]
                        rightline.reverse()
                        rightline = np.array(rightline)

                        leftline = leftline + shift
                        rightline = rightline + shift
                        
                        
                        im = np.array(Image.open(original_file))
                        imnoisy = np.array(Image.open(noisy_file))
                        
                        im = im/256
                        imnoisy = imnoisy/256
                        imnoisy = imnoisy.reshape(1,1024,64,1)
                        im = im.reshape(1,1024,64,1)
                        
                        #test_score = model.evaluate(imnoisy, im)
                        start = time.time()
                        impredict = model.predict(imnoisy)
                        prediction_time = time.time() - start
                        
                        
                        impredict = impredict.reshape(1024,64)
                        imnoisy = imnoisy.reshape(1024,64)
                        im = im.reshape(1024,64)
                        impredict = impredict.astype(float)
                        imnoisy = imnoisy.astype(float)
                        im = im.astype(float)
                        
                        mse_predict = (256**2)*((im - impredict) ** 2).mean()
                        mse_noisy = (256**2)*((im - imnoisy) ** 2).mean()
                        
                        psnr_noisy = measure.compare_psnr(im,imnoisy)
                        psnr_predict = measure.compare_psnr(im,impredict)
                        
                        canny_sigma = 1
                        edges_im = feature.canny(im, sigma = canny_sigma)
                        edges_imnoisy = feature.canny(imnoisy, sigma = canny_sigma)
                        edges_impredict = feature.canny(impredict, sigma = canny_sigma)
                        
                        ledge_orig = np.argmax(edges_im, axis = 1)[256:768]
                        redge_orig = np.argmax(np.fliplr(edges_im), axis = 1)[256:768]
                        redge_orig = 62 - redge_orig
                        
                        ledge_pred = np.argmax(edges_impredict, axis = 1)[256:768]
                        redge_pred = np.argmax(np.fliplr(edges_impredict), axis = 1)[256:768]
                        redge_pred = 62 - redge_pred
                        
                        lorig_rmse = np.sqrt(((leftline[255:767].astype(int) - ledge_orig)**2).mean())
                        rorig_rmse = np.sqrt(((rightline[255:767].astype(int) - redge_orig)**2).mean())
                        
                        lpred_rmse = np.sqrt(((leftline[255:767].astype(int) - ledge_pred)**2).mean())
                        rpred_rmse = np.sqrt(((rightline[255:767].astype(int) - redge_pred)**2).mean())
                        
                        ldiff_rmse = np.sqrt(((ledge_orig - ledge_pred)**2).mean())
                        rdiff_rmse = np.sqrt(((redge_orig - redge_pred)**2).mean())
                        
                        df.loc[count] = [noise, sigma, alpha, Xi, width, space, mse_noisy, mse_predict, \
                                         psnr_noisy, psnr_predict, prediction_time, leftline[255:767].std()/2, \
                                         leftline[255:767].astype(int).std()/2, rightline[255:767].std()/2, \
                                         rightline[255:767].astype(int).std()/2, ledge_orig.std()/2, \
                                         ledge_pred.std()/2, redge_orig.std()/2, redge_pred.std()/2, lorig_rmse, \
                                         lpred_rmse, rorig_rmse, rpred_rmse, ldiff_rmse, rdiff_rmse]
                        
                        count += 1
                        
print("Test count: ", count)

df.to_csv(path + 'Test_results_gpu.csv')


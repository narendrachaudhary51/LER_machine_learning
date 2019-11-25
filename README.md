# LER_machine_learning
Line edge roughness estimation with deep learning
This code has been written by Narendra Chaudhary.
All rights reserved.

The single line original image dataset can be downloaded from this link - https://drive.google.com/a/tamu.edu/file/d/13_u6IpFfprnCmfy82vYsGRov6YTbdl56/view?usp=sharing

The single line noisy image dataset can be downloaded from this link - 
https://drive.google.com/a/tamu.edu/file/d/1DTbKKd9GSLHMbx_3IxiBzs7LWgT7IusZ/view?usp=sharing

The linescan dataset can be downloaded from this link - 
https://drive.google.com/a/tamu.edu/file/d/11LcLFm-cmUwHwLG1HC9Ie0l2vckrsEuc/view?usp=sharing

The linescans folder has multiple text files that contain the X and Y positions of line edges.
The images generated from ARTIMAGEN without any noise are in the original_images2 folder. 
The images generated from ARTIMAGEN with Poisson noise are in the noisy_images2 folder. 
The images in both original_image2 and noisy_images2 folder are of the TIFF data format type. 

Every file has a naming convention. 

linescan file name = linescan_sigma_alpha_Xi_width_space.txt
original image name = oim_sigma_alpha_Xi_width_space_shift.tiff
noisy image name = nim_sigma_alpha_Xi_width_space_shift_noiselevel.tiff 

Here shift = - math.floor(-25 + (width + space/2 + Xi + alpha*10 + sigma*10)%16) 

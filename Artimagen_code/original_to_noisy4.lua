
count = 0
math.randomseed(os.time()) -- Initialization of the random number generator with current time as seed
path = '/home/grads/n/narendra5/Desktop/Programs/LER_machine_learning/'
--do
--s = 1
   for sigma = 0.4, 1.6, 0.2 do
      for alpha = 0.1, 0.9, 0.1 do
         for Xi = 7, 39, 2 do
            for width = 20, 30, 10 do
               for s = 0,1,1 do
                  for n = 1,10,1 do
                     if     n == 1  then noise = 2
		     elseif n == 2  then noise = 3 
                     elseif n == 3  then noise = 4
		     elseif n == 4  then noise = 5
                     elseif n == 5  then noise = 10
                     elseif n == 6  then noise = 20
                     elseif n == 7  then noise = 30
                     elseif n == 8  then noise = 50
                     elseif n == 9  then noise = 100
                     elseif n == 10 then noise = 200 
                     end
                      
                     space = math.floor(width*2^s)
		     shift = math.floor(-25 + (width + space/2 + Xi + alpha*10 + sigma*10)%16)
                     local input_file = path .. 'original_images4/oim_' .. tostring(sigma*1e-09) .. '_' .. tostring(alpha) .. '_' .. tostring(Xi*1e-09) .. '_' .. tostring(width) .. '_' .. tostring(space) ..'_'..tostring(-shift)..'.tiff'
	             
                     im=aig_load_image(input_file);

		     aig_apply_noise(im, "poisson", noise) -- application of Poisson noise
		     --aig_apply_noise(im, "gaussian", 0.6)

                     local output_file = path .. 'noisy_images4/nim_' .. tostring(sigma*1e-09) .. '_' .. tostring(alpha) .. '_' .. tostring(Xi*1e-09) .. '_' .. tostring(width) .. '_' .. tostring(space) ..'_'..tostring(-shift).. '_' ..tostring(noise)..'.tiff'
                     local description = 'Image with'..'sigma='..tostring(sigma*1e-09)..' alpha='..tostring(alpha)..' coorelation length='..tostring(Xi*1e-09)..' width='..tostring(width).. ' space='..tostring(space)..' shift='..tostring(shift).. ' noise_level='..tostring(noise)
		  
                     aig_save_image(im, output_file, description) --saving of the image to a file
		     aig_delete_image(im) -- deletion of the image
                     
                     count = count + 1
                     --if count > 10 then return end
                   end
		end
            end
         end
      end
   end
--end
print (count)

-- This is a ARTIMAGEN Lua script that generate the ARTIMAGEN logo SEM image
-- by Petr Cizmar @ NIST

--[[
    As this software was developed as part of work done by the United States
    Government, it is not subject to copyright, and is in the public domain.
    Note that according to GNU.org public domain is compatible with GPL.
]]--

function file_exists(file)
  local f = io.open(file, "rb")
  if f then f:close() end
  return f ~= nil
end

-- get all lines from a file, returns an empty 
-- list/table if the file does not exist
function lines_from(file)
  if not file_exists(file) then return {} end
  lines = {}
  for line in io.lines(file) do 
    lines[#lines + 1] = line
  end
  return lines
end

math.randomseed(os.time()) -- Initialization of the random number generator with current time as seed
N = 1024

count = 0
do
   for sigma = 0.4, 1.8, 0.2 do
      for alpha = 0.1, 0.9, 0.1 do
         for Xi = 6, 40, 1 do
            for width = 20, 30, 10 do
	       for s = 0,1,1 do
                  space = math.floor(width*2^s)
                  count = count + 1
	          local file = 'linescans/linescan_' .. tostring(sigma*1e-09) .. '_' .. tostring(alpha) .. '_' .. tostring(Xi*1e-09) .. '_' .. tostring(width) .. '_' .. tostring(space) .. '.txt'
                  local lines = lines_from(file)
                  print(lines[1])
                  if(count > 10) then return end
               end
            end
         end
      end
   end
end
print (count)
-- read the file 
local file = 'linescans/linescan_1.2e-09_0.1_1.1e-08_30_30.txt'
local lines = lines_from(file)


-------------------------------------------------------------


im = aig_new_image(N/16,N) -- Creation of a new empty image sized 1024x1024

background = {}		 -- New empty table for the background
background_density = 8 	 -- background "map" will be 8x8 pixels large
for i = 1, background_density*background_density, 1 do
  table.insert(background, math.random()*(0.2-0.1) + 0.1) -- value for each pixel is randomly generated, value varies from 0.1 to 0.2
end
aig_apply_background_image(im, {background_density, background_density}, background) -- application of the background to the image

edge_effect = aig_new_effect("edge", 0.4, 0.5) -- definition of the edge-effect
fine_structure = aig_new_effect("finestructure", 70e-4, 6, 10, 0.95, 1.05) -- definition of the fine structure


curves = {} -- new empty table for crves 
 
for j = 1,2*N,1  do
   k1,v1 =string.match(lines[j], "(%S+),(%S+)")
   k2,v2 = string.match(lines[j+1], "(%S+),(%S+)")
   --curves[i] = aig_new_curve("segment", {{tonumber(v1),tonumber(k1)}, {tonumber(v2),tonumber(k2)}})
   table.insert(curves, aig_new_curve("segment", {{tonumber(v1),tonumber(k1)}, {tonumber(v2),tonumber(k2)}}))
end


logo_feature = aig_new_feature(curves, {edge_effect, fine_structure}, 0.3) -- composition of the curves and effect into a feature
aig_move_feature(logo_feature, {-10,0}) -- shifting of the feature to the center of the image

features = {} -- new empty table for the feature
features[1] = logo_feature -- one only feature is the logo_feature

logo_sample = aig_new_sample(N/16, N, features) -- creation if the sample
aig_paint_sample(im, logo_sample) -- painting of the sample to the image
aig_delete_sample(logo_sample) -- sample is no more needed, thus it should be deleted
aig_apply_gaussian_psf(im, 0.5,1,30) -- application of the Gaussian blur

--aig_apply_noise(im, "poisson", 2) -- application of Poisson noise
--aig_apply_noise(im, "gaussian", 0.01)


aig_save_image(im, "original_images/oim.tiff","Rough curve by Narendra Chaudhary") --saving of the image to a file
aig_delete_image(im) -- deletion of the image


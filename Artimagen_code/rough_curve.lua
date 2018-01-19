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

-- tests the functions above
local file = 'linescan_1.7e-09_0.95_2.5e-08.txt'
local lines = lines_from(file)

--k,v =string.match(lines[1], "(%d+),(%d+)")
--print(tonumber(k),type(tonumber(v)))


-- print all line numbers and their contents
--for k,v in pairs(lines) do
--  print('line[' .. k .. ']', v)
--end



-------------------------------------------------------------

math.randomseed(os.time()) -- Initialization of the random number generator with current time as seed

N = 1024

im = aig_new_image(N,N) -- Creation of a new empty image sized 512x512

background = {} -- New empty table for the background
background_density = 8 -- background "map" will be 8x8 pixels large
for i = 1, background_density*background_density, 1 do
  table.insert(background, math.random()*(0.2-0.1) + 0.1) -- value for each pixel is randomly generated, value varies from 0.1 to 0.2
end
aig_apply_background_image(im, {background_density, background_density}, background) -- application of the background to the image

edge_effect = aig_new_effect("edge", 0.2, 0.5) -- definition of the edge-effect
fine_structure = aig_new_effect("finestructure", 70e-4, 6, 10, 0.95, 1.05) -- definition of the fine structure

curves = {} -- new empty table for crves 

i=1
while i <= 14*N - 1 do
   k1,v1 =string.match(lines[i], "(%d+),(%d+)")
   k2,v2 = string.match(lines[i+1], "(%d+),(%d+)")
   curves[i] = aig_new_curve("segment", {{tonumber(v1),tonumber(k1)}, {tonumber(v2),tonumber(k2)}})
   i = i+1
end

logo_feature = aig_new_feature(curves, {edge_effect, fine_structure}, 0.3) -- composition of the curves and effect into a feature
--aig_move_feature(logo_feature, {60,60}) -- shifting of the feature to the center of the image

features = {} -- new empty table for the feature
features[1] = logo_feature -- one only feature is the logo_feature

logo_sample = aig_new_sample(N, N, features) -- creation if the sample
aig_paint_sample(im, logo_sample) -- painting of the sample to the image
aig_delete_sample(logo_sample) -- sample is no more needed, thus it should be deleted
aig_apply_gaussian_psf(im, 0.5,1,30) -- application of the Gaussian blur

-- definition of the drift/vibration.
freqs = 8 -- the vib. function will be composed of 8 sine functions
ampl=0.2 -- amplitude on each function is 0.2 pixels
freq=100 -- maximum frequency is 100, minimum is 0
vibs={} -- empty table for sine functions
for i = 1, freqs, 1 do
  vibs[i]={math.random()*(ampl), math.random()*(ampl), math.random()*(freq), math.random()*(2*math.pi)} -- random generation of parameters
end

--aig_apply_vib(im,10000,50,100,0,vibs) -- application of vibrations
aig_apply_noise(im, "poisson", 2) -- application of Poisson noise
--aig_apply_noise(im, "gaussian", 0.01)

--gaig_preview(im) -- gAIG specific! preview. If running with artimagenl, comment this line out.
im1 = aig_copy_image(im)
im2 = aig_copy_image(im)
im3 = aig_copy_image(im)
im4 = aig_copy_image(im)
aig_crop_image(im1,0,0,256,1024)
aig_crop_image(im2,256,0,512,1024)
aig_crop_image(im3,512,0,768,1024)
aig_crop_image(im4,768,0,1024,1024)

aig_save_image(im1, "rough_curve1.tiff","Rough curve by Narendra Chaudhary") 
aig_save_image(im2, "rough_curve2.tiff","Rough curve by Narendra Chaudhary") 
aig_save_image(im3, "rough_curve3.tiff","Rough curve by Narendra Chaudhary") 
aig_save_image(im4, "rough_curve4.tiff","Rough curve by Narendra Chaudhary") 
aig_delete_image(im1)
aig_delete_image(im2)
aig_delete_image(im3)
aig_delete_image(im4)

aig_save_image(im, "rough_curve.tiff","Rough curve by Narendra Chaudhary") --saving of the image to a file
aig_delete_image(im) -- deletion of the image


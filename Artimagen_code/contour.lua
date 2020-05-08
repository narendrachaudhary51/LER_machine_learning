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
local file = '/home/grads/n/narendra5/Desktop/Programs/LER_machine_learning/linescans/linescan_1e-09_0.3_2.7e-08_30_60.txt'
local lines = lines_from(file)


-------------------------------------------------------------

math.randomseed(os.time()) -- Initialization of the random number generator with current time as seed

N = 1024

im = aig_new_image(N/4,N) -- Creation of a new empty image sized 512x512

background = {} -- New empty table for the background
background_density = 8 -- background "map" will be 8x8 pixels large
for i = 1, background_density*background_density, 1 do
  --table.insert(background, math.random()*(0.2-0.1) + 0.1) -- value for each pixel is randomly generated, value varies from 0.1 to 0.2
  table.insert(background, 0) -- value for each pixel is randomly generated, value varies from 0.1 to 0.2
end
aig_apply_background_image(im, {background_density, background_density}, background) -- application of the background to the image

base = math.random()*0.4 + 0.1
edge_effect = aig_new_effect("edge", math.random()*(0.8) + 0.1, math.random()*(0.9-base) + 0.1) -- definition of the edge-effect
fine_structure = aig_new_effect("finestructure", 70e-4, 6, 10, 0.95, 1.05) -- definition of the fine structure

curves = {} -- new empty table for crves 

--width = 30
--space = 30
--Xi = 37
--alpha = 0.3
--shift = math.floor(-25 + (width + space/2 + Xi + alpha*10 + sigma*10)%16)

--[[
i=1
while i <= 3*N - 1 do
   k1,v1 =string.match(lines[i], "(%d+),(%d+)")
   k2,v2 = string.match(lines[i+1], "(%d+),(%d+)")
   v1 = v1 - 30      --remove offset
   v2 = v2 - 30
   roll = math.floor(i/1024) + 1
   
   if roll %2 == 0 then
      print(v1 - (roll/2)*width - ((roll/2) - 1)*space, v2 - (roll/2)*width - ((roll/2) - 1)*space)
   elseif roll %2 == 1 then
      print(v1 - ((roll - 1)/2)*width - ((roll-1)/2)*space, v2 - ((roll - 1)/2)*width - ((roll-1)/2)*space)
      --curves[i] = aig_new_curve("segment", {{tonumber(v1),tonumber(k1)}, {tonumber(v2),tonumber(k2)}})
   end
   i = i+1
end
--]]



function square_contour(x,y,width,space, r_index)
   k1,v1 = y,x
   k2,v2 = y,(x+1)                            -- start of the square

   local i = 1
   while i <= 4*width do
      nk1,nv1 =string.match(lines[r_index + i], "(%S+),(%S+)")
      nk1 = math.floor(nk1+0.5)
      nv1 = math.floor(nv1+0.5)
      --nk2,nv2 = string.match(lines[r_index+ i + 1], "(%d+),(%d+)")
      nv1 = nv1 - 30                           --remove offset
      --nv2 = nv2 - 30                           --remove offset
      
      roll = math.floor((r_index + i - 1)/1024) + 1
   
      if roll %2 == 0 then
         nv1 = nv1 - (roll/2)*width - ((roll/2) - 1)*space
      --   nv2 = nv2 - (roll/2)*width - ((roll/2) - 1)*space
      elseif roll %2 == 1 then
         nv1 = nv1 - ((roll - 1)/2)*width - ((roll-1)/2)*space
      --   nv2 = nv2 - ((roll - 1)/2)*width - ((roll-1)/2)*space
      end
      if r_index < 100 then
         print(nv1)
      end

      if i < width then
        v1 = v1 + 1
        v2 = v2 + 1
        k1 = k2
        k2 = y + nv1
      elseif i == width then
         --v2 = v2 - 1
         --k2 = k2 + 1
         v1 = x + width - 1
         v2 = x + width - 1
         k1 = y
         k2 = y + 1
         --print(v1,v2,k1,k2)
      elseif i < 2*width then
         v1 = v2
         v2 = x + width - 1 + nv1
         k1 = k1 + 1
         k2 = k2 + 1
      elseif i == 2*width then
         --v2 = v2 - 1
         --k2 = k2 - 1
         v1 = x + width - 1
         v2 = x + width - 2
         k1 = y + width - 1
         k2 = y + width - 1
         --print(v1,v2,k1,k2)
      elseif i < 3*width then
         v1 = v1 - 1
         v2 = v2 - 1
         k1 = k2
         k2 = y + width - 1 + nv1
      elseif i == 3*width then
         --v2 = v2 + 1
         --k2 = k2 - 1
         v1 = x
         v2 = x
         k1 = y + width - 1 
         k2 = y + width - 2
         --print(v1,v2,k1,k2)
      elseif i < 4*width then
         v1 = v2
         v2 = x + nv1
         k1 = k1 - 1
         k2 = k2 - 1
      elseif i == 4*width then
         --v2 = v2 + 1
         --k2 = k2 + 1
         v1 = x
         v2 = x + 1
         k1 = y
         k2 = y
      end
      curves[i] = aig_new_curve("segment", {{tonumber(v1),tonumber(k1)}, {tonumber(v2),tonumber(k2)}})
      i = i+1
   end
   square_feature = aig_new_feature(curves, {}, 1) -- composition of the curves and effect into a feature
   --square_feature = aig_new_feature(curves, {edge_effect, fine_structure}, base) -- composition of the curves and effect into a feature
   --aig_move_feature(square_feature, {60,60}) -- shifting of the feature to the center of the image
   return square_feature
end

sigma = 1
Xi = 27
alpha = 0.3
width = 30
space = 60
shift = math.floor(-25 + (width + space/2 + Xi + alpha*10 + sigma*10)%16)
print(shift)
x = shift
y = shift
rough_index = 0
features = {} -- new empty table for the feature

for j = 1,8 do
   for k = 1,8 do
      features[(j-1)*8 + k] = square_contour(x + (j-1)*(width + space),y + (k-1)*(width + space),width, space, rough_index) -- one only feature is the logo_feature
      rough_index = rough_index + 4*width
   end
end


logo_sample = aig_new_sample(N/4, N, features) -- creation if the sample
aig_paint_sample(im, logo_sample) -- painting of the sample to the image
aig_delete_sample(logo_sample) -- sample is no more needed, thus it should be deleted

--aig_apply_gaussian_psf(im, math.random(1,3)*0.5,1,30) -- application of the Gaussian blur

-- definition of the drift/vibration.
freqs = 8 -- the vib. function will be composed of 8 sine functions
ampl=0.2 -- amplitude on each function is 0.2 pixels
freq=100 -- maximum frequency is 100, minimum is 0
vibs={} -- empty table for sine functions
for i = 1, freqs, 1 do
  vibs[i]={math.random()*(ampl), math.random()*(ampl), math.random()*(freq), math.random()*(2*math.pi)} -- random generation of parameters
end

--aig_apply_vib(im,10000,50,100,0,vibs) -- application of vibrations
--aig_apply_noise(im, "poisson", 200) -- application of Poisson noise
--aig_apply_noise(im, "gaussian", 0.01)


aig_save_image(im, "rough_curve.tiff","Rough curve by Narendra Chaudhary") --saving of the image to a file
aig_delete_image(im) -- deletion of the image


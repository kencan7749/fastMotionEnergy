# fastMotionEnergy

Here is the python repository for calculating MotionEnergy from gray scale video (height, width, frame).

I transported and refactored the matlab version publically avairable at gallant lab (https://github.com/gallantlab/motion_energy_matlab).


## usage
```
from fastMotionEnergy import fastMotionEnergyModel
from preprocWavelets_grid_GetMetaParams import preprocWavelets_gird_GetMetaParams

params = preprocWavelets_gird_GetMetaParams(2) #2 is minimum one, the detail is at the original github page.

stimsize = (224, 224, 16) #set the video size for analyzing
#initialize
me = fastMotionEnergyModel(stimsize, params)

motionenergy = me.calculate(vid) #vid is gray scale video whose shape is (height, width, frame)

#motion energy is the 2D map whose shape is (frame, filter), and each elements indiated the motion energy of filter output at certain time t.
```


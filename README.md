# scattering
Animations of photon scattering: Thomson, Rayleight and Resonant
scattering processes. Note: this is not a simulation code!

Version 0.1: written.
P. Simoes, 14 Feb 2018
paulo.simoes@glasgow.ac.uk

Written in Python 2. Requires matplotlib, scipy, numpy.

Call it via terminal:
python scat.0.1.py

usage: scat.0.1.py [-h] [-d DENSITY] [-c CHOICE] [-s SCALE] [-l LENGTH]
                   [-t TIME] [-p PSIZE] [-v VIDEO] [-r RATE]

optional arguments:
  -h, --help            show this help message and exit
  -d DENSITY, --density DENSITY
                        number of particles (default=400)
  -c CHOICE, --choice CHOICE
                        Scattering process: CHOICE = 0 Resonant, CHOICE = 1
                        Rayleight, CHOICE = 2 Thomson (default: 0)
  -s SCALE, --scale SCALE
                        scale factor for the cross-section radius (default=1)
  -l LENGTH, --length LENGTH
                        length of the scattering region (0 < L < 1)
                        (default=0.7)
  -t TIME, --time TIME  time duration of the animation, in seconds
                        (default=16)
  -p PSIZE, --psize PSIZE
                        scale factor for display photon size (default=1)
  -v VIDEO, --video VIDEO
                        set filename of mp4 video
  -r RATE, --rate RATE  rate of new photons per frame (default=5)

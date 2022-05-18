"""

Animations of photon scattering: Thomson, Rayleight and Resonant
scattering processes. Note: this is not a simulation code!

Version 0.1: written.
P. Simoes, 14 Feb 2018
Version 0.2: Python 3 version, fixed get_colormap() indexing
P. Simoes, 18 May 2022
paulo@craam.mackenzie.br

Many thanks to Jake Vanderplas for his animation tutorial
https://github.com/jakevdp

TO DO: 
- check args for valid inputs
- add absorption?
- optimize the code
- clean up a bit

"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from numpy import sin, cos
from scipy.spatial.distance import cdist
import matplotlib.gridspec as gridspec

# from matplotlib.patches import Circle
# from matplotlib.collections import PatchCollection

# import sys
import argparse

class anime:
    def __init__(self,
        nphot=1000, npart=200, nwav=15, Lx=0.7,choice=1,scale=1.0,v=0.4
        ,video=False,duration=16,DEBUG=False,filename='animation.mp4',
        psize=1.0,rate=5):

        self.upbound = 0.75
        self.model = ('Resonant','Rayleigh','Thomson')[choice]
        self.nphot = nphot
        self.nwav = nwav
        self.npart = npart
        self.video = video
        self.cross_section = 0.005 * scale
        self.bounds = [0,1,0,self.upbound]
        self.cm = self.get_colormap()

        self.data = np.random.uniform(0.0,1.0,(self.nphot,3))
        self.color = np.zeros((self.nphot,4))

        self.wave = np.zeros(self.nphot, dtype=int)
        self.draw = np.zeros(self.nphot)

        self.distance = np.zeros(self.nphot, dtype=float)
        self.countcoll = np.zeros(self.nphot, dtype=int)

        self.detected = np.zeros(self.nwav)
        self.created = np.zeros(self.nwav)
        self.rate = rate
        self.dt = 1.0 / 30.
        self.v = v
        self.psize = 0.001 * psize

        frames = int(duration/self.dt)

        self.part = np.c_[np.random.uniform(0.5-Lx/2.0, 0.5+Lx/2.0, npart)
            ,np.random.uniform(0.05, self.bounds[3]-0.05, npart)]
        
        if DEBUG:
            self.part = np.c_[np.ones(npart)*0.5,np.linspace(0,self.upbound,npart)]

        if self.model == 'Thomson':
            sigmaR = np.full(3,self.cross_section+self.psize,dtype=float)
            
        if self.model == 'Rayleigh':
            sigmaR = np.array([0,self.nwav/2,self.nwav-1])/float(self.nwav-1)*(self.cross_section+self.psize)

        if self.model == 'Resonant':
            w = np.array([0.0,self.nwav/2,self.nwav-1])
            sigmaR = (self.cross_section+self.psize) * np.exp( -(w-self.nwav/2)**2.0 *0.15)
                                                       
        sigmaR += np.finfo(float).eps
        self.MFPtheory = (Lx*self.upbound)/float(self.npart)/sigmaR ## 2D case (np.pi*sigmaR**2.0)

        print(self.MFPtheory)

        ## setup plot and animation
        self.fig = plt.figure(figsize=(7,7),facecolor='white')

        gs = gridspec.GridSpec(4, 5)
        gs.update(left=0.03, right=0.97, top=0.98,bottom=0.05,wspace=0.0,hspace=0.175)
        self.axh = plt.subplot(gs[:1, 3:],frameon=True)
        self.axm = plt.subplot(gs[1:, :],frameon=True)
        self.axt = plt.subplot(gs[:1, :2],frameon=False)

        self.axt.set_xlim(0, 1)
        self.axt.set_ylim(0, 1)
        self.axt.set_xticks([])
        self.axt.set_yticks([])
        
        self.axh.set_yticklabels([])

        self.axm.set_xlim(0, 1)
        self.axm.set_ylim(0, self.upbound)
        self.axm.set_yticks([])

        self.axh.set_xlim(self.nwav, -1)
        self.axh.set_ylim(-0.05, 1.05)

        self.axh.set_ylabel('Detected')
        self.axh.set_xticks([0,3.5,7,10.5,14])
        dl = '$\Delta \lambda$'
        self.axh.set_xticklabels(['Red','+'+dl,'$\lambda$','-'+dl,'Blue'])

        self.ani = animation.FuncAnimation(self.fig, self.update, frames=frames,
            interval=30, blit=True, init_func=self.setup,repeat=False)

        # Set up formatting for the movie files
        if self.video:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=30, metadata=dict(artist='@pjasimoes')
                , bitrate=1800,extra_args=['-vcodec', 'libx264'])
            self.ani.save(filename+'.mp4', writer=writer)

    def setup(self):
        ms = np.diff(self.axm.transData.transform((0,self.psize)))
        print(ms)

        if self.video == True:
            ms *= 3

        self.scat = self.axm.scatter(self.data[:,0],self.data[:,1]
            ,c=self.color,s=int(ms),edgecolors='None')

        # patches = []
        for i in range(0,self.npart):
            circle = plt.Circle(self.part[i,:],self.cross_section,color='k',fill=False,clip_on=False)
            self.axm.add_artist(circle)

        self.ctext = self.axt.text(0.1, 0.98, [], transform=self.axt.transAxes,verticalalignment='top')

        self.cre, = self.axh.plot(range(0,self.nwav),self.created,c='gray',marker='o',alpha=0)
        self.det, = self.axh.plot(range(0,self.nwav),self.detected,color='k',marker='o',markeredgecolor='None')

        return self.scat,self.ctext,self.det,self.cre,

    def get_colormap(self):
        colors = np.ones((self.nwav,4))
        tables = ('Reds','Greens','Blues')
        for i in range(len(tables)):
            mymap = plt.get_cmap(tables[i])
            # used int() in self.nwav/3; nwav=15, so things work fine
            colors[i*int(self.nwav/3):i*int(self.nwav/3)+int(self.nwav/3),:] = mymap(np.linspace(0.5, 1.0, int(self.nwav/3)))
        return colors

    def getalive(self):
        idx = np.where(self.draw == 1)[0]
        idx = np.asarray(idx)
        return idx

    def getdead(self):
        idx = np.where(self.draw == 0)[0]
        idx = np.asarray(idx)
        return idx

    def create(self):
        if self.draw.sum() < self.draw.size:
            idxa = self.getdead()
            np.random.shuffle(idxa)
            idx = idxa[:self.rate]
            self.wave[idx] = np.random.randint(0,self.nwav,
                idx.size)
            self.color[idx,:] = self.cm[self.wave[idx]]
            self.draw[idx] = 1
            self.created += np.histogram(self.wave[idx],bins=self.nwav,range=(0,14))[0]
            self.data[idx,0] = 0.0
            self.data[idx,1] = np.random.uniform(
                self.upbound/2.-0.1,self.upbound/2.+0.1,idx.size)
            self.data[idx,2] = 0.0
            self.countcoll[idx] = 0
            self.distance[idx] = 0.0

    def boundary(self):
        # check for crossing boundary
        idx = self.getalive()
        crossed_x1 = (self.data[idx, 0] < self.bounds[0])
        crossed_x2 = (self.data[idx, 0] > self.bounds[1])
        crossed_y1 = (self.data[idx, 1] < self.bounds[2])
        crossed_y2 = (self.data[idx, 1] > self.bounds[3])

        oob = crossed_x1 | crossed_x2 | crossed_y1 | crossed_y2

        self.detected += np.histogram(self.wave[idx[crossed_x2]]
            ,bins=self.nwav,range=(0,14))[0]

        self.draw[idx[oob]] = 0

    def collisions(self):

        # array shape: self.nphot,self.npart
        D = cdist(self.data[:,:2],self.part)

        ## Define collision threshold based on cross-section radius
        ## get theoretical mean free path: l=1/(n*sigma)
        if self.model == 'Thomson': 
            threshold = self.cross_section+self.psize # scalar
    
        # np.tile(a,[2,1]).transpose().shape
        if self.model == 'Rayleigh':
            threshold = np.tile( ((self.cross_section+self.psize) 
                * ( (self.wave/float(self.nwav-1))**2. ) )
                ,([self.npart,1])).transpose()

        if self.model == 'Resonant':
            threshold = np.tile( ( (self.cross_section+self.psize) 
                * np.exp( -(self.wave-np.floor(self.nwav/2.0))**2.0 *0.15) ) 
                ,([self.npart,1] ) ).transpose()

        ind1, ind2 = np.where(D < threshold)

        # count collisions
        self.countcoll[ind1] += 1

        self.data[ind1,2] = np.random.uniform(0,4*np.pi,ind1.shape)
        # update direction of colliding photons (randomize direction)

    def showtext(self):

        idx = self.getalive()
        
        # red = np.where(self.wave[idx] <= 4)
        # blu = np.where(np.logical_and(self.wave[idx] >= 5, self.wave[idx] <= 9))
        # grn = np.where(self.wave[idx] >= 10)

        label = self.model + ' scattering\n\n'
        label += 'Mean Free Path \n'

        name=('R','G','B')

        if len(idx) == 0:
            for i in name:
                label += i
        else:
            red = np.where(self.wave[idx] == 0)[0]
            grn = np.where(self.wave[idx] == self.nwav/2)[0]
            blu = np.where(self.wave[idx] == self.nwav-1)[0]

            for j,i in enumerate((red,grn,blu)):
                if len(i) <= 1:
                    str1 = '$\gg$ 1'
                else:
                    dist = np.mean(self.distance[idx[i]])
                    coll = np.mean(self.countcoll[idx[i]])
                    mfp = dist/(np.sqrt(coll)+np.finfo(float).eps)
                    if mfp > 10:
                        str1 = '$\gg$ 1'
                    else:
                        str1 = '='+'{:4.2f}'.format(mfp)
                label += name[j]+str1+'\n'

        #label += '\nanimation by @pjasimoes\n'

        self.ctext.set_text(label)

        return self.ctext

    def update(self,i):

        self.create()
        self.showtext()
        
        idx = self.getalive()
        
        self.data[idx, 0] += self.dt * self.v * cos(self.data[idx, 2])
        self.data[idx, 1] += self.dt * self.v * sin(self.data[idx, 2])

        self.distance[idx] += self.dt * self.v

        self.boundary()        
        self.collisions()

        self.scat.set_offsets(self.data[:,:2])
        self.scat.set_color(self.color[:])

        norm = np.amax(self.created)
        
        self.cre.set_data(range(0,self.nwav),self.created/norm)
        self.det.set_data(range(0,self.nwav),self.detected/norm)

        return self.scat,self.det,self.cre,self.ctext,

    def show(self):
        plt.show()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--density", help="number of particles (default=400)",type=int)
    parser.add_argument("-c","--choice", help="Scattering process: CHOICE = 0 Resonant, CHOICE = 1 Rayleight, CHOICE = 2 Thomson (default: 0)",type=int)
    parser.add_argument("-s","--scale", help="scale factor for the cross-section radius (default=1)",type=float)
    parser.add_argument("-l","--length", help="length of the scattering region (0 < L < 1) (default=0.7)",type=float)
    parser.add_argument("-t","--time", help="time duration of the animation, in seconds (default=16)",type=float)
    parser.add_argument("-p","--psize", help="scale factor for display photon size (default=1)",type=float)
    parser.add_argument("-v","--video", help="set filename of mp4 video")
    parser.add_argument("-r","--rate", help="rate of new photons per frame (default=5)",type=int)

    parser.parse_args()
    args = parser.parse_args()

    rate = 5
    Lx=0.7
    duration=16
    scale=1.0
    npart=400
    video=False
    choice=0
    psize = 0.001
    filename = 'tmp.mp4'

    if args.density:
        npart = int(args.density)
    if args.choice:
        choice = args.choice
    if args.scale:
        scale = args.scale
    if args.length:
        Lx = args.length
    if args.time:
        duration = args.time
    if args.video:
        filename = args.video
        print('mp4 file: '+filename)
        video = True
    if args.psize:
        psize = args.psize
    if args.rate:
        rate = int(args.rate)

    a = anime(Lx=Lx,duration=duration,choice=choice,scale=scale,
        video=video,npart=npart,filename=filename,psize=psize)


## python test5.py -c 0 -v resonant
## python test5.py -c 1 -v rayleigh
## python test5.py -c 2 -v thomson

    if not video:
        np.seterr(all='raise')
        a.show()

if __name__ == '__main__':

    main()

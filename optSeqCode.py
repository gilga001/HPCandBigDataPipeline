class Frame(object):
    def __init__(self, timestep, type1, type2, type3,
                 type4, type5, type6, type7):
        self.timestep = timestep
        self.type1 = type1
        self.type2 = type2
        self.type3 = type3
        self.type4 = type4
        self.type5 = type5
        self.type6 = type6
        self.type7 = type7

class Atom(object):
    def __init__(self, atom_id, atom_type,
                 atom_x, atom_y, atom_z):
        self.atom_id = atom_id
        self.atom_type = atom_type
        self.atom_x = atom_x
        self.atom_y = atom_y
        self.atom_z = atom_z

import string,sys
import math
import os
from inspect import currentframe, getframeinfo
import re
import fileinput
import numpy as np
from operator import itemgetter
import scipy as sp
import scipy.stats
import warnings

actualStart = 0   # if dump file not start at 0
timestep = 10          # in the original data, the minimum time span between two frames
N_first = 0        # the beginning time which computation starts, absolute time
N_last  = 5000000         # the end time which computation finishes
nevery = 10              # print every this many snapshots
width = 100000            # the time span which computation uses, test 50ps lifetime
frameLine = 203           # the number of lines between two "ITEM:TIMESTEP", i.e. atom# + 9
warnings.simplefilter("ignore")

#################################################################################################                                                                                     ################
rC = [5]                                                                                                           ################
rHS = [3,4]                                                                                                            ################
rOS = [6]                                                                                                              ################
rHW = [1]                                                                                                                ################
rOW = [2]                                                                                                                ################
#################################################################################################
################################################################################################################
def readFile():
    frame = []
    frames = []
    lineNo = 0 #line NO in a frame
    timestep = 0
    type1 = []
    type2 = []
    type3 = []
    type4 = []
    type5 = []
    type6 = []
    type7 = []
    rootDir = '/scratch2/yli25'#input file directory
    for lammpfile in os.listdir(rootDir):
        if lammpfile.endswith("dump.ch2oh33_prod.lammpstrj"):
            myfile = lammpfile
            print(myfile)
    for line in fileinput.input("%s" %(str(rootDir + "/" + myfile))):
        #if (fileinput.lineno() >= frameLine*(N_first-actualStart)//timestep+1 and fileinput.lineno() <= frameLine*((N_last-actualStart)//timestep+1)):
        if( ((fileinput.lineno() - 1) >= 0) and((fileinput.lineno() - 1) < 9) ):
            wholeline = line.split()   # split the line on runs of whitespace
            words = [s for s in wholeline if s != None]
            frame.append(words)
            lineNo = fileinput.lineno() - 1
            if( lineNo == 1 ):
                timestep = int(words[0])
        elif ( ((fileinput.lineno() - 1) != 0) and (fileinput.lineno() - 1) % frameLine == 0):
            atoms = Frame(timestep, type1, type2, type3, type4, type5, type6, type7)
            frames.append(atoms)
            lineNo = 0
            type1 = []
            type2 = []
            type3 = []
            type4 = []
            type5 = []
            type6 = []
            type7 = []
        elif( (lineNo < 8) ):#skip 9 lines
            if( lineNo == 0 ):
                wholeline = line.split()   # split the line on runs of whitespace
                words = [s for s in wholeline if s != None]
                timestep = int(words[0])
            lineNo += 1
            continue
        else:
            wholeline = line.split()   # split the line on runs of whitespace
            words = [s for s in wholeline if s != None]
            atom = Atom(int(words[0]), int(words[1]), float(words[2]), float(words[3]), float(words[4]))
            if(atom.atom_type == 7): type7.append(atom)
            elif(atom.atom_type == 2): type2.append(atom)
            elif(atom.atom_type == 1): type1.append(atom)
            elif(atom.atom_type == 3): type3.append(atom)
            elif(atom.atom_type == 4): type4.append(atom)
            elif(atom.atom_type == 5): type5.append(atom)
            elif(atom.atom_type == 6): type6.append(atom)

    fileinput.close()
    return frame, frames


def dist(X,Y): # square of distance
    x1 = X[1]
    x2 = Y[1]
    y1 = X[2]
    y2 = Y[2]
    z1 = X[3]
    z2 = Y[3]
    return 2*(x1-x2)*(y1-y2)*a*b*cosgamma + 2*(y1-y2)*(z1-z2)*b*c*cosalpha + 2*(x1-x2)*(z1-z2)*a*c*cosbeta + (x1-x2)*(x1-x2)*a*a + (y1-y2)*(y1-y2)*b*b + (z1-z2)*(z1-z2)*c*c

def dipo(X,Y):  # input is dipole vector rather than point coords
    x1 = X[1]*(float(words[1])-float(words[0])-float(words[2])) + X[2]*float(words[2])
    x2 = Y[1]*(float(words[1])-float(words[0])-float(words[2])) + Y[2]*float(words[2])
    y1 = X[2]*(float(words[4])-float(words[3]))
    y2 = Y[2]*(float(words[4])-float(words[3]))
    z1 = X[3]*(float(words[7])-float(words[6]))
    z2 = Y[3]*(float(words[7])-float(words[6]))
    if (x2*x2+y2*y2+z2*z2 == 0 or x1*x1+y1*y1+z1*z1 == 0):
        return 0
    else:
        return (x1*x2+y1*y2+z1*z2)/math.sqrt((x2*x2+y2*y2+z2*z2)*(x1*x1+y1*y1+z1*z1))

def init(items):
    temp = []
    coords_replica = {}  # the real coords that are used, one of the 9 replica
    x = []
    tmp = []
    OW = []
    h1 = h2 = o1 = o2 = [None] * 100000   # at least 8 times larger than one box h2o atoms.

    type34 = items.type3 + items.type4
    h1 = findNonperiodic (rHS,type34)   # item# and x, y, z for each h1 or h2 or o1 or o2
    h2 = find (rHW,items.type1)   # item# and x, y, z for each h1 or h2 or o1 or o2
    o1 = findNonperiodic (rOS,items.type6)   # item# and x, y, z for each h1 or h2 or o1 or o2
    o2 = find (rOW,items.type2)   # item# and x, y, z for each h1 or h2 or o1 or o2
    h1 = [k for k in h1 if k != None]  #h1
    h2 = [k for k in h2 if k != None]  #h2
    o1 = [k for k in o1 if k != None]  #o1
    o2 = [k for k in o2 if k != None]  #o2

    for i in range(len(o1)):
        for j in range(len(o2)):
            do1o2 = dist(o1[i] , o2[j])
            if do1o2 <= float(13.5):
                for l in range(len(h2)):
                    do2h2 = dist(o2[j],h2[l])
                    if do2h2<=1.2:
                        do1h2 = dist(o1[i] , h2[l])
                        #A1 = ang( do1o2 , do1h2, do2h2)
                        A2 = ang( do2h2 , do1h2, do1o2)
                        if A2>120 and do1h2<6.25:
                        #if A1<=30 and A2>=120 and do1h2<=6.25:
                            for n in range(len(h2)):
                                if dist(o2[j], h2[n])<=1.2:
                                    tmp.append((o2[j][4],h2[n][4]))
                                    if (o2[j][4] not in coords_replica.keys()):
                                        coords_replica[o2[j][4]] = (o2[j][1],o2[j][2],o2[j][3])  # save found O, H coords, x y z
                                    if (h2[n][4] not in coords_replica.keys()):
                                        coords_replica[h2[n][4]] = (h2[n][1],h2[n][2],h2[n][3])  # save found O, H coords, x y z
                for l in range(len(h1)):
                    do1h1 = dist(o1[i],h1[l])
                    if do1h1<=1.2:
                        do2h1 = dist(o2[j] , h1[l])
                        # print do1h1,do1o2,do2h1
                        #A1 = ang( do1o2 , do2h1, do1h1 )
                        A2 = ang( do1h1 ,do2h1, do1o2 )
                        if  A2>120 and do2h1<6.25:
                        #if  A1<=30 and A2>=120 and do2h1<=6.25:
                            for m in range(len(h2)):
                                if dist(o2[j],h2[m])<=1.2:  # output HBed Ow and corresponed Hw. At least two pairs, Ow-Hw1 AND Ow-Hw2
                                    tmp.append((o2[j][4],h2[m][4])) # o2[j][4] is j line with item# ([4], already reordered
                                    if (o2[j][4] not in coords_replica.keys()):
                                        coords_replica[o2[j][4]] = (o2[j][1],o2[j][2],o2[j][3])  # save found O, H coords, x y z
                                    if (h2[m][4] not in coords_replica.keys()):
                                        coords_replica[h2[m][4]] = (h2[m][1],h2[m][2],h2[m][3])  # save found O, H coords, x y z
    tmp = list(set(tmp))   # convert 2 lines(2 pairs in 1 h2o mlc: Ow-Hw1 AND Ow-Hw2) to 1 lines: Ow Hw1 Hw2, thus deletes repeating Ow
    for p in range(len(tmp)):
        for q in range(p+1,len(tmp)):  # set the repeated value to -1
            if(tmp[p][0] == tmp[q][0] and tmp[p][1] == tmp[q][1]):
                tmp[q][0]= -1
        for t in range(p+1, len(tmp)):
            if(tmp[t][0]>0 and tmp[p][0] == tmp[t][0]):
                OW.append((tmp[p][0],tmp[p][1],tmp[t][1]))
    #print (coords_replica)
    #print(OW)
    #print(len(OW))
    return OW, coords_replica, len(OW)

def ang(d1,d2,d3):# Note: square of the distances
    return math.degrees(math.acos((d1 + d2 - d3)/(2.0 * math.sqrt(d1) * math.sqrt(d2))))

def findNonperiodic(y,c):
    x = [None] * len(c)

    for i in range(len(c)):
        x[i] = ((c[i].atom_type,c[i].atom_x,c[i].atom_y,c[i].atom_z,c[i].atom_id))  # read the whole line and reorder each string
    return (x)

#below is periodic boundary
def find(y,c):
    x=grab= grabxp = grabxn = grabyp = grabyn = grabxpyp = grabxnyn = grabxnyp = grabxpyn = [None] * len(c) * 9
    for i in range(len(c)):
        grab = ((c[i].atom_type,c[i].atom_x,c[i].atom_y,c[i].atom_z,c[i].atom_id))
        grabxp = ((c[i].atom_type,c[i].atom_x+1,c[i].atom_y,c[i].atom_z,c[i].atom_id))
        grabxn = ((c[i].atom_type,c[i].atom_x-1,c[i].atom_y,c[i].atom_z,c[i].atom_id))
        grabyp = ((c[i].atom_type,c[i].atom_x,c[i].atom_y+1,c[i].atom_z,c[i].atom_id))
        grabyn = ((c[i].atom_type,c[i].atom_x,c[i].atom_y-1,c[i].atom_z,c[i].atom_id))
        grabxpyp = ((c[i].atom_type,c[i].atom_x+1,c[i].atom_y+1,c[i].atom_z,c[i].atom_id))
        grabxpyn = ((c[i].atom_type,c[i].atom_x+1,c[i].atom_y-1,c[i].atom_z,c[i].atom_id))
        grabxnyp = ((c[i].atom_type,c[i].atom_x-1,c[i].atom_y+1,c[i].atom_z,c[i].atom_id))
        grabxnyn = ((c[i].atom_type,c[i].atom_x-1,c[i].atom_y-1,c[i].atom_z,c[i].atom_id))
        x[9*i+0]=grab
        x[9*i+1]=grabxp
        x[9*i+2]=grabxn
        x[9*i+3]=grabyp
        x[9*i+4]=grabyn
        x[9*i+5]=grabxpyp
        x[9*i+6]=grabxpyn
        x[9*i+7]=grabxnyp
        x[9*i+8]=grabxnyn
    return (x)

def coor(words):  # calculate a,b,c,cosalpha,cosbeta,cosgamma
    data = [None] * 6
    xlo_bound = float(words[0])
    xhi_bound = float(words[1])
    xy =  float(words[2])
    ylo_bound =  float(words[3])
    yhi_bound =  float(words[4])
    xz =  float(words[5])
    zlo =  float(words[6])
    zhi =  float(words[7])
    yz =  float(words[8])
    xlo = xlo_bound - min(0.0,xy,xz,xy+xz)
    xhi = xhi_bound - max(0.0,xy,xz,xy+xz)
    ylo = ylo_bound - min(0.0,yz)
    yhi = yhi_bound - max(0.0,yz)
    lx = xhi-xlo
    ly = yhi-ylo
    lz = zhi-zlo
    data[0] = lx   #a
    data[1] = math.sqrt(ly*ly+xy*xy)   #b
    data[2] = math.sqrt(lz*lz+xz*xz+yz*yz)  #c
    data[3] = (xy*xz+ly*yz)/data[1]/data[2]   #cosalpha
    data[4] = xz/data[2]    #cosbeta
    data[5] = xy/data[1]   #cosgamma
    return (data)

def computeMSD():
    i = 0
    msd_all = {}  # {'10':[0.3,#], '20':[0.8,#], .....}
    dipole_all = {}
    msd_avg = []
    dipole_avg = []

    while (i < len(my_MSD)):
        origin_MSD = []
        origin_dipole = []
        origin_len = my_MSD[i][-1]
        for j in range(origin_len):
            origin_MSD.append(my_MSD[i+j]) # origin coords and hb information
            origin_dipole.append(my_dipole[i+j])

        m = i + origin_len # next frame time
        match_list = []
        while (m < len(my_MSD) and my_MSD[m][0] - my_MSD[i][0] <= width): # block width
            dynamic_time = my_MSD[m][0] - my_MSD[i][0]

            for n in range(origin_len):
                if origin_MSD[n][-2] and origin_MSD[n][-2] == my_MSD[m][-2]:  # check if this H2O still exists in next frame
                    match_list.append(origin_MSD[n][-2]) # check if this H2O still exists in next frame
                    if (dynamic_time in msd_all.keys()):
                        msd_all[dynamic_time].append(dist(my_MSD[m], origin_MSD[n]))
                        dipole_all[dynamic_time].append(dipo(my_dipole[m], origin_dipole[n]))
                    else:
                        msd_all[dynamic_time] = [dist(my_MSD[m], origin_MSD[n])]
                        dipole_all[dynamic_time] = [dipo(my_dipole[m], origin_dipole[n])]
                    if (dynamic_time == timestep * nevery):
                        print(origin_MSD[n])
                        print(my_MSD[m])
                        print("=====================")
            m += 1
            # check if m is a new dynamic_time different from the previous one
            if m < len(my_MSD) and (my_MSD[m][0] - my_MSD[i][0]) != dynamic_time:
                # if the count is 0, all missing, break
                if len(match_list) == 0:
                    break
                    # if this count is less than the origin_len, some one missing
                elif len(match_list) < origin_len:
                    for c, row in enumerate(origin_MSD):
                        if not row[-2] in match_list: # delete individual origin H2O coords if HB breaks
                            origin_MSD[c][-2] = -1  # set it to -1 to hide it
                match_list = [] # update match_list for next timestep
        i += my_MSD[i][-1]  # next initial origin
    for key in msd_all.keys():
        msd_m, msd_h = mean_confidence_interval(msd_all[key])
        dipole_m, dipole_h = mean_confidence_interval(dipole_all[key])
        msd_avg.append([key,msd_m, msd_h])
        dipole_avg.append([key,dipole_m, dipole_h])
    msd_avg.sort(key=itemgetter(0))
    dipole_avg.sort(key=itemgetter(0))
    return msd_avg, dipole_avg

# reference:
# https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    #    if n < 5:
    #        m = str()
    #        h = str()
    #    else:
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    m = format(m, '.15f')
    h = format(h, '.15f')
    return m, h

###############################################################################################

myFrame, myFrames = readFile()  # skip the first skip frames
my_MSD = []
my_dipole = []
out_hblist = []
out_coord = []

words = [None]*9
N = int(myFrame[3][0])
for l in range(len(myFrame[5])):
    words[l] = myFrame[5][l]
if (len(myFrame[5]) == 2):
    words[2] = "0.0"
for m in range(len(myFrame[6])):
    words[3+m] = myFrame[6][m]
if (len(myFrame[6]) == 2):
    words[5] = "0.0"
for n in range(len(myFrame[7])):
    words[6+n] = myFrame[7][n]
if (len(myFrame[7]) == 2):
    words[8] = "0.0"
result = coor(words)
a = result[0]
b = result[1]
c = result[2]
cosalpha = result[3]
cosbeta = result[4]
cosgamma = result[5]

lenSum = 0.0;
for i in range(len(myFrames)):    # number of frames
    frameTime = int(myFrames[i].timestep)
    hb_list = []
    hb_list, real_coords, lenOW = init(myFrames[i])   ## hydrogen bonded water list, each line is Ow Hw1 Hw2, e.g. 41 76 77s
    lenSum += lenOW
    out_hblist.append(hb_list)
    out_coord.append(real_coords)

average = lenSum/(float(len(myFrames)))
print("average HB:", average)
# keep the output finally
outfile1 = open("out_hblist", "a")
outfile2 = open("out_coord", "a")
outfile1.write(str(out_hblist))
outfile2.write(str(out_coord))
import mdtraj as md
import numpy as np

t = md.load_lammpstrj('/scratch2/yli25/ch2oh.1g.lammpstrj', top='project/ch2oh33.psf')

hbLen = 0.0
#i = 50
for i in range(len(t)):
    hbonds = md.baker_hubbard(t[i], freq = 0.0, exclude_water=False, periodic = True)

    hbondList = []
    for hbond in hbonds:
        if(t[i].topology.atom(hbond[0]).residue.name != t[i].topology.atom(hbond[2]).residue.name):
            hbondList.append([hbond[0], hbond[1], hbond[2]])
    
    #print(hbondList)
    hbLen += len(hbondList)
    
averageHB = hbLen / (float(len(t)))
print(averageHB)
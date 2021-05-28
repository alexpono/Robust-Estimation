import STMFunctions as stmf
import sys
import math
import numpy as np
import itertools as it
import copy
import struct
from datetime import datetime

if(len(sys.argv)==2):
    filename = sys.argv[1]         # First argument is filename
    print(filename)

    maxframes = 999999
    tstart = datetime.now().timestamp()
    neighbours = 6
    boundingbox = [[0, 75], [-15, 60], [115,185]]
    [nx,ny,nz] = [70,70,70]
    cammatchfunc = lambda x: len(x)>=2        # We require a match to satisfy this requirement for the number of cameras (and thus the number of rays)
    maxmatchesperray = 1                      # Number of matches/ray
    maxdistance = 0.15                        # Max distance allowed for a match.

    fileout = copy.copy(filename).split(".")
    fileout = ".".join(fileout[0:len(fileout)-1])
    filelog = fileout + ".log"
    fileout = fileout.replace("rays","matched")+".dat"

    fout = open(fileout, 'wb')
    fin = open(filename, 'rb')
    frameid = 0
    numpts = fin.read(4)                                                 # Read 4 bytes header
    while(len(numpts)>0 and frameid < maxframes):                        # If something is read
        frameid += 1
        numpts = struct.unpack('I', numpts)[0]                           # Interpret header as 4 byte uint
        flog = open(filelog, 'a')
        flog.write("#######\n")
        flog.write("Frame: " + str(frameid) + "\nNumber of rays: " + str(numpts) + "\n")
        flog.close()
        
        print("Frame:",frameid,". # of rays:", numpts)
        
        # Read rays
        raydata = fin.read(numpts*27)                                     # 27 bytes per line 2+1+6*4
        raydata = struct.unpack('='+('BH6f'*numpts),raydata)              # Create string '=BHFFFFFFBHFFFFFFBHFFFFFF...BHFFFFFF'
        raydata = list(map(lambda i: list(raydata[8*i:8*i+8]) ,range(len(raydata)//8)))     # Reshape to 8*N np.arreyreshape converts everything to floats...
        # The actual call
        output = stmf.SpaceTraversalMatching(list(raydata),boundingbox,nx=nx,nz=nz,ny=ny,cammatchfunc=cammatchfunc,neighbours=neighbours,logfile=filelog,maxdistance=maxdistance)
        
        # Prepare output
        print("Matches found:",len(output))
        coutput = []
        for o in output:
            tmp = [len(o[0])]
            tmp.extend(o[1])
            tmp.append(o[2])
            for j in o[0]:
                tmp.extend(j)
            coutput.append(tmp)
        output = coutput
        del coutput
        
        # Output the output
        buf = struct.pack('I', len(output))                               # Write the number of matches
        fout.write(buf)
        
        for out in output:
            buf = struct.pack('=B4f' + out[0]*"BH", *out)                 # Write each to the output file
            fout.write(buf)

        numpts = fin.read(4)                                              # Read next header
        #print("numpts:",numpts)
        #print("type:",type(numpts))
    fout.close()
    fin.close()
    print("Finished")

    elapsed = datetime.now().timestamp() - tstart
    print("Elapsed time:",elapsed)
    print("Elapsed time/frame:",elapsed/frameid)
else:
    print("There should be an argument with the filename!")

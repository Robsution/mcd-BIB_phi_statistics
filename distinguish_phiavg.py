 #!/usr/bin/env python
"""This script converts a FLUKA binary file to an SLCIO file with LCIO::MCParticle instances"""

import argparse
import numpy as np


parser = argparse.ArgumentParser(description='Convert FLUKA binary file to SLCIO file with MCParticles')
parser.add_argument('files_in', metavar='FILE_IN', help='Input binary FLUKA file(s)', nargs='+')
parser.add_argument('-c', '--comment', metavar='TEXT',  help='Comment to be added to the header', type=str)
parser.add_argument('-n', '--normalization', metavar='N',  help='Normalization of the generated sample', type=float, default=1.0)
parser.add_argument('-f', '--files_event', metavar='L',  help='Number of files to merge into a single LCIO event (default: 1)', type=int, default=1)
parser.add_argument('-m', '--max_lines', metavar='M',  help='Maximum number of lines to process', type=int, default=None)
parser.add_argument('-z', '--invert_z',  help='Invert Z position/momentum', action='store_true', default=False)
parser.add_argument('--pdgs', metavar='ID',  help='PDG IDs of particles to be included', type=int, default=None, nargs='+')
parser.add_argument('--nopdgs', metavar='ID',  help='PDG IDs of particles to be excluded', type=int, default=None, nargs='+')
parser.add_argument('--t_max', metavar='T',  help='Maximum time of accepted particles [ns]', type=float, default=None)

args = parser.parse_args()

import random
import math

import csv

def bytes_from_file(filename):
	with open(filename, 'rb') as f:
		while True:
			chunk = np.fromfile(f, dtype=line_dt, count=1)
			if not len(chunk):
				return
			yield chunk
   
def difference_matrix(a):
    x = np.reshape(a, (len(a), 1))
    return (x - x.transpose()).flatten()

def find_most_antipodal(phis):
    # foundarr = np.zeros(phis.size)
    i = int(len(phis)/2)
    end = False
    
    while (np.abs(np.pi - (phis[i-1] - phis[0])) < np.abs(np.pi - (phis[i] - phis[0]))):
        i -= 1
    if i == int(len(phis)/2):
        while (np.abs(np.pi - (phis[i+1] - phis[0])) < np.abs(np.pi - (phis[i] - phis[0]))):
            i += 1
            if i+1 >= len(phis):
                end = True
                break
    maxdPhi = phis[i] - phis[0]
    if maxdPhi > np.pi:
        maxdPhi = np.abs(maxdPhi - 2 * np.pi)
    for iPhi, phi in enumerate(phis):
        if iPhi == 0:
            continue
        if not end:
            while (np.abs(np.pi - (phis[i+1] - phi)) < np.abs(np.pi - (phis[i] - phi))):
                i += 1
                if i+1 >= len(phis):
                    end = True
                    break
        newmax = phis[i] - phi
        if newmax > np.pi:
            newmax = np.abs(newmax - 2 * np.pi)
        maxdPhi = max(maxdPhi, newmax)
    return maxdPhi
    
def find_average(phi):
    '''
    For each phi with index i we take the difference with all elements in front of it.
    To avoid calculating both arc lengths the element k with which it exceeds the maximum arc length is found.
    Then since the phi list is assumed to be sorted we have a clear boundary for which direction to take to calculate the arc length
    '''
    N = len(phi)

    if N < 2:
        return 0.0

    total = 0
    k = 1
    for i in range(N-1):
        phi_i = phi[i]
        
        for idx in range(k, N):
            if np.abs(phi[idx] - phi_i) > np.pi:
                k = idx
                break
            if idx == N - 1:
                k = N
            
        total_i = 0.0
        diffs_part1 = phi[i+1:k] - phi_i
        total_i += np.sum(diffs_part1)
        
        if k < N:
            diffs_part2_sum = 0.0
            diffs_part2 = 2 * np.pi - (phi[k:N] - phi_i)
            diffs_part2_sum = np.sum(diffs_part2)
            
            total_i += diffs_part2_sum
            
        total += total_i

    avg_diff = total / (N*(N-1)/2)
    
    if phis.size < 5:
        print(phis)
        print(avg_diff)
    
    return avg_diff
        
def piRange(x1,x2):
    return np.abs(x1 - x2) if np.abs(x1 - x2) > np.pi else np.abs(np.abs(x1 - x2) - 2*np.pi)


# Binary format of a single entry
line_dt=np.dtype([
	('fid',  np.int32),
	('fid_mo',  np.int32),
	('E', np.float64),
	('x', np.float64),
	('y', np.float64),
	('z', np.float64),
	('cx', np.float64),
	('cy', np.float64),
	('cz', np.float64),
	('time', np.float64),
	('x_mu', np.float64),
	('y_mu', np.float64),
	('z_mu', np.float64)
])

######################################## Start of the processing
print(f'Converting data from {len(args.files_in)} file(s)\nwith normalization: {args.normalization:.1f}')
print(f'Storing {args.files_event:d} files/event')
if args.pdgs is not None:
	print(f'Will only use particles with PDG IDs: {args.pdgs}')

# Bookkeeping variables
random.seed()
nEventFiles = 0
nLines = 0
nEvents = 0
col = None
evt = None
nPar = 0

minPhis = None
maxPhis = None
minPhisUnif = None
maxPhisUnif = None
size1 = 0

flavors = set()
countGroups = -1
famSizeMax = 0
famSizeMin = np.inf

filename = "example1.csv"

# Reading the complete files
with open(filename, "a") as f:
    
    w = csv.writer(f)
    
    for iF, file_in in enumerate(args.files_in):
        if args.max_lines and nLines >= args.max_lines:
            break
    
        # Keeping track of position for correlation
        currPos_mu = ()
        phis = np.array([])
        phiDeltaCorr = np.array([np.ceil(args.normalization)])
        # Looping over particles from the file
        for iL, data in enumerate(bytes_from_file(file_in)):
            if args.max_lines and nLines >= args.max_lines:
                break
            nLines += 1

            # Extracting relevant values from the line
            fid,e, x,y,z, cx,cy,cz, x_mu,y_mu,z_mu, time = (data[n][0] for n in [
                'fid', 'E',
                'x','y','z',
                'cx', 'cy', 'cz',
                'x_mu', 'y_mu', 'z_mu',
                'time'
            ])

            # Converting the absolute time of the particle [s -> ns]
            t = time * 1e9

            # Converting the len units from cm to mm
            x = x * 10
            y = y * 10
            z = z * 10
            # Skipping if particle's time is greater than allowed
            if args.t_max is not None and t > args.t_max:
                continue

            # Calculating the components of the momentum vector
            mom = np.array([cx, cy, cz], dtype=np.float32)
            mom *= e

            nP_frac, nP = math.modf(args.normalization)
            #if nP_frac > 0 and random.random() < nP_frac:
            #    nP += 1
            nP = int(nP)

            # Do it!
            if (x_mu,y_mu,z_mu) != currPos_mu:
                print('Family',countGroups)
                # print("Family size:", phis.size)
                if phis.size > 1:
                    # print(phis)
                    #First sample
                    row = np.zeros(3)
                    phis = np.sort(phis)
                    phisUnif = np.sort((phis + np.random.uniform(0, 2 * np.pi, len(phis)) + np.pi) % (2 * np.pi) - np.pi)
                    phisCorr = np.sort((phis + phiDeltaCorr[0] + np.pi) % (2 * np.pi) - np.pi)
                    
                    row[0] = find_average(phis)
                    row[1] = find_average(phisUnif)
                    row[2] = find_average(phisCorr)
                    
                    
                    w.writerow(row)
                    # Oversampling
                    for iP in range(1, nP):
                        row[0] = None
                        
                        phisUnif = np.sort((phis + np.random.uniform(0, 2 * np.pi, len(phis)) + np.pi) % (2 * np.pi) - np.pi)
                        phisCorr = np.sort((phis + phiDeltaCorr[iP] + np.pi) % (2 * np.pi) - np.pi)
                        row[1] = find_average(phisUnif)
                        row[2] = find_average(phisCorr)
                        
                        w.writerow(row)
                    
                elif phis.size == 1:
                    size1 += 1
                    
                phiDeltaCorr = np.random.uniform(0, 2 * np.pi, nP)
                
                famSizeMax = np.maximum(famSizeMax, phis.size)
                famSizeMin = np.minimum(famSizeMin, phis.size)
                countGroups += 1
                currPos_mu = (x_mu,y_mu,z_mu)
                phis = np.array([])
            phis = np.append(phis, np.arctan2(y, x))
            # Converting position: cm -> mm
            pos = np.array([x, y, z], dtype=np.float64)

            # Inverting Z position/momentum (if requested)
            if args.invert_z:
                pos[2] *= -1
                mom[2] *= -1

print(f'Number of lines in input file: {nLines}')
print(f'There are {countGroups} groups of particles, max size {famSizeMax}, min size {famSizeMin}')

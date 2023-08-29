#!/usr/bin/env python

from ionize import *

from simtk.openmm import app
import simtk.openmm as omm
from simtk import unit
from numpy import diag
import itertools as it
import time
import sys
import argparse
from math import sqrt, acos, atan2, ceil
import re

import numpy as np
import HBstuff_1na2
from HBstuff_1na2 import *


def prev_and_next(iterable):
    prevs, items, nexts = it.tee(iterable, 3)
    prevs = it.chain([None], prevs)
    nexts = it.chain(it.islice(nexts, 1, None), [None])
    return zip(prevs, items, nexts)

def atm_index(res):
    #return res.atoms[0].index
    for atom in res.atoms():
        return atom.index


def build_by_seq(seq, number, box_size, forcefield):
    bl = 3.41*unit.angstrom
    Vec_PtoS=[1,2,3]*unit.angstrom #modify these values.
    Vec_PtoN=[4,5,6]*unit.angstrom

    name_map = {'A': 'ADE', 'C': 'CYT', 'G': 'GUA', 'U': 'URA'}

    ite = int(number**(1./3))
    print(ite)
    distance = box_size / ite
    num_add = ite**3
    topo = app.Topology()
    positions = []
    def get_atom(res,atomName):
        for atom in res.atoms():
            if atomName == "N":
                if atom.name != "P" and atom.name != "S":
                    return atom 
            else: 
                if atom.name == atomName:    
                    return atom

    Nrepeat = 1
    #### process sequence

    if seq.find('(') > -1 and seq.find(')') > -1 and seq.split(")", 1)[1].isdigit():
        Nrepeat = int(seq.split(")", 1)[1])
        newseq = re.search('\((.*)\)', seq)
        seq = newseq.group(1)
        #print seq, Nrepeat

    for xshift in range(ite):
        for yshift in range(ite):
            for zshift in range(ite):
                chain = topo.addChain()
                atoms = []
                curr_idx = -1
                for it in range(Nrepeat):
                    for i, resSymbol in enumerate(seq):
                        symbol = name_map[resSymbol]
                        #if (it == 0 and i == 0) or (it == Nrepeat - 1 and i == len(seq) - 1):
                        if (it == Nrepeat - 1 and i == len(seq) - 1):
                            symbol = symbol + "T"
                        if (it == 0 and i == 0):
                            symbol = symbol + "S"

                        res = topo.addResidue(symbol, chain)
                        atomP = forcefield._templates[symbol].atoms[0]
                        atomS = forcefield._templates[symbol].atoms[1]
                        atomN = forcefield._templates[symbol].atoms[2]
                        atoms.append(topo.addAtom(atomP.name, forcefield._atomTypes[atomP.type].element, res))
                        atoms.append(topo.addAtom(atomS.name, forcefield._atomTypes[atomS.type].element, res))
                        atoms.append(topo.addAtom(atomN.name, forcefield._atomTypes[atomN.type].element, res))
                        curr_idx += 1
                        positions.append([curr_idx*bl + xshift*distance, curr_idx*bl + yshift*distance, curr_idx*bl + zshift*distance])
                        positions.append([curr_idx*bl + xshift*distance + Vec_PtoS[0], curr_idx*bl + yshift*distance + Vec_PtoS[1], curr_idx*bl + zshift*distance + Vec_PtoS[2]])
                        positions.append([curr_idx*bl + xshift*distance + Vec_PtoN[0], curr_idx*bl + yshift*distance + Vec_PtoN[1], curr_idx*bl + zshift*distance + Vec_PtoN[2]])
                        print (curr_idx*bl + xshift*distance, curr_idx*bl + yshift*distance, curr_idx*bl + zshift*distance)

                for prev, item, nxt in prev_and_next(chain.residues()):
                    topo.addBond(get_atom(item,"P"), get_atom(item,"S"))
                    topo.addBond(get_atom(item,"S"), get_atom(item,"N"))
                
                    if prev != None:
                        topo.addBond(get_atom(prev,"S"), get_atom(item,"P"))

    return topo, positions

###################################################################
def AllAtom2CoarseGrain(pdb, forcefield):
    name_map = {'A': 'ADE', 'C': 'CYT', 'G': 'GUA', 'U': 'URA'}
    cg_topo = app.Topology()
    chain = cg_topo.addChain('X')
    beads = []
    cg_positions = []
    Nucleobase_indx = -1
    prevSbead = None

    for i, aa_res in enumerate(pdb.topology.residues()):
        resname = name_map[aa_res.name]
        if i == pdb.topology.getNumResidues() - 1:
            resname += "T"
        if i == 0:
            resname += "S"

        #print "Res %s %s" % (resname, aa_res.id)
        cg_res = cg_topo.addResidue(resname, chain, aa_res.id)

        Pcount = 0
        P_x = 0.*unit.angstrom
        P_y = 0.*unit.angstrom
        P_z = 0.*unit.angstrom

        Scount = 0
        S_x = 0.*unit.angstrom
        S_y = 0.*unit.angstrom
        S_z = 0.*unit.angstrom

        Ncount = 0
        N_x = 0.*unit.angstrom
        N_y = 0.*unit.angstrom
        N_z = 0.*unit.angstrom

        for aa_atom in aa_res.atoms():
            #print aa_atom.name
            if "P" in aa_atom.name and (not "H" in aa_atom.name):
                Pcount += 1
                P_x += pdb.positions[aa_atom.index][0]
                P_y += pdb.positions[aa_atom.index][1]
                P_z += pdb.positions[aa_atom.index][2]
            if "'" in aa_atom.name and (not "H" in aa_atom.name):
                Scount += 1
                S_x += pdb.positions[aa_atom.index][0]
                S_y += pdb.positions[aa_atom.index][1]
                S_z += pdb.positions[aa_atom.index][2]
            if (not "P" in aa_atom.name) and (not "'" in aa_atom.name) and (not "H" in aa_atom.name):
                Ncount += 1
                N_x += pdb.positions[aa_atom.index][0]
                N_y += pdb.positions[aa_atom.index][1]
                N_z += pdb.positions[aa_atom.index][2]
            else:
                continue

        P_x /= Pcount
        P_y /= Pcount
        P_z /= Pcount

        S_x /= Scount
        S_y /= Scount
        S_z /= Scount
        
        N_x /= Ncount
        N_y /= Ncount
        N_z /= Ncount
 
        element = None
        Nucleobasename = None

        if "ADE" in cg_res.name:
            element = forcefield._atomTypes["2"].element
            Nucleobasename = "A"
        elif "GUA" in cg_res.name:
            element = forcefield._atomTypes["3"].element
            Nucleobasename = "G"
        elif "CYT" in cg_res.name:
            element = forcefield._atomTypes["4"].element
            Nucleobasename = "C"
        elif "URA" in cg_res.name:
            element = forcefield._atomTypes["5"].element
            Nucleobasename = "U"

        beads.append(cg_topo.addAtom("P", forcefield._atomTypes["0"].element, cg_res))
        cg_positions.append([P_x, P_y, P_z])

        beads.append(cg_topo.addAtom("S", forcefield._atomTypes["1"].element, cg_res))
        cg_positions.append([S_x, S_y, S_z])
        
        beads.append(cg_topo.addAtom(Nucleobasename, element, cg_res))
        cg_positions.append([N_x, N_y, N_z])
	
        Nucleobase_indx += 3
        cg_topo.addBond(beads[Nucleobase_indx-2], beads[Nucleobase_indx-1]) # Adding the P-S bond	
        cg_topo.addBond(beads[Nucleobase_indx-1], beads[Nucleobase_indx]) # Adding the S-N bond	
 
        if prevSbead != None:
            cg_topo.addBond(beads[prevSbead], beads[Nucleobase_indx-2])  # Adding the S-P bond with the previous S.

        prevSbead = Nucleobase_indx-1 

    return cg_topo, cg_positions

KELVIN_TO_KT = unit.AVOGADRO_CONSTANT_NA * unit.BOLTZMANN_CONSTANT_kB / unit.kilocalorie_per_mole
#print KELVIN_TO_KT

parser = argparse.ArgumentParser(description='Coarse-grained simulation using OpenMM')
parser.add_argument('-p','--pdb', type=str, help='pdb structure')
parser.add_argument('-f','--sequence', type=str, help='input structure')
parser.add_argument('-C','--RNA_conc', type=float, default='10.',
                    help='RNA concentration (microM) [10.0]')
parser.add_argument('-K','--monovalent_concentration', type=float, default='100.',
                    help='Monovalent concentration (mM) [100.0]')
parser.add_argument('-v','--box_size', type=float, default='80.',
                    help='Box length (A) [80.0]')
parser.add_argument('-c','--cutoff', type=float, default='30.',
                    help='Electrostatic cutoff (A) [30.0]')
parser.add_argument('-H','--hbond_energy', type=float, default='2.22',
                    help='Hbond strength (kcal/mol) [2.22]')
parser.add_argument('-b','--hbond_file', type=str,
                    help='file storing tertiary Hbond')
parser.add_argument('-T','--temperature', type=float, default='20.',
                    help='Temperature (oC) [20.0]')
parser.add_argument('-t','--traj', type=str, default='md.dcd',
                    help='trajectory output')
parser.add_argument('-e','--energy', type=str, default='energy.out',
                    help='energy decomposition')
parser.add_argument('-o','--output', type=str, default='md.out',
                    help='status and energy output')
parser.add_argument('-x','--frequency', type=int, default='10000',
                    help='output and restart frequency')
parser.add_argument('-n','--step', type=int, default='10000',
                    help='Number of step [10000]')
parser.add_argument('-R','--restart', action='store_true',
                    help='flag to restart simulation')
parser.add_argument('-k','--chkpoint', type=str, default='checkpoint.xml',
                    help='initial xml state')
parser.add_argument('-r','--res_file', type=str, default='checkpnt.chk',
                    help='checkpoint file for restart')
parser.add_argument('-M','--Mprofiling', type=int, default='5',
                    help='M parameter for time profiling')
args = parser.parse_args()

class simu:    ### structure to group all simulation parameter
    box = 0.
    temp = 0.
    Kconc = 0.
    Nstep = 0
    cutoff = 0.
    epsilon = 0.
    #b = 4.38178046 * unit.angstrom / unit.elementary_charge
    b = 4.38178046 * unit.angstrom
    restart = False
#    list = None ### list cannot be initialized here!!

#simu.list = []
simu.box = args.box_size * unit.angstrom
Hbond_Uhb = args.hbond_energy*unit.kilocalorie_per_mole
simu.temp = (args.temperature + 273.15)*unit.kelvin
simu.Nstep = args.step
simu.cutoff = args.cutoff*unit.angstrom
simu.Kconc = args.monovalent_concentration
simu.restart = args.restart

#simu.epsilon = (87.740 - 0.4008 * args.temperature + 9.398 * 10**-4 * args.temperature**2 - 1.410 * 10**-6 * args.temperature**3) ;
#simu.l_Bjerrum = 0.01671* 10**7/(simu.epsilon*simu.temp/unit.kelvin) * unit.angstrom # the prefactor is e^2/(4pi*epsilon_0*kB) * 10^10 to convert to Angstrom.
#print("Bjerrum length  ", simu.l_Bjerrum)
#simu.Qe = simu.b  * unit.elementary_charge**2 / simu.l_Bjerrum
#print("Phosphate charge   ", -simu.Qe)
#simu.kappa = unit.sqrt (4*3.14159 * simu.l_Bjerrum * 2*simu.Kconc*6.022e-7 /unit.angstrom**3)
#print("kappa   ", simu.kappa)

forcefield = app.ForceField('/ocean/projects/bio220054p/ahassan1/XMLfiles/rna_TIS.xml')
topology = None
positions = None

if args.pdb != None:
    print("Reading PDB file ...")
    pdb = app.PDBFile(args.pdb)
    tmp_topology, tmp_positions = AllAtom2CoarseGrain(pdb, forcefield)
    #app.PDBFile.writeFile(topology, positions, open("beforeminimization.pdb", "w"), keepIds=True)
elif args.sequence != None:
    print("Building from sequence %s ..." % args.sequence)
    #N_RNA = args.RNA_conc * 6.022e-10 * args.box_size**3
    #N_RNA_added = (int(N_RNA**(1./3)))**3
    N_RNA_added = 1
    #real_conc = N_RNA_added / (6.022e-10 * args.box_size**3)
    real_conc = args.RNA_conc
    simu.box = (N_RNA_added / (real_conc * 6.022e-10))**(1./3) * unit.angstrom
    print("Box size    %f A" % (simu.box/unit.angstrom))
    print("Numbers added   %d ----> %f microM" % (N_RNA_added, real_conc))
    tmp_topology, tmp_positions = build_by_seq(args.sequence, N_RNA_added, simu.box, forcefield)
else:
    print("Need at least structure or sequence !!!")
    sys.exit()

tmp_topology.setPeriodicBoxVectors([[simu.box.value_in_unit(unit.nanometers),0,0], [0,simu.box.value_in_unit(unit.nanometers),0], [0,0,simu.box.value_in_unit(unit.nanometers)]])
tmp_system = forcefield.createSystem(tmp_topology)
######################################## FOR IONIZE: ################################################## 
simulation = app.Simulation(tmp_topology, tmp_system, omm.LangevinIntegrator(simu.temp, 0.5/unit.picosecond, 50*unit.femtoseconds))
simulation.context.setPositions(tmp_positions)
boxvector = diag([simu.box/unit.angstrom for i in range(3)]) * unit.angstrom
simulation.context.setPeriodicBoxVectors(*boxvector)
state = simulation.context.getState(getPositions=True)
app.PDBFile.writeFile(tmp_topology, state.getPositions(), open("input.pdb", "w"), keepIds=True)
ionized_inputFile=ionize("input.pdb","TIS")
######################################################################################################

### Overwrite previous definitions after ionizing the structure:
pdb = app.PDBFile(ionized_inputFile)
topology=pdb.topology
positions=pdb.positions
system=forcefield.createSystem(topology)

print(topology)
for atom in topology.atoms():
    print(atom)
for bond in topology.bonds():
    print(bond)

########## bond force
bondforce = omm.HarmonicBondForce()
for bond in topology.bonds():
    if bond[0].name == "P" and bond[1].name == "S":
        bondforce.addBond(bond[0].index, bond[1].index, 4.6010*unit.angstroms, 23.0*unit.kilocalorie_per_mole/(unit.angstrom**2))
    if bond[0].name == "S" and bond[1].name == "P":
        bondforce.addBond(bond[0].index, bond[1].index, 3.8157*unit.angstroms, 64.0*unit.kilocalorie_per_mole/(unit.angstrom**2))
    if bond[0].name == "S" and bond[1].name == "A":
        bondforce.addBond(bond[0].index, bond[1].index, 4.8515*unit.angstroms, 10.0*unit.kilocalorie_per_mole/(unit.angstrom**2))
    if bond[0].name == "S" and bond[1].name == "G":
        bondforce.addBond(bond[0].index, bond[1].index, 4.9659*unit.angstroms, 10.0*unit.kilocalorie_per_mole/(unit.angstrom**2))
    if bond[0].name == "S" and bond[1].name == "C":
        bondforce.addBond(bond[0].index, bond[1].index, 4.2738*unit.angstroms, 10.0*unit.kilocalorie_per_mole/(unit.angstrom**2))
    if bond[0].name == "S" and bond[1].name == "U":
        bondforce.addBond(bond[0].index, bond[1].index, 4.2733*unit.angstroms, 10.0*unit.kilocalorie_per_mole/(unit.angstrom**2))

bondforce.setUsesPeriodicBoundaryConditions(True)
bondforce.setForceGroup(0)
system.addForce(bondforce)

######### angle force
angleforce = omm.HarmonicAngleForce()

for chain in topology.chains():
    if chain.index == 0:  ## only the RNA chain not the ions chain
        for prev, item, nxt in prev_and_next(chain.residues()):
            for atom in item.atoms():  ## Intra-residue PSN angle
                if atom.name == "P":
                    Patom=atom
                elif atom.name == "S":
                    Satom=atom
                elif atom.name == "A":
                    Natom=atom
                    #theta0=1.9259
                    theta0=97.6*np.pi/180
                elif atom.name == "G":
                    Natom=atom
                    #theta0=1.9150
                    theta0=101.4*np.pi/180
                elif atom.name == "C":
                    Natom=atom
                    #theta0=1.9655
                    theta0=90.5*np.pi/180
                elif atom.name == "U":
                    Natom=atom
                    #theta0=1.9663
                    theta0=90.2*np.pi/180
            angleforce.addAngle(Patom.index, Satom.index, Natom.index, theta0*unit.radian, 5.0*unit.kilocalorie_per_mole/(unit.radians**2))
            if prev != None: 
                for atom in prev.atoms():
                    if atom.name == "P":
                        pre_P=atom
                    elif atom.name == "S":
                        pre_S=atom
                    elif atom.name == "A":
                        pre_N=atom
                        #theta0=1.7029
                        theta0=110.3*np.pi/180
                    elif atom.name == "G":
                        pre_N=atom
                        #theta0=1.7690
                        theta0=109.7*np.pi/180
                    elif atom.name == "C":
                        pre_N=atom
                        #theta0=1.5803
                        theta0=112.6*np.pi/180
                    elif atom.name == "U":
                        pre_N=atom
                        #theta0=1.5735
                        theta0=112.7*np.pi/180

                for item_atom in item.atoms():
                    if item_atom.name == "P":
                        item_P=item_atom
                    elif item_atom.name == "S":
                        item_S=item_atom
                angleforce.addAngle(pre_N.index, item_S.index, item_P.index, theta0*unit.radian, 5.0*unit.kilocalorie_per_mole/(unit.radians**2))  #NSP

                angleforce.addAngle(pre_P.index, pre_S.index, item_P.index, 1.4440*unit.radian, 20.0*unit.kilocalorie_per_mole/(unit.radians**2))  #PSP
                angleforce.addAngle(pre_S.index, item_P.index, item_S.index, 1.5256*unit.radian, 20.0*unit.kilocalorie_per_mole/(unit.radians**2)) #SPS

angleforce.setUsesPeriodicBoundaryConditions(True)
angleforce.setForceGroup(1)
system.addForce(angleforce)

######## Stacking interactions
Stack_expression="(-h+Kb*(T-Tm)*s)/(1 + kr*(r-r0)^2 + kphi1*(phi1-phi1_0+sel1)^2 + kphi2*(phi2-phi2_0+sel2)^2 ); sel1=select(x1,y1,0); sel2=select(x2,y2,0); x1=step(abs(phi1-phi1_0) - pi); y1= -2*pi*sgn1; sgn1= 2*step(phi1-phi1_0)-1; x2=step(abs(phi2-phi2_0) - pi); y2= -2*pi*sgn2; sgn2= 2*step(phi2-phi2_0)-1; r=distance(p1,p2); phi1=dihedral(p3,p4,p5,p6); phi2=dihedral(p7,p6,p5,p4);"
StackForce=omm.CustomCompoundBondForce(7,Stack_expression)
StackForce.addPerBondParameter("h");
StackForce.addPerBondParameter("s");
StackForce.addPerBondParameter("Tm");
StackForce.addPerBondParameter("r0");
StackForce.addPerBondParameter("phi1_0");
StackForce.addPerBondParameter("phi2_0");
StackForce.setUsesPeriodicBoundaryConditions(True)
StackForce.addGlobalParameter('kr', 1.4/unit.angstroms**2)
StackForce.addGlobalParameter('kphi1', 4.0/unit.radians**2)
StackForce.addGlobalParameter('kphi2', 4.0/unit.radians**2)
StackForce.addGlobalParameter('pi', 3.141592653*unit.radians)
StackForce.addGlobalParameter('Kb', 0.001987204259*unit.kilocalorie_per_mole)  # the unit is unit.kilocalorie_per_mole and not unit.kilocalorie_per_mole/unit.kelvin as T and Tm are used here unitless
StackForce.addGlobalParameter('T', args.temperature+273.15)  #unitless
T=args.temperature +273.15 #unitless

StackForce.addEnergyParameterDerivative('T')

def Stack_params(atom1,atom2,T):
    p1=atom1
    p2=atom2
                #(p1 , p2): [   h   ,   s   ,   Tm   ,  r0]    h in kilocal/mol, s unitless, Tm in Kelvin, r0 in Angstrom
    Params_dict={('A','A'): [3.99578, -0.319, 299.058, 4.18],
                 ('A','C'): [3.96314, -0.319, 299.058, 3.83],
                 ('A','G'): [4.75714,  5.301, 341.349, 4.43],
                 ('A','U'): [3.96314, -0.319, 299.058, 3.83],
                 ('C','A'): [3.92173, -0.319, 299.058, 4.70],
                 ('C','C'): [3.65726, -1.567, 285.968, 4.25],
                 ('C','G'): [4.23498,  0.774, 315.673, 4.98],
                 ('C','U'): [3.62896, -1.567, 285.968, 4.23],
                 ('G','A'): [4.71668,  5.301, 341.349, 4.01],
                 ('G','C'): [4.71997,  4.370, 343.363, 3.68],
                 ('G','G'): [5.19674,  7.346, 366.523, 4.24],
                 ('G','U'): [4.61901,  2.924, 338.329, 3.66],
                 ('U','A'): [3.92173, -0.319, 299.058, 4.70],
                 ('U','C'): [3.62234, -1.567, 285.968, 4.27],
                 ('U','G'): [4.67613,  2.924, 338.329, 5.0],
                 ('U','U'): [2.99569, -3.563, 251.733, 4.25],
                 }

    h,s,Tm,r0 = Params_dict[(p1.name,p2.name)] 
    h=h*unit.kilocalorie_per_mole  ## s is unitless and Tm is reported here unitless but is given in units of Kelvin.
    r0=r0*unit.angstroms

    phi1_0= -2.58684 * unit.radians
    phi2_0= 3.07135 * unit.radians

    #Kb=0.001987204259*unit.kilocalorie_per_mole/unit.kelvin # Units in kilocalorie_per_mole.kelvin
    #Ust0=-h+Kb*(T-Tm)*unit.kelvin*s 

    return [h,s,Tm,r0,phi1_0,phi2_0]

for chain in topology.chains():
    for prev, item, nxt in prev_and_next(chain.residues()):
        if prev == None or nxt == None:
            continue
        for prev_atom in prev.atoms():
            if prev_atom.name == "P":
                p3=prev_atom
            elif prev_atom.name == "S":
                p4=prev_atom
            elif prev_atom.name in {"A", "U", "G", "C"}:
                p1=prev_atom

        for item_atom in item.atoms():
            if item_atom.name == "P":
                p5=item_atom
            elif item_atom.name == "S":
                p6=item_atom
            elif item_atom.name in {"A", "U", "G", "C"}:
                p2=item_atom

        for nxt_atom in nxt.atoms():
            if nxt_atom.name == "P":     
                p7=nxt_atom
            
        group_add=[p1.index,p2.index,p3.index,p4.index,p5.index,p6.index,p7.index]
        Parameters = Stack_params(p1,p2,T)
        StackForce.addBond(group_add,Parameters)

StackForce.setForceGroup(2)
system.addForce(StackForce)

###################################################
#WCA force:
WCA_cutoff = 7*unit.angstroms

energy_function =  'step(D-r) * ep * ((R6 - 2)*R6 + 1)' #WCA
energy_function += '; R6=(Cone/(r+Cone-D))^6; ep=sqrt(ep1*ep2); D=select(I,D1+D2,Ctwo); I=I1*I2-1;' #Definitions .. 0.32 nm = 3.2 A and 0.16 nm = 1.6 A
WCAforce = omm.CustomNonbondedForce(energy_function)
WCAforce.addGlobalParameter("Cone", 1.6 *unit.angstroms)
WCAforce.addGlobalParameter("Ctwo", 3.2 *unit.angstroms)
WCAforce.addPerParticleParameter("I")
WCAforce.addPerParticleParameter("D")
WCAforce.addPerParticleParameter("ep")

def WCA_parameters(atom):
                 #name: [ isRNA,  Di   , ep_i  ]  #Di in Angstrom, and ep_i in kcal/mol
    params_dict={'P':  [  1,   2.1  ,   0.2],
                 'S':  [  1,   2.9  ,   0.2],
                 'A':  [  1,   2.8  ,   0.2],
                 'G':  [  1,   3.0  ,   0.2],
                 'C':  [  1,   2.7  ,   0.2],
                 'U':  [  1,   2.7  ,   0.2],
                 'Mg': [ 0,   0.7926,  0.894700],
                 'Ca': [ 0,   1.7131,  0.459789],
                 'Cl': [ 0,    1.948,  0.265000],
                 'K':  [ 0,   2.6580,  0.000328],
                 'Na': [ 0,    1.868,  0.002770],
                 }
    I,D,ep=params_dict[atom.name]
    D=D*unit.angstroms
    ep=ep*unit.kilocalorie_per_mole
    return [I,D,ep]

for atom in topology.atoms():
    parameters= WCA_parameters(atom)
    WCAforce.addParticle(parameters)

#bonds=[]
#for bond in topology.bonds():
#    bonds.append((bond[0].index,bond[1].index))
#WCAforce.createExclusionsFromBonds(bonds, 3)


for bond in topology.bonds():
    WCAforce.addExclusion(bond[0].index, bond[1].index)

num_angles = angleforce.getNumAngles()

# Iterate over all angles and retrieve the particle indices
for angle_index in range(num_angles):
    particle1, particle2, particle3, angle0, k = angleforce.getAngleParameters(angle_index)
    #WCAforce.addExclusion(particle1,particle2)
    #WCAforce.addExclusion(particle2,particle3)   ## exclude here only 1,3 atoms as all 1,2 and 2,3 atoms were excluded when looping over all bonds previously.
    WCAforce.addExclusion(particle1,particle3)

#for angleA_index in range(num_angles):  #Needs revision to include all dihedrals.
#    p1A,p2A,p3A,ang0A,kA= angleforce.getAngleParameters(angleA_index)
#    for angleB_index in range(num_angles):
#        if angleB_index > angleA_index:
#            p1B,p2B,p3B,ang0B,kB= angleforce.getAngleParameters(angleB_index)
#            if (p2A==p1B and p3A==p2B and p1A != p3B):  #exclude atoms 3 bonds apart.
#                WCAforce.addExclusion(p1A,p3B)        
#            #if (p2A==p3B and p3A==p2B and p1A != p1B):
#            #    WCAforce.addExclusion(p1A,p1B)



#for chain in topology.chains():
#    for prev, item, nxt in prev_and_next(chain.residues()):
#        if prev != None:
#            for atom in prev.atoms():
#                if atom.name == "P":
#                    pre_P=atom
#                elif atom.name == "S":
#                    pre_S=atom
#                elif atom.name in {"A","G","U","C"}:
#                    pre_N=atom
#            for atom in item.atoms():
#                if atom.name == "P":
#                    item_P=atom
#                if atom.name == "S":
#                    item_S=atom
#            WCAforce.addExclusion(pre_N.index,item_P.index)
#            WCAforce.addExclusion(pre_P.index,item_P.index)
#            WCAforce.addExclusion(pre_S.index,item_S.index)

WCAforce.setCutoffDistance(WCA_cutoff)
WCAforce.setForceGroup(3)
WCAforce.setNonbondedMethod(omm.CustomNonbondedForce.CutoffPeriodic)
WCAforce.setUseLongRangeCorrection(True)
system.addForce(WCAforce)

###################~~~~~~~~Add Electrostatics using PME~~~~~~~~###################

##READ1: Particle Charges are rescaled by sqrt(water dielectric constant): q_scaled = q /sqrt(water dielectric constant)##
##Default LJ Interactions are set to 0 ##

scale_factor = (87.740 - 0.4008 * args.temperature + 9.398 * 10**-4 * args.temperature**2 - 1.410 * 10**-6 * args.temperature**3) ;
CoulombForce = omm.NonbondedForce()
CoulombForce.setNonbondedMethod(omm.NonbondedForce.PME)
#CoulombForce.setNonbondedMethod(omm.NonbondedForce.CutoffPeriodic)
#CoulombForce.setNonbondedMethod(omm.NonbondedForce.NoCutoff)
CoulombForce.setCutoffDistance(0.3*simu.box)
CoulombForce.setEwaldErrorTolerance(0.005)
#CoulombForce.setUseDispersionCorrection(True)

def charge(atom):
    charge_dict={'P':  -1,
                 'S':  0,
                 'A':  0,
                 'G':  0,
                 'C':  0,
                 'U':  0,
                 'Mg': 2,
                 'Ca': 2,
                 'Cl': -1,
                 'K':  1,
                 'Na': 1,
                }
    q=charge_dict[atom.name]
    q=q*unit.elementary_charge
    return q

sigma=0*unit.angstroms
epsilon=0*unit.kilocalorie_per_mole
for atom in topology.atoms():
    q=charge(atom)/np.sqrt(scale_factor)
    CoulombForce.addParticle(q,sigma,epsilon) 

for bond in topology.bonds():
    CoulombForce.addException(bond[0].index, bond[1].index,0,0,0)

num_angles = angleforce.getNumAngles()

# Iterate over all angles and retrieve the particle indices
for angle_index in range(num_angles):
    particle1, particle2, particle3, angle0, k = angleforce.getAngleParameters(angle_index)
    #WCAforce.addExclusion(particle1,particle2)
    #WCAforce.addExclusion(particle2,particle3)   ## exclude here only 1,3 atoms as all 1,2 and 2,3 atoms were excluded when looping over all bonds previously.
    CoulombForce.addException(particle1,particle3,0,0,0)

#for angleA_index in range(num_angles):  #Needs revision to include all dihedrals
#    p1A,p2A,p3A,ang0A,kA= angleforce.getAngleParameters(angleA_index)
#    for angleB_index in range(num_angles):
#        if angleB_index > angleA_index:
#            p1B,p2B,p3B,ang0B,kB= angleforce.getAngleParameters(angleB_index)
#            if (p2A==p1B and p3A==p2B and p1A != p3B):  #exclude atoms 3 bonds apart.
#                CoulombForce.addException(p1A,p3B,0,0,0)        
#                print("Non bonded Exception: {},{},{},{}".format(p1A,p2A,p3A,p3B))
#            #if (p2A==p3B and p3A==p2B and p1A != p1B):
#            #    CoulombForce.addException(p1A,p1B,0,0,0)

#bonds=[]
#for bond in topology.bonds():
#    bonds.append((bond[0].index,bond[1].index))
#
#CoulombForce.createExceptionsFromBonds(bonds,0,0)

CoulombForce.setForceGroup(4)
system.addForce(CoulombForce)
#MTHD=CoulombForce.getNonbondedMethod()
#CHK=CoulombForce.usesPeriodicBoundaryConditions()

###################################################

list_donorGC = []
list_acceptorGC = []
list_donorAU = []
list_acceptorAU = []
list_donorGU = []
list_acceptorGU = []

for chain in topology.chains():
    for prev, item, nxt in prev_and_next(chain.residues()):
        if nxt != None:
            if "A" in item.name:
                list_donorAU.append(item.index)
            elif "G" in item.name:
                list_donorGC.append(item.index)
                list_donorGU.append(item.index)
            elif "C" in item.name:
                list_acceptorGC.append(item.index)
            elif "U" in item.name:
                list_acceptorAU.append(item.index)
                list_acceptorGU.append(item.index)

#def check_same_chain(id1, id2, topology):
#    check1 = 0
#    check2 = 0
#    for chain in topology.chains():
#        for res in chain.residues():
#            if res.index == id1:
#                check1 = 1
#            if res.index == id2:
#                check2 = 1
#        if (check1 or check2):
#            break
#    return (check1 and check2)

same_chain_list = []
for chain in topology.chains():
    for res1 in chain.residues():
        connect_list = []
        for res2 in chain.residues():
            if res1.index == res2.index:
                continue
            connect_list.append(res2.index)
        same_chain_list.append(connect_list)

#print same_chain_list

GCfactor=3
AUfactor=2
GUfactor=2

da_list_GC=[]
if (len(list_donorGC) > 0 and len(list_acceptorGC) > 0):
    for ind1, res1 in enumerate(list_donorGC):
        for ind2, res2 in enumerate(list_acceptorGC):
            if res2 in same_chain_list[res1] and abs(res1-res2) > 4:
                da_list_GC.append([[res1,res2],GCfactor])

da_list_AU=[]
if (len(list_donorAU) > 0 and len(list_acceptorAU) > 0):
    for ind1, res1 in enumerate(list_donorAU):
        for ind2, res2 in enumerate(list_acceptorAU):
            if res2 in same_chain_list[res1] and abs(res1-res2) > 4:
                da_list_AU.append([[res1,res2],AUfactor])

da_list_GU=[]
if (len(list_donorGU) > 0 and len(list_acceptorGU) > 0):
    for ind1, res1 in enumerate(list_donorGU):
        for ind2, res2 in enumerate(list_acceptorGU):
            if res2 in same_chain_list[res1] and abs(res1-res2) > 4:
                da_list_GU.append([[res1,res2],GUfactor])
print(da_list_GC)
print(da_list_AU)
print(da_list_GU)
print("Number of possible donor-acceptor GC pairs = {}".format(len(da_list_GC)))
print("Number of possible donor-acceptor AU pairs = {}".format(len(da_list_AU)))
print("Number of possible donor-acceptor GU pairs = {}".format(len(da_list_GU)))


kdist = 5.0/unit.angstrom**2
kang = 1.5/unit.radian**2
kdih = 0.15/unit.radian**2
HBcutoff = 1.8*unit.nanometers
HBcutoff_unitless=1.8 #nm
UHb=2.93168175*unit.kilocalorie_per_mole

GCdist0 = 5.6550*unit.angstrom
GCang1_0 = 2.8230*unit.radians
GCang2_0 = 2.4837*unit.radians
GCdih1_0 = 1.3902*unit.radians
GCdih2_0 = 0.7619*unit.radians
GCdih3_0 = 1.2174*unit.radians

AUdist0 = 5.8815*unit.angstrom
AUang1_0 = 2.7283*unit.radians
AUang2_0 = 2.5117*unit.radians
AUdih1_0 = 1.2559*unit.radians
AUdih2_0 = 0.9545*unit.radians
AUdih3_0 = 1.1747*unit.radians

GUdist0 = 5.8815*unit.angstrom
GUang1_0 = 2.7283*unit.radians
GUang2_0 = 2.5117*unit.radians
GUdih1_0 = 1.2559*unit.radians
GUdih2_0 = 0.9545*unit.radians
GUdih3_0 = 1.1747*unit.radians

Expression="D*UHb*factor*step(cf-distance(p1,p4))*exp(-1.0*(kdist*(r-dist0)^2 + kang*(ang1-ang1_0)^2 + kang*(ang2-ang2_0)^2 + kdih*(dih1-dih1_0+sel1)^2 + kdih*(dih2-dih2_0+sel2)^2 + kdih*(dih3-dih3_0+sel3)^2)); sel1=select(x1,y1,0); sel2=select(x2,y2,0); sel3=select(x3,y3,0); x1=step(abs(dih1-dih1_0) - pi); y1= -2*pi*sgn1; sgn1= 2*step(dih1-dih1_0)-1; x2=step(abs(dih2-dih2_0) - pi); y2= -2*pi*sgn2; sgn2= 2*step(dih2-dih2_0)-1; x3=step(abs(dih3-dih3_0) - pi); y3= -2*pi*sgn3; sgn3= 2*step(dih3-dih3_0)-1; r=distance(p1,p4); ang1=angle(p2,p1,p4); ang2=angle(p5,p4,p1); dih1=dihedral(p2,p1,p4,p5); dih2=dihedral(p3,p2,p1,p4); dih3=dihedral(p6,p5,p4,p1);"
SingleHBforce = omm.CustomCompoundBondForce(6,Expression)
SingleHBforce.addGlobalParameter('pi', 3.141592653*unit.radians)
SingleHBforce.addGlobalParameter('UHb', -UHb)
SingleHBforce.addGlobalParameter('kdist', kdist)
SingleHBforce.addGlobalParameter('kang', kang)
SingleHBforce.addGlobalParameter('kdih', kdih)
SingleHBforce.addGlobalParameter('cf', HBcutoff)
SingleHBforce.addPerBondParameter('D')
SingleHBforce.addPerBondParameter('factor')
SingleHBforce.addPerBondParameter('dist0')
SingleHBforce.addPerBondParameter('ang1_0')
SingleHBforce.addPerBondParameter('ang2_0')
SingleHBforce.addPerBondParameter('dih1_0')
SingleHBforce.addPerBondParameter('dih2_0')
SingleHBforce.addPerBondParameter('dih3_0')
SingleHBforce.setForceGroup(5)
SingleHBforce.setUsesPeriodicBoundaryConditions(True)

bondIndexDict={}
count=0

#da_list_GC=[[0,26],[1,25],[24,2],[4,22],[16,10]] # for hTR HP 1na2
#list_donorGC=[0,1,4,16,24]
#list_acceptorGC=[2,10,22,25,26]

for pair,factor in da_list_GC:
    d=pair[0]
    a=pair[1]
    p1=d*3+2  # Nucleobase bead
    p2=d*3+1  # Sugar bead
    p3=(d+1)*3   # next residue phostphate bead.
    
    p4=a*3+2  # Nucleobase bead
    p5=a*3+1  # Sugar bead
    p6=(a+1)*3   # next residue phostphate bead.

    group_add = [p1,p2,p3,p4,p5,p6]
    parameters= [0, factor, GCdist0, GCang1_0, GCang2_0, GCdih1_0, GCdih2_0, GCdih3_0]
    SingleHBforce.addBond(group_add, parameters)
    SingleHBforce.setBondParameters(count,group_add, parameters)
    #print(SingleHBforce.getBondParameters(count))
    bondIndexDict[(d,a)]=count
    count+=1

for pair,factor in da_list_AU:
    d=pair[0]
    a=pair[1]
    p1=d*3+2  # Nucleobase bead
    p2=d*3+1  # Sugar bead
    p3=(d+1)*3   # next residue phostphate bead.
    
    p4=a*3+2  # Nucleobase bead
    p5=a*3+1  # Sugar bead
    p6=(a+1)*3   # next residue phostphate bead.

    group_add = [p1,p2,p3,p4,p5,p6]
    parameters= [0, factor, AUdist0, AUang1_0, AUang2_0, AUdih1_0, AUdih2_0, AUdih3_0]
    SingleHBforce.addBond(group_add, parameters)
    SingleHBforce.setBondParameters(count,group_add, parameters)
    #print(SingleHBforce.getBondParameters(count))
    bondIndexDict[(d,a)]=count
    count+=1

for pair,factor in da_list_GU:
    d=pair[0]
    a=pair[1]
    p1=d*3+2  # Nucleobase bead
    p2=d*3+1  # Sugar bead
    p3=(d+1)*3   # next residue phostphate bead.
    
    p4=a*3+2  # Nucleobase bead
    p5=a*3+1  # Sugar bead
    p6=(a+1)*3   # next residue phostphate bead.

    group_add = [p1,p2,p3,p4,p5,p6]
    parameters= [0, factor, GUdist0, GUang1_0, GUang2_0, GUdih1_0, GUdih2_0, GUdih3_0]
    SingleHBforce.addBond(group_add, parameters)
    SingleHBforce.setBondParameters(count,group_add, parameters)
    #print(SingleHBforce.getBondParameters(count))
    bondIndexDict[(d,a)]=count
    count+=1

da_list_factor=da_list_GC + da_list_AU + da_list_GU
system.addForce(SingleHBforce)

totalforcegroup=6

########## Simulation ############
class EnergyReporter(object):
    def __init__ (self, file, reportInterval):
        self._out = open(file, 'w')
        self._reportInterval = reportInterval

    def __del__ (self):
        self._out.close()

    def describeNextReport(self, simulation):
        step = self._reportInterval - simulation.currentStep%self._reportInterval
        return (step, False, False, False, True)
        #return (step, position, velocity, force, energy)

    def report(self, simulation, state):
        energy = []
        self._out.write(str(simulation.currentStep))
        for i in range(totalforcegroup):
            state = simulation.context.getState(getEnergy=True, getParameterDerivatives=True, groups=2**i)
            energy = state.getPotentialEnergy() / unit.kilocalorie_per_mole
            derivative_energy = state.getEnergyParameterDerivatives() #/ (unit.kilocalorie_per_mole/unit.kelvin)
            self._out.write("  " + str(energy))
            self._out.write("  " + str(derivative_energy['T']))
        self._out.write("\n")
######################################################################################################
class MySimulation(app.Simulation):
    def __init__(self, topology, system, integrator, platform, HBforceobject, da_list_factor, HBcutoff_unitless, GRID, bondIndexDict, modify_frequency):
        super().__init__(topology, system, integrator, platform)
        self.HBforce = HBforceobject
        self.modify_frequency = modify_frequency
        self.da_list_factor=da_list_factor
        self.HBcutoff_unitless=HBcutoff_unitless
        self.NumShoots=None
        self.counter=0
        self.GRID=GRID
        self.bondIndexDict=bondIndexDict

        self.current_da_pairs=set()
    
    def Energy(self,Variables,factor):
        return Energy(Variables,factor)

    def compute_angle_atomPositions(self,p1,p2,p3):
        return compute_angle_atomPositions(p1,p2,p3)

    def compute_angle_bondVectors(self,v1,v2):
        return compute_angle_bondVectors(v1,v2)

    def compute_dihedral_atomPositions(self,p1,p2,p3,p4):
        return compute_dihedral_atomPositions(p1,p2,p3,p4)

    def compute_dihedral_bondVectors(self,v1,v2,v3):
        return compute_dihedral_bondVectors(v1,v2,v3)

    def MIC(self,deltaVector,lx,ly,lz):
        return MIC(deltaVector,lx,ly,lz)

    #def calc_Uij_fast(self,state,d_list, a_list, factor, cutoff):
    #    return calc_Uij(state,d_list, a_list, factor, cutoff)
    
    def calc_Uij(self,state,da_list_factor, cutoff, GRID):
        return calc_Uij(state,da_list_factor, cutoff, GRID)

    def new_Dij_nearGroundState(self,Uij):
        return new_Dij_nearGroundState(Uij)
    
    def Characterize_HB_network(self,Dict,Type):
        return Characterize_HB_network(Dict,Type)
    
    def set_HB_network(self,context,da_list_factor, HBcutoff_unitless, force, GRID, Old_Dij_pairs, bondIndexDict): 
        return set_HB_network(context,da_list_factor, HBcutoff_unitless, force, GRID, Old_Dij_pairs, bondIndexDict) 

    #def set_HB_network_fast(self,context,d_list, a_list, HBfactor,HBcutoff_unitless,force, GRID, Old_Dij_pairs, bondIndexDict):
    #    return set_HB_network_fast(context,d_list, a_list, HBfactor,HBcutoff_unitless,force, GRID, Old_Dij_pairs, bondIndexDict)

    def step(self, steps):
        self.NumShoots=int(steps/self.modify_frequency)
        while self.counter < self.NumShoots:
            t1=time.time()
            super().step(self.modify_frequency)
            t2=time.time()
            print("Single shoot simulation time = {}".format(t2-t1))
            t1=time.time()
            #print(self.current_da_pairs)

            self.current_da_pairs=self.set_HB_network(self.context,self.da_list_factor,self.HBcutoff_unitless,self.HBforce, self.GRID, self.current_da_pairs, self.bondIndexDict)
            
            #self.current_da_pairs_GC=self.set_HB_network_fast(self.context,self.d_list_GC, self.a_list_GC, self.GCfactor,self.HBcutoff_unitless,self.forceGC, self.GRID, self.current_da_pairs_GC, self.bondIndexDict_GC)
            
            #for i in range(10000):
            #    self.add_numbers(self.counter,5)
            #print("Numbers added = {}".format(self.add_numbers(self.counter,5)))
            
            t2=time.time()
            print("HB network update time = {}".format(t2-t1))
            self.counter+=1


## Import the functions as methods in your simulation class
#for function_name in function_module.__dict__:
#    if not function_name.startswith("__"):
#        print(function_name)
#        setattr(MySimulation, function_name, getattr(function_module, function_name))

###########################################################################################

boxvector = diag([simu.box/unit.angstrom for i in range(3)]) * unit.angstrom
boxvector_unitless = diag([(simu.box/10)/unit.angstrom for i in range(3)]) # in nm
GRID=Establish_Grid(boxvector_unitless,HBcutoff_unitless)

#myintegrator = MyCustomIntegrator(simu.temp, 0.5/unit.picosecond, 50*unit.femtoseconds, SingleHBforceGC, SingleHBforceAU, modify_frequency=100)
integrator = omm.LangevinIntegrator(simu.temp, 0.5/unit.picosecond, 2.5*unit.femtoseconds)
platform = omm.Platform.getPlatformByName('CUDA')

# Create the simulation with the system and integrator
simulation = MySimulation(topology, system, integrator, platform, SingleHBforce, da_list_factor, HBcutoff_unitless, GRID, bondIndexDict, modify_frequency=100)
simulation.context.setPositions(positions) 
simulation.context.setPeriodicBoxVectors(*boxvector)


print("HB network initializing ... ")
simulation.current_da_pairs=set_HB_network(simulation.context,da_list_factor, HBcutoff_unitless, SingleHBforce, GRID, set(),bondIndexDict)

#myintegrator.context = simulation.context

print('Minimizing ...')
Min_start=time.time()

simulation.minimizeEnergy(tolerance=0.1*unit.kilojoules_per_mole/unit.nanometer, maxIterations=0)
#simulation.minimizeEnergy(1*unit.kilocalorie_per_mole, 10000)
#simulation.context.setVelocitiesToTemperature(0)
#simulation.integrator.setStepSize(0.0001) # in picoseconds, 0.0001 ps = 0.1 fs.
#simulation.integrator.setTemperature(0)
#simulation.integrator.setFriction(0.5)
#simulation.step(1000000)
#simulation.context.setTime(0)  #reset step index to the beginning.
Min_end=time.time()
print("Minimization time = {}".format(Min_end-Min_start))

# Run the simulation
simulation.reporters.append(app.DCDReporter(args.traj, args.frequency))
simulation.reporters.append(app.StateDataReporter(args.output, args.frequency, step=True, potentialEnergy=True, temperature=True, remainingTime=True, totalSteps=simu.Nstep, separator='  '))
simulation.reporters.append(EnergyReporter(args.energy, args.frequency))
simulation.reporters.append(app.CheckpointReporter(args.res_file, int(args.frequency)*100))

print('Running ...')
#simulation.context.setVelocitiesToTemperature(simu.temp)
#simulation.integrator.setTemperature(simu.temp)
#simulation.integrator.setFriction(0.5)
#simulation.integrator.setStepSize(0.0025) #2.5 fs
t0=time.time()
simulation.step(simu.Nstep)
prodtime = time.time() - t0
print("Simulation time = %5.2f" % (prodtime))
print("Simulation speed: % .2e steps/day" % (86400*simu.Nstep/(prodtime)))
#####################

#
#integrator = omm.LangevinIntegrator(simu.temp, 0.5/unit.picosecond, 50*unit.femtoseconds)
#platform = omm.Platform.getPlatformByName('CUDA')
##properties = {'CudaPrecision': 'mixed'}
#
##simulation = app.Simulation(topology, system, myintegrator, platform)
##print(simulation.context.getIntegrator())
#
#simulation = app.Simulation(topology, system, integrator, platform)
##simulation = app.Simulation(modeller.topology, system, integrator, platform)
##simulation = app.Simulation(topology, system, integrator, platform, properties)
##simulation = app.Simulation(topology, system, integrator)
#
#
#if simu.restart == False:
#    simulation.context.setPositions(positions) 
#    #simulation.loadState(args.chkpoint)
#    boxvector = diag([simu.box/unit.angstrom for i in range(3)]) * unit.angstrom
#    simulation.context.setPeriodicBoxVectors(*boxvector)
#    #print(simulation.usesPeriodicBoundaryConditions())
#
#    #positions = simulation.context.getState(getPositions=True).getPositions()
#    #newpost = []
#    #for pos in positions:
#        #pos[0] = pos[0] * simu.box / (150*unit.nanometers)
#        #pos[1] = pos[1] * simu.box / (150*unit.nanometers)
#        #pos[2] = pos[2] * simu.box / (150*unit.nanometers)
#
#        #newpost.append([pos[0]*simu.box/(150*unit.nanometers), pos[1]*simu.box/(150*unit.nanometers), pos[2]*simu.box/(150*unit.nanometers)])
#
#    #simulation.context.setPositions(positions)
#    #simulation.context.setPositions(newpost)
#    #print "Initial energy   %f   kcal/mol" % (simulation.context.getState(getEnergy=True).getPotentialEnergy() / unit.kilocalorie_per_mole)
#
#    #parsingTime=time.time()-tstart
#    #print("Parsing time = %5.2f" % (parsingTime))
#
#################################################### HB network initialization:
#    print("HB network initializing ... ")
#    set_HB_network(simulation.context,da_list_GC,GCfactor,HBcutoff_unitless,SingleHBforceGC)
#    set_HB_network(simulation.context,da_list_AU,AUfactor,HBcutoff_unitless,SingleHBforceAU)
#
########################################################################################
#    
#    Min_start=time.time()
#    print('Minimizing ...')
#    simulation.minimizeEnergy(1*unit.kilocalorie_per_mole, 10000)
#    Min_end=time.time()
#    print("Minimization time = {}".format(Min_end-Min_start))
#
#    state = simulation.context.getState(getPositions=True)
#    app.PDBFile.writeFile(topology, state.getPositions(), open("input.pdb", "w"), keepIds=True)
#
#    simulation.context.setVelocitiesToTemperature(simu.temp)
#else:
#    print("Loading checkpoint ...")
#    simulation.loadCheckpoint(args.res_file)
#
#simulation.reporters.append(app.DCDReporter(args.traj, args.frequency))
#simulation.reporters.append(app.StateDataReporter(args.output, args.frequency, step=True, potentialEnergy=True, temperature=True, remainingTime=True, totalSteps=simu.Nstep, separator='  '))
#simulation.reporters.append(EnergyReporter(args.energy, args.frequency))
#simulation.reporters.append(app.CheckpointReporter(args.res_file, int(args.frequency)*100))
#
#print('Running ...')
#t0 = time.time()
#
######################## Shooting with Updating HB network ############################
#Freq_to_update_HB_network=100 # every 100 timesteps.
#Shoot=int(simu.Nstep/Freq_to_update_HB_network)
#for i in range(Shoot):
#    t1=time.time()
#    simulation.step(Freq_to_update_HB_network)
#    t2=time.time()
#    
#    set_HB_network(simulation.context,da_list_GC,GCfactor,HBcutoff_unitless,SingleHBforceGC)
#    set_HB_network(simulation.context,da_list_AU,AUfactor,HBcutoff_unitless,SingleHBforceAU)
#
#    t3=time.time()
#    #print(int(ManyParticleHBforce.getBondParameters(40)[1][0]))
#    #print(float(ManyParticleHBforce.getBondParameters(0)[1][1]))
#    #state=simulation.context.getState(getPositions=True, getVelocities=False, getForces=True, getEnergy=False, getParameters=False, getParameterDerivatives=False, getIntegratorParameters=False, enforcePeriodicBox=False, groups={6})
#    #print(state.getForces())
#    print("HB network update time = {}; Single shoot simulation time = {}".format(t3-t2,t2-t1))    
#
##simulation.step(simu.Nstep)
##simulation.saveState('checkpoint.xml')
#prodtime = time.time() - t0
#print("Simulation time = %5.2f" % (prodtime))
#print("Simulation speed: % .2e steps/day" % (86400*simu.Nstep/(prodtime)))

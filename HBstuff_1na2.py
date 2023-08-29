#!/usr/bin/env python

import numpy as np
import itertools as ite
import time
############################################################# UPDATING HB NETWORK ALGORITHM ##############################
factor=3
cutoff=1.8 #nm
UHb=2.22  #kilocalorie_per_mole
ground=-1.0*factor*UHb  #kilocalorie_per_mole
kB=0.0019872  ## Boltzmann constant in Kilocalorie/mol.K
#Temp=args.temperature+273.15  ## temperature in Kelvin without the omm unit.
Temp=1000000000
#Temp=293.15

def Energy(Variables,factor):
    d=Variables[0]
    ang1=Variables[1]
    ang2=Variables[2]
    dih1=Variables[3]
    dih2=Variables[4]
    dih3=Variables[5]
    
    ###### PARAMETERS:
    if factor == 3 :  ## for GC pairs:
        dist0  =   0.5655 #*unit.nm  
        ang1_0 =   2.8230 #*unit.radians
        ang2_0 =   2.4837 #*unit.radians
        dih1_0 =   1.3902 #*unit.radians
        dih2_0 =   0.7619 #*unit.radians
        dih3_0 =   1.2174 #*unit.radians
    elif factor == 2 :  ## for AU pairs and GU pairs:
        dist0  =   0.58815#*unit.nm
        ang1_0 =   2.7283 #*unit.radians
        ang2_0 =   2.5117 #*unit.radians
        dih1_0 =   1.2559 #*unit.radians
        dih2_0 =   0.9545 #*unit.radians
        dih3_0 =   1.1747 #*unit.radians

    kdist=500   #/nm**2  = 5 /Angstrom**2
    kang=1.5 # /radians**2
    kdih=0.15 #/radians**2
    
    UHb=2.93168175 # kilocalorie_per_mole

    if abs(dih1-dih1_0) < np.pi:
        sel1=0
    if (dih1-dih1_0) < -1*np.pi:
        sel1=2*np.pi
    if (dih1-dih1_0) > np.pi:
        sel1=-2*np.pi

    if abs(dih2-dih2_0) < np.pi:
        sel2=0
    if (dih2-dih2_0) < -1*np.pi:
        sel2=2*np.pi
    if (dih2-dih2_0) > np.pi:
        sel2=-2*np.pi

    if abs(dih3-dih3_0) < np.pi:
        sel3=0
    if (dih3-dih3_0) < -1*np.pi:
        sel3=2*np.pi
    if (dih3-dih3_0) > np.pi:
        sel3=-2*np.pi

    terms=(kdist*(d-dist0)**2 + kang*(ang1-ang1_0)**2 + kang*(ang2-ang2_0)**2 + kdih*(dih1-dih1_0+sel1)**2 + kdih*(dih2-dih2_0+sel2)**2 + kdih*(dih3-dih3_0+sel3)**2)
    #energy=factor*(-UHb)/(1.0+terms)
    energy=factor*(-UHb)*np.exp(-1.0*terms)

    return energy

def compute_angle_atomPositions(p1,p2,p3):
    v1 = p1-p2
    v2 = p3-p2

    norm1=np.linalg.norm(v1)
    norm2=np.linalg.norm(v2)
    dot=np.dot(v1,v2)
    return np.arccos(dot/(norm1*norm2))

def compute_angle_bondVectors(v1,v2):
    norm1=np.linalg.norm(v1)
    norm2=np.linalg.norm(v2)
    dot=np.dot(v1,v2)
    return np.arccos(dot/(norm1*norm2))

def compute_dihedral_atomPositions(p1,p2,p3,p4):
    v1=p2-p1
    v2=p3-p2
    v3=p4-p3

    n1=np.cross(v1,v2)
    n2=np.cross(v2,v3)
   
    return np.arctan2(np.dot(v1,n2)*np.linalg.norm(v2), np.dot(n1,n2))

def compute_dihedral_bondVectors(v1,v2,v3):
    n1=np.cross(v1,v2)
    n2=np.cross(v2,v3)
   
    return np.arctan2(np.dot(v1,n2)*np.linalg.norm(v2), np.dot(n1,n2))


def MIC(deltaVector,lx,ly,lz):
    #lx,ly,lz are the lengths of the box sides.
    #MICVector=np.array([None,None,None])
    MICVector=deltaVector

    if deltaVector[0] >= lx/2:
        MICVector[0] = deltaVector[0]-lx
    if deltaVector[1] >= ly/2:
        MICVector[1] = deltaVector[1]-ly
    if deltaVector[2] >= lz/2:
        MICVector[2] = deltaVector[2]-lz

    if deltaVector[0] < -lx/2:
        MICVector[0] = deltaVector[0]+lx
    if deltaVector[1] < -ly/2:
        MICVector[1] = deltaVector[1]+ly
    if deltaVector[2] < -lz/2:
        MICVector[2] = deltaVector[2]+lz

    return MICVector


def Establish_Grid(boxVectors,cutoff):
    #boxVectors is a 3*3 matrix of the 3 rectangular lattice vectors. 
    ## Establish Grid:
    lx=np.linalg.norm(boxVectors[0])
    ly=np.linalg.norm(boxVectors[1])
    lz=np.linalg.norm(boxVectors[2])

    tmp_cell_dim=cutoff*1.0
    NumX=int(np.ceil(lx/tmp_cell_dim))
    NumY=int(np.ceil(ly/tmp_cell_dim))
    NumZ=int(np.ceil(lz/tmp_cell_dim))
    
    cell_dimX=lx/NumX
    cell_dimY=ly/NumY
    cell_dimZ=lz/NumZ

    GRID=[[NumX,NumY,NumZ],[cell_dimX,cell_dimY,cell_dimZ]]

    return GRID


def calc_Uij_fast(state,d_list,a_list,factor,cutoff,GRID):
    ## Implements Grid cell method for fast computation:
    Energy_dict={}
    positions=np.array(state.getPositions(asNumpy=True))
    lx=GRID[0][0]*GRID[1][0]
    ly=GRID[0][1]*GRID[1][1]
    lz=GRID[0][2]*GRID[1][2]

    def assign_to_Grid_cells(positions,GRID,d_list,a_list):
        NumX=GRID[0][0]
        NumY=GRID[0][1]
        NumZ=GRID[0][2]

        cell_dimX=GRID[1][0]
        cell_dimY=GRID[1][1]
        cell_dimZ=GRID[1][2]

        donor_Grid=[[ [set() for k in range(NumZ)] for j in range(NumY)] for i in range(NumX)]
        acceptor_Grid=[[ [set() for k in range(NumZ)] for j in range(NumY)] for i in range(NumX)]

        for index in d_list:
            pos=positions[index]
            ##cell indices:
            i=int(np.floor(pos[0]/cell_dimX)%NumX)
            j=int(np.floor(pos[1]/cell_dimY)%NumY)
            k=int(np.floor(pos[2]/cell_dimZ)%NumZ)   ### This makes i,j,k insensitive to the origin of the coordinates, but it doesn't mean that the 0 and NumX cells are the peripheral cells in the box .. similarly for y and z directions.

            donor_Grid[i][j][k].add(index)

        for index in a_list:
            pos=positions[index]
            ##cell indices:
            i=int(np.floor(pos[0]/cell_dimX)%NumX)
            j=int(np.floor(pos[1]/cell_dimY)%NumY)
            k=int(np.floor(pos[2]/cell_dimZ)%NumZ)   ### This makes i,j,k insensitive to the origin of the coordinates, but it doesn't mean that the 0 and NumX cells are the peripheral cells in the box .. similarly for y and z directions.

            acceptor_Grid[i][j][k].add(index)
        
        return donor_Grid,acceptor_Grid

    def generate_sub_da_list(donor_Grid,acceptor_Grid,GRID):
        sub_da_list=[]

        NumX=GRID[0][0]
        NumY=GRID[0][1]
        NumZ=GRID[0][2]

        for i in range(NumX):
            nxt_i=(i+1)%NumX
            pre_i=(i-1)%NumX
            for j in range(NumY):
                nxt_j=(j+1)%NumY
                pre_j=(j-1)%NumY
                for k in range(NumZ):
                    nxt_k=(k+1)%NumZ
                    pre_k=(k-1)%NumZ
                    if len(donor_Grid[i][j][k]) != 0:
                        acceptors=set().union(*[acceptor_Grid[k][l][m] for k,l,m in ite.product({i,nxt_i,pre_i},{j,nxt_j,pre_j},{k,nxt_k,pre_k})])
                        for d_index in donor_Grid[i][j][k]:
                            for a_index in acceptors:
                                if abs(d_index - a_index) > 4:   ## exclude nearby bases
                                    sub_da_list.append([d_index,a_index])
        return sub_da_list

    
    #### MAIN:
    #t1=time.time()
    donor_Grid, acceptor_Grid = assign_to_Grid_cells(positions,GRID,d_list,a_list)
    #print("Length of donor_Grid = {}, Length of acceptor_Grid = {}".format(sum(len(s) for sub_arr in donor_Grid for inner_arr in sub_arr for s in inner_arr),sum(len(s) for sub_arr in acceptor_Grid for inner_arr in sub_arr for s in inner_arr)))
    sub_da_list = generate_sub_da_list(donor_Grid,acceptor_Grid,GRID) 
    #t2=time.time()
    #print("Length of sub_da_list = {}".format(len(sub_da_list)))
    #print("Grid assignment and sub_da_list formation takes time: {}".format(t2-t1))
    #print(donor_Grid)
    #print(acceptor_Grid)
    #print(sub_da_list)

    for pair in sub_da_list:
        d=pair[0]
        a=pair[1]
        p1=d*3+2  # Nucleobase bead
        p2=d*3+1  # Sugar bead
        p3=(d+1)*3   # next residue phostphate bead.

        p4=a*3+2  # Nucleobase bead
        p5=a*3+1  # Sugar bead
        p6=(a+1)*3   # next residue phostphate bead.

        p1_p4 = MIC(positions[p4]-positions[p1],lx,ly,lz)
        dist=np.linalg.norm(p1_p4)
        if dist < cutoff:
            #t3=time.time()
            p1_p2= MIC(positions[p2]-positions[p1],lx,ly,lz)
            p4_p5= MIC(positions[p5]-positions[p4],lx,ly,lz)
            p4_p1= MIC(positions[p1]-positions[p4],lx,ly,lz)
            p3_p2= MIC(positions[p2]-positions[p3],lx,ly,lz)
            p6_p5= MIC(positions[p5]-positions[p6],lx,ly,lz)
            #t4=time.time()
            #print("MIC stuff takes time: {}".format(t4-t3))
 
            #t5=time.time()
            ang1=compute_angle_bondVectors(p1_p2,p1_p4)
            ang2=compute_angle_bondVectors(p4_p5,-p1_p4)
            dih1=compute_dihedral_bondVectors(-p1_p2,p1_p4,p4_p5)
            dih2=compute_dihedral_bondVectors(p3_p2,-p1_p2,p1_p4)
            dih3=compute_dihedral_bondVectors(p6_p5,-p4_p5,-p1_p4)

            variables=[dist,ang1,ang2,dih1,dih2,dih3]
            energy=Energy(variables,factor)
            #t6=time.time()
            #print("Variables and energy calculation took time: {}".format(t6-t5))
      
            if energy <-0.1:
                Energy_dict[(d,a)]=float('%.2f' % energy)

    return Energy_dict

def calc_Uij(state,da_list_factor, cutoff, GRID):
    ### The factor is 2 if AU and 3 if GC
    positions=np.array(state.getPositions(asNumpy=True))
    lx=GRID[0][0]*GRID[1][0]
    ly=GRID[0][1]*GRID[1][1]
    lz=GRID[0][2]*GRID[1][2]

    def get_dictionaries(positions,da_list_factor, cutoff,lx,ly,lz):
      #  dist_dict={} # dictionary
        
      #  angle1_dict={}
      #  angle2_dict={}
      #  angle3_dict={}
      #  angle4_dict={}
      #  
      #  dihed1_dict={}
      #  dihed2_dict={}
        
        Energy_dict={}        
        for pair,factor in da_list_factor:
            d=pair[0]
            a=pair[1]
            p1=d*3+2  # Nucleobase bead
            p2=d*3+1  # Sugar bead
            p3=(d+1)*3   # next residue phostphate bead.

            p4=a*3+2  # Nucleobase bead
            p5=a*3+1  # Sugar bead
            p6=(a+1)*3   # next residue phostphate bead.

            p1_p4 = MIC(positions[p4]-positions[p1],lx,ly,lz)
            dist=np.linalg.norm(p1_p4)
            if dist < cutoff:
                #t3=time.time()
                p1_p2= MIC(positions[p2]-positions[p1],lx,ly,lz)
                p4_p5= MIC(positions[p5]-positions[p4],lx,ly,lz)
                p4_p1= MIC(positions[p1]-positions[p4],lx,ly,lz)
                p3_p2= MIC(positions[p2]-positions[p3],lx,ly,lz)
                p6_p5= MIC(positions[p5]-positions[p6],lx,ly,lz)
                #t4=time.time()
                #print("MIC stuff takes time: {}".format(t4-t3))

                #t5=time.time()
                ang1=compute_angle_bondVectors(p1_p2,p1_p4)
                ang2=compute_angle_bondVectors(p4_p5,-p1_p4)
                dih1=compute_dihedral_bondVectors(-p1_p2,p1_p4,p4_p5)
                dih2=compute_dihedral_bondVectors(p3_p2,-p1_p2,p1_p4)
                dih3=compute_dihedral_bondVectors(p6_p5,-p4_p5,-p1_p4)

                variables=[dist,ang1,ang2,dih1,dih2,dih3]
                energy=Energy(variables,factor)
                #t6=time.time()
                #print("Variables and energy calculation took time: {}".format(t6-t5))

               # angle1_dict[(d,a)]=ang1
               # angle2_dict[(d,a)]=ang2
               # angle3_dict[(d,a)]=ang3
               # angle4_dict[(d,a)]=ang4
               # dihed1_dict[(d,a)]=dih1
               # dihed2_dict[(d,a)]=dih2

                #Energy_dict[(d,a)]=energy
                if energy <-0.1:
                    Energy_dict[(d,a)]=float('%.2f' % energy)
            #else:
            #    Energy_dict[(d,a)]=float('%.2f' % 0)       ### This makes the dictionary populated only with non zero terms. For fast computation.

        return Energy_dict #, dist_dict, angle1_dict, angle2_dict, angle3_dict, angle4_dict, dihed1_dict, dihed2_dict
                
    ##### MAIN:
   
    return get_dictionaries(positions,da_list_factor, cutoff, lx,ly,lz)
    

def new_Dij_MC(da_list, Uij, ground, kBT):
    ## ground=-1.0*UHb*factor
    Dij={}

    ################ Randomize the order of updating the HB network:
    #start_point=int(np.random.random()*len(Uij))
    #shifted_da_list=da_list[start_point:]+da_list[:start_point]  #shifts the da_list so that the starting point is the first element in shifted_da_list
    
    def generate_random_list(input_list):
        # Create a copy of the input list
        random_list = input_list[:]
    
        # Shuffle the elements in the random_list
        np.random.shuffle(random_list)
    
        return random_list

    shuffled_da_list=generate_random_list(da_list)

    #### Product vectors for donors and acceptors: ## Introduced to memoize the algorithm and make it O(N^2) instead of O(N^3). This happens by removing one for loop.
    Product_donor={}
    Product_acceptor={}

    #for pair in shifted_da_list:
    for pair in shuffled_da_list:   ## After shuffling its elements.
        d=pair[0]
        a=pair[1]    
        D=1 # initial definition for Dij
        if d not in Product_donor.keys():
            Product_donor[d]=1  #initial definition for Product_donor[d]
        if a not in Product_acceptor.keys():
            Product_acceptor[a]=1  #initial definition for Product_acceptor[a]

        #### Global condition:
        D=D*Product_donor[d]*Product_acceptor[a]
        #for key in {k for k in Dij.keys() if (k[0]==d or k[1]==a)}:
        #    D=D*(1-Dij[key])
        #### Local condition:
        if D==0:
            Dij[(d,a)]=0
        else:
            if Uij[(d,a)]==0: 
                Dij[(d,a)]=0
                #continue
            else:
                bolt=np.exp(-1.0*(Uij[(d,a)]-ground)/kBT)
                D=D*np.heaviside(bolt-np.random.random(),0)
                Dij[(d,a)]=int(D)

        ####Update the product values:
        Product_donor[d]*=(1-Dij[(d,a)])
        Product_acceptor[a]*=(1-Dij[(d,a)])

    return Dij


def new_Dij_nearGroundState(Uij):
    Dij={}

    def generate_random_list(input_list):
        # Create a copy of the input list
        random_list = input_list[:]
    
        # Shuffle the elements in the random_list
        np.random.shuffle(random_list)
    
        return random_list

    ################### Search for the lowest-energy bond for a specific donor:
    def get_min_Energy_bond_for_residue(Uij, Residue):
        resnum=Residue[0]
        restype=Residue[1]
        if restype == "donor":
            d_or_a=0
        if restype == "acceptor":
            d_or_a=1
        
        bonds = {key: value for key, value in Uij.items() if key[d_or_a] == resnum}
        
        if bonds:  ## True if donor_bonds are in the Uij dictionary
            min_key = min(bonds, key=bonds.get)
            return min_key, Uij[min_key]
        else:
            return None
    ########################
    #### MAIN:
    
#    for pair in da_list: 
#        d=pair[0]
#        a=pair[1]
#        DefinitelyNotbond=(d,a)
#        if DefinitelyNotbond not in Uij.keys():   #DefinitelyNotbond are pairs with Uij=0 which won't be included in the Uij dictionary for fast computation. SEE Energy function.
#            Dij[DefinitelyNotbond]=0

    donors_in_Uij=set(key[0] for key in Uij.keys())
    acceptors_in_Uij=set(key[1] for key in Uij.keys())

    #donors_list=list(np.unique([k[0] for k in da_list]))  # the resulting list will be ordered ascendingly.
    #acceptors_list=list(np.unique([k[1] for k in da_list]))
    Donors=[]
    Acceptors=[]
    #for d in donors_list:
    for d in donors_in_Uij:
        Donors.append([d,"donor"])
    #for a in acceptors_list:
    for a in acceptors_in_Uij:
        Acceptors.append([a,"acceptor"])
 
    Residues=Donors+Acceptors
    Shuffled_Residues_list=generate_random_list(Residues)

    for Res in Shuffled_Residues_list:
        resnum=Res[0]
        restype=Res[1]
        if restype == "donor":
            d_or_a=0
        if restype == "acceptor":
            d_or_a=1

        #print(Res)
        Remaining_residues=list({key[0] for key in Uij.keys()}.union({key[1] for key in Uij.keys()}))
        #print("Remaining residues in Uij: {}".format(Remaining_residues))
        if resnum in Remaining_residues:
            bond, minE = get_min_Energy_bond_for_residue(Uij,Res)
            #print(bond)
            #print(minE)
            donor=bond[0]  
            acceptor=bond[1]  ## either the donor or the acceptor is the same as Res.
            if minE !=0:
                Dij[bond]=1
                del Uij[bond]
                notbond_set={key for key in Uij.keys() if (key[0]==donor or key[1]==acceptor)}  #This nullifies all elements in the same row AND column as the chosen element ... amounts for the global condition. applied when the bond is formed.
            else:
                Dij[bond]=0
                del Uij[bond]
                notbond_set={key for key in Uij.keys() if (key[d_or_a]==resnum)} #This means that all elements on the same row OR column are zero so we nullify the correspoing Dijs.
            
            #print("notbond_set = {}".format(notbond_set))
            for notbond in notbond_set: 
                #if (notbond[0]==donor or notbond[1]==acceptor):    #this applies the global condition on Dij matrix. No need for products this way.     
                Dij[notbond]=0
                del Uij[notbond]

        else:
            continue

        #print("Uij = {}".format(Uij))        
    return Dij
    #################################################################
    #################################################################
    ################ Randomize the order of updating the HB network:
#    #start_point=int(np.random.random()*len(Uij))
#    #shifted_da_list=da_list[start_point:]+da_list[:start_point]  #shifts the da_list so that the starting point is the first element in shifted_da_list
#    
#    shuffled_donors_list=generate_random_list(donors_list)
#    shuffled_acceptors_list=generate_random_list(acceptors_list)
#
#    #### Product vectors for donors and acceptors: ## Introduced to memoize the algorithm and make it O(N^2) instead of O(N^3). This happens by removing one for loop.
#    Product_donor={}
#    Product_acceptor={}
#
#    done=0
#
#
#
#
#    for d in shuffled_donors_list:   ## After shuffling its elements.
#        if d not in Product_donor.keys():
#            Product_donor[d]=1  #initial definition for Product_donor[d]
#        #### Local-semiGlobal condition:
#        bond, energy = get_min_Energy_bond_for_residue(Uij,d,"donor")
#        if energy<0:
#	    Dij[bond]=1
#        if energy==0:
#            Dij[bond]=0
#        a_bond=bond[1]
#        for a in shuffled_acceptors_list:
#            if a!=a_bond:
#               Dij[(d,a)]=0 
#        
#            if a not in Product_acceptor.keys():
#                Product_acceptor[a]=1  #initial definition for Product_acceptor[a]
#
############# This method could miss some HBs, as if 2 donors has the same acceptor as the minimum HB energy, the acceptor will form a HB with the first sampled donor and the second donor won't have a HB. But it should have a HB with the acceptor showing the second least HB. (as the lowest energy bond is already taken) ... 
#
#        #### Global condition:
#        D=D*Product_donor[d]*Product_acceptor[a]
#        #for key in {k for k in Dij.keys() if (k[0]==d or k[1]==a)}:
#        #    D=D*(1-Dij[key])
#        
#
#        ####Update the product values:
#        Product_donor[d]*=(1-Dij[(d,a)])
#        Product_acceptor[a]*=(1-Dij[(d,a)])
#
#        done+=1
#        if done%10000==0:
#            print("Done Dij for {} pairs".format(done))
#    
#    return Dij

def Conv_Dij_Uij(Dij,Uij):
    conv={}
    for key in Dij.keys():
        con=Dij[key]*Uij[key]
        conv[key]=con
    return conv

def Characterize_HB_network(Dict,Type):
    donors_list=list(np.unique([k[0] for k in Dict.keys()]))  # the resulting list will be ordered ascendingly.
    acceptors_list=list(np.unique([k[1] for k in Dict.keys()]))
    n = len(donors_list)    # gets the number of unique donors in the dictionary. Equals the number of donors in da_list
    m = len(acceptors_list)    
    
    # Create an n x m matrix with all elements initialized to zero
    matrix = [[0] * m for a in range(n)]
    
    # Set the matrix elements based on the dictionary values .. this way the row indices and column indices are ordered based on the residue number in the topology.
    NumHB=0
    for k, v in Dict.items():
        i=donors_list.index(k[0])
        j=acceptors_list.index(k[1])
        matrix[i][j] = v
        NumHB+=v        

    ##Print the matrix
    print("{} matrix: (ij element: bond between ith donor and jth acceptor when they are ordered ascendingly)".format(Type))
    print("Donors on rows: {}".format(donors_list))
    print("Acceptors on columns: {}".format(acceptors_list))
    for row in matrix:
        print(row)
    if Type == "Dij":
        print("Number of HBs = {}".format(NumHB))
    if Type == "Uij":
        print("Total multiple HB energy = {}".format(NumHB))
    if Type == "Conv":
        print("Total single HB energy = {}".format(NumHB))

    print("############################################################################")
    print("############################################################################")
    print("############################################################################")

def set_HB_network(Context,da_list_factor, HBcutoff,HBforceObject, GRID, Old_Dij_pairs, bondIndexDict): #,MODEL):
    
    state = Context.getState(getPositions=True)
    Uij=calc_Uij(state, da_list_factor, HBcutoff, GRID)
    Characterize_HB_network(Uij,"Uij")
    #Dij=new_Dij_MC(da_list, Uij, HBstuff.ground, HBstuff.kB*HBstuff.Temp)
    CompleteUij=Uij.copy()   ### This is to conserve the dictionary as the Uij dictionary gets modified a lot in the new_Dij_nGS function.
    Dij=new_Dij_nearGroundState(Uij)
    Characterize_HB_network(Dij,"Dij")

    Dij_pairs=set(Dij.keys())
#
#    #print(Dij)
#    #print(CompleteUij)
    Conv=Conv_Dij_Uij(Dij,CompleteUij)
    Characterize_HB_network(Conv,"Conv")    

################
    pairs_only_in_Old_Dij = Old_Dij_pairs - Dij_pairs  ### These elements will be set to zero in Dij

    for pair in pairs_only_in_Old_Dij:
        d=pair[0]
        a=pair[1]
        #p1=d*3+2  # Nucleobase bead
        #p2=d*3+1  # Sugar bead
        #p3=(d+1)*3   # next residue phostphate bead.
        #p4=a*3+2  # Nucleobase bead
        #p5=a*3+1  # Sugar bead
        #p6=(a+1)*3   # next residue phostphate bead.
        index=bondIndexDict.get((d,a))
        bond_particles, bond_parameters = HBforceObject.getBondParameters(index)
        other_params=list(bond_parameters)[1:]
        HBforceObject.setBondParameters(index, bond_particles, [0]+other_params)
        #print("Done pair: {}".format(pair))

    for pair in Dij.keys():
        d=pair[0]
        a=pair[1]
        #p1=d*3+2  # Nucleobase bead
        #p2=d*3+1  # Sugar bead
        #p3=(d+1)*3   # next residue phostphate bead.
        #p4=a*3+2  # Nucleobase bead
        #p5=a*3+1  # Sugar bead
        #p6=(a+1)*3   # next residue phostphate bead.
        index=bondIndexDict.get((d,a))
        bond_particles, bond_parameters = HBforceObject.getBondParameters(index)
        other_params=list(bond_parameters)[1:]
        HBforceObject.setBondParameters(index, bond_particles, [Dij[(d,a)]]+other_params)
        #print("Done pair: {}".format(pair))

    HBforceObject.updateParametersInContext(Context)
    #print("DONE update in context !!!!")
    #print("Dij pairs = {}".format(Dij_pairs))
    return Dij_pairs ## Old_Dij_pairs feeded back into the set_HB_network method to compare to the new Dij quickly and update only the different elements.
#################

#    for bond_index in range(HBforceObject.getNumBonds()):
#        bond_particles, bond_parameters = HBforceObject.getBondParameters(bond_index)
#        bond_params=list(bond_parameters)
#        da=list(bond_particles)[:2] # donor and acceptor indices are the first 2 elements.
#        bond_params[0]=Dij[(da[0],da[1])]
#        HBforceObject.setBondParameters(bond_index, bond_particles, bond_params)
#        HBforceObject.updateParametersInContext(Context)


################
   # if MODEL == "SIS":
   #     for pair in da_list:   #### The way bonds are defined in this case depends on the exact nature of the force expression, so it is a 6-body term in this case.
   #         d=pair[0]
   #         a=pair[1]
   #         dp=d-1; ap=a-1; dn=d+1; an=a+1;
   #         HBforceObject.setBondParameters(count,[d,a,dp,ap,dn,an],[Dij[(d,a)]])
   #         #print("Done pair: {}".format(pair))
   #     HBforceObject.updateParametersInContext(Context)

   # if MODEL == "TIS":
   #     count=0
   #     for pair in da_list:   #### The way bonds are defined in this case depends on the exact nature of the force expression, so it is a 6-body term in this case.
   #         d=pair[0]
   #         a=pair[1]
   #         dB=d*3+2
   #         aB=a*3+2
   #         dBp=dB-1; aBp=aB-1; dBn=dB+1; aBn=aB+1;
   #         HBforceObject.setBondParameters(count,[dB,aB,dBp,aBp,dBn,aBn],[Dij[(d,a)]])
   #         count+=1
   #         #print("Done pair: {}".format(pair))
   #     HBforceObject.updateParametersInContext(Context)
  


def set_HB_network_fast(Context,d_list, a_list, HBfactor,HBcutoff,HBforceObject,GRID, Old_Dij_pairs, bondIndexDict): #,MODEL):

    state = Context.getState(getPositions=True)
    Uij=calc_Uij_fast(state,d_list,a_list,HBfactor,HBcutoff,GRID)
    Characterize_HB_network(Uij,"Uij")
    CompleteUij=Uij.copy()   ### This is to conserve the dictionary as the Uij dictionary gets modified a lot in the new_Dij_nGS function.
    Dij=new_Dij_nearGroundState(Uij)
    Characterize_HB_network(Dij,"Dij")

    Conv=Conv_Dij_Uij(Dij,CompleteUij)
    Characterize_HB_network(Conv,"Conv")    

    Dij_pairs=set(Dij.keys())


    pairs_only_in_Old_Dij = Old_Dij_pairs - Dij_pairs  ### These elements will be set to zero in Dij

    for pair in pairs_only_in_Old_Dij:
        d=pair[0]
        a=pair[1]
        p1=d*3+2  # Nucleobase bead
        p2=d*3+1  # Sugar bead
        p3=(d+1)*3   # next residue phostphate bead.
        p4=a*3+2  # Nucleobase bead
        p5=a*3+1  # Sugar bead
        p6=(a+1)*3   # next residue phostphate bead.
        index=bondIndexDict.get((d,a))
        HBforceObject.setBondParameters(index,[p1,p2,p3,p4,p5,p6],[0])
        #print("Done pair: {}".format(pair))

    for pair in Dij.keys():
        d=pair[0]
        a=pair[1]
        p1=d*3+2  # Nucleobase bead
        p2=d*3+1  # Sugar bead
        p3=(d+1)*3   # next residue phostphate bead.
        p4=a*3+2  # Nucleobase bead
        p5=a*3+1  # Sugar bead
        p6=(a+1)*3   # next residue phostphate bead.
        index=bondIndexDict.get((d,a))
        HBforceObject.setBondParameters(index,[p1,p2,p3,p4,p5,p6],[Dij[(d,a)]])
        #print("Done pair: {}".format(pair))

    HBforceObject.updateParametersInContext(Context)
    #print("DONE update in context !!!!")
    #print("Dij pairs = {}".format(Dij_pairs))
    return Dij_pairs ## Old_Dij_pairs feeded back into the set_HB_network method to compare to the new Dij quickly and update only the different elements.
#################


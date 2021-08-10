#!/usr/bin/env python
# coding: utf-8

'''
Last update: 20. 11. 26. by KS.Kwon

To Do [20.11.26.] atomic mass and van der waals radius features are float type feature but the others are one-hot encoded feature. Fix the data type.

[20.11.26.]
- Add atomic feature (atomic mass, van der waals radius)
'''

import sys, re, os
from itertools import chain

import numpy as np
import pandas as pd

from rdkit.Chem import ChemicalFeatures, AllChem, Descriptors
from rdkit import Chem, RDConfig

fdefName = '/home/ailon26/00_working/01_dataset/AilonFeatures_YTK_v5.fdef'
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

def inputs():
	in_file = sys.argv[1] # Mol2 file
	in_type = sys.argv[2]
	
	return in_type, in_file

def one_of_k_encoding(x, allowable_set):
	if x not in allowable_set:
		raise Exception("The input {0} is not in allowable_set: ".format(x))
		
	return map(lambda s: x == s, allowable_set)

def one_of_k_encoding_unk(x, allowable_set):
	"""Maps inputs not in the allowable set to the last element."""
	if x not in allowable_set:
		x = allowable_set[-1]
	return map(lambda s: x == s, allowable_set)


### Atom features ###
def get_atom_feature(mol, 
					 H_don=False, 
					 H_acc=False, 
					 atom_type=False, 
					 atom_degree=False, 
					 hybridization=False, 
					 partial_charge=False, 
					 chirality=False, 
					 num_H=False, 
					 aromaticity=False, 
					 implicit_valence=False, 
				 	 atom_mass=False, 
				 	 vdw=False):
	
	features = []
	H_donors, H_acceptors = Compute_H_Donor_Acceptor(mol)
	
	if partial_charge:
		AllChem.ComputeGasteigerCharges(mol)
	
	if VDW:
		ptable = Chem.GetPeriodicTable()

	try:
		for atom in mol.GetAtoms():
			features.append(np.array(
				AtomType(atom=atom, act=atom_type) + 
				Degrees(atom=atom, act=atom_degree) + 
				Hybridization(atom=atom, act=hybridization) + 
				PartialCharge(atom=atom, act=partial_charge) + 
				Chirality(atom=atom, act=chirality) + 
				NumOfHs(atom=atom, act=num_H) + 
				Aromaticity(atom=atom, act=aromaticity) + 
				ImplicitValence(atom=atom, act=implicit_valence) + 
				H_donor(atom=atom, H_donor_obj=H_donors, act=H_don) + 
				H_acceptor(atom=atom, H_acceptor_obj=H_acceptors, act=H_acc) +
				AtomMass(atom=atom, act=atom_mass) + 
				VDW(ptable=ptable, atom=atom, act=vdw)
			))
	except Exception: pass
	except TypeError: pass

	return np.asarray(features)

def Compute_H_Donor_Acceptor(mol):
	
	donor = []
	acceptor = []
	ligand_feats = factory.GetFeaturesForMol(mol)

	for feat in ligand_feats:
		fam = feat.GetFamily()
		if fam == 'Donor':
			donor += list(feat.GetAtomIds())
		elif fam == 'Acceptor':
			acceptor += list(feat.GetAtomIds())

	H_donor = []
	H_acceptor = []
		
	for idx, _ in enumerate(mol.GetAtoms()):
		if idx in donor:
			H_donor.append(1)
		else:
			H_donor.append(0)

		if idx in acceptor:
			H_acceptor.append(1)
		else:
			H_acceptor.append(0)

	return H_donor, H_acceptor

def H_donor(atom, H_donor_obj, act):
	if act is False:
		return []
	return [H_donor_obj[atom.GetIdx()]]

def H_acceptor(atom, H_acceptor_obj, act):
	if act is False:
		return []
	return [H_acceptor_obj[atom.GetIdx()]]

def AtomType(atom, act):
	if act is False:
		return []
	
	atom_type = one_of_k_encoding_unk(
		atom.GetSymbol(), 
		['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
		 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
		 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
		 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
		 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
	)

	return list(map(int, atom_type))

def Degrees(atom, act):
	if act is False:
		return []
	
	degree = one_of_k_encoding(
		atom.GetDegree(), 
		[0, 1, 2, 3, 4]
	)
	
	return list(map(int, degree))

def Hybridization(atom, act):
	if act is False:
		return []
	
	hybridization = one_of_k_encoding_unk(
		str(atom.GetHybridization()),
		['UNSPECIFIED', 'S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'OTHER']
	)

	return list(map(int, hybridization))

def FormalCharge(atom, act):
	if act is False:
		return []
	
	formal_charge = one_of_k_encoding(
		atom.GetFormalCharge(), 
		[-1, 0, 1]
	)
	
	return list(map(int, formal_charge))

def PartialCharge(atom, act):
	if act is False:
		return []
	
	partial_charge = atom.GetDoubleProp('_GasteigerCharge')
	
	if 0 < partial_charge:
		return [1, 0, 0]
	elif partial_charge == 0:
		return [0, 1, 0]
	elif partial_charge < 0:
		return [0, 0, 1]

def Chirality(atom, act):
	if act is False:
		return []
	
	chirality = one_of_k_encoding_unk(
		str(atom.GetChiralTag()), 
		['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER']
	)

	return list(map(int, chirality))

def NumOfHs(atom, act):
	if act is False:
		return []
	if atom.GetTotalNumHs() > 4:
		return False, []
	
	Hs = one_of_k_encoding(
		atom.GetTotalNumHs(), 
		[0, 1, 2, 3, 4]
	)
	
	return list(map(int, Hs))

def Aromaticity(atom, act):
	if act is False:
		return []
	return list(map(int, [atom.GetIsAromatic()]))

def ImplicitValence(atom, act):
	if act is False:
		return []
	
	impVal = one_of_k_encoding_unk(
		atom.GetImplicitValence(), 
		[0, 1, 2, 3, 4, 5]
	)
	
	return list(map(int, impVal))

def AtomMass(atom, act):
	if act is False:
		return []
	return [atom.GetMass()]

def VDW(ptable, atom, act):
	if act is False:
		return []
	return [ptable.GetRvdw(ptable.GetAtomicNumber(atom.GetSymbol()))]


### Bond features ###
def get_bond_feature(mol, 
					 bond_type=False, 
				 	 stereotype=False, 
				 	 conjugate=False, 
				 	 inRing=False):
	features = []
	for bond in mol.GetBonds():
		features.append(np.array(
			BondType(bond, act=bond_type) +
			BondStereo(bond, act=stereotype) +
			ConjugatedBond(bond, act=conjugate) +
			InRing(bond, act=inRing)
		))
	
	return np.asarray(features)

def BondType(bond, act):
	if act is False:
		return []
	bond_type = one_of_k_encoding_unk(
		str(bond.GetBondType()),
		['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
	)
	return list(map(int, bond_type))

def BondStereo(bond, act):
	if act is False:
		return []
	stereo = one_of_k_encoding_unk(
		str(bond.GetStereo()),
		['STEREONONE', 'STEREOANY', 'STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS']
	)
	return list(map(int, stereo))

def ConjugatedBond(bond, act):
	if act is False:
		return []
	return list(map(int, [bond.GetIsConjugated()]))

def InRing(bond, act):
	if act is False:
		return []
	return list(map(int, [bond.IsInRing()]))
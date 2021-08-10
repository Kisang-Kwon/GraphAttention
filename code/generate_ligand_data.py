#!/usr/bin/env python
# coding: utf-8

import os
import sys
import tensorflow as tf
import numpy as np
import csv
import argparse

from rdkit import Chem
from utils.dircheck import dircheck
from utils.get_time import get_time
from utils.get_file_length import get_file_length
from data.ligand.graph import MolGraph


def get_arguments():
    parser = argparse.ArgumentParser(description='Data generation')

    parser.add_argument('-o', '--outdir', dest='outdir', type=str, required=True)
    parser.add_argument('-f', '--input', dest='filepath', type=str)
    parser.add_argument('-s', '--single_input', dest='single_input', nargs=2)

    parser.add_argument(
        '--max_mol_node', dest='max_mol_node', type=int, default=60
    )
    parser.add_argument(
        '--max_mol_degree', dest='max_mol_degree', type=int, default=6
    )
    parser.add_argument('--max_bond', dest='max_bond', type=int, default=100)

    return parser.parse_args()


def gen_ligand_feature(cid, smiles, outdir, max_mol_node, max_mol_degree, max_bond):
    lig_filepath = os.path.join(outdir, f'{cid}.npy')
    
    # Convert to RDkit Canonical SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        pass
    else:
        smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        
        if os.path.isfile(lig_filepath):
            sys.stderr.write(
                f'[NOTICE] {cid}: Ligand feature file already exists.\n'
            )
        else:
            try:
                compnd_graph = MolGraph(cid=cid, smiles=smiles)
                
                if compnd_graph.mol is None:
                    sys.stderr.write(
                        f'[NOTICE] {cid}: Cannot convert to rdkit Mol object.\n'
                    )
                else:
                    matrices = compnd_graph.make_matrix(
                        max_mol_node, max_mol_degree, max_bond
                    )
                    
                    M_atom_feat = matrices[0]
                    M_bond_feat = matrices[1]
                    M_atom_adj = matrices[2]
                    M_bond_adj = matrices[3]
                    mask = matrices[4]
                    
                    ligand_features = np.array([
                        M_atom_feat,
                        M_bond_feat,
                        M_atom_adj, 
                        M_bond_adj, 
                        mask, 
                        cid
                    ], dtype=object)
                    np.save(lig_filepath, ligand_features)

            except IndexError as e:
                sys.stderr.write(f'[IndexError] {cid}: {e}\n')
            except TypeError as e:
                sys.stderr.write(f'[TypeError] {cid}: {e}\n')


if __name__ == '__main__':
    start_time = get_time('Start: ')
    args = get_arguments()

    if os.path.isdir(args.outdir) is False:
        dircheck(args.outdir)

    if args.single_input:
        cid = args.single_input[0]
        smiles = args.single_input[1]
        gen_ligand_feature(
            cid=cid,
            smiles=smiles,
            outdir=args.outdir,
            max_mol_node=args.max_mol_node,
            max_mol_degree=args.max_mol_degree,
            max_bond=args.max_bond
        )
    elif args.filepath:
        f_smi = open(args.filepath)
        csvreader = csv.reader(f_smi)
        #next(csvreader)

        total = get_file_length(args.filepath)
        now = 0
        for lig in csvreader:
            cid = lig[1]
            smiles = lig[3] # canonical smiles

            now+=1
            print(get_time(f'{cid} ({now} / {total}): '))
            
            gen_ligand_feature(
                cid=cid,
                smiles=smiles,
                outdir=args.outdir,
                max_mol_node=args.max_mol_node,
                max_mol_degree=args.max_mol_degree,
                max_bond=args.max_bond
            )
        f_smi.close()        
    else:
        raise ValueError('No input file passed')

    print(f'\n{start_time}')
    print(get_time('End: '))

import os
import csv
import sys
import argparse
import urllib.request
from urllib.error import URLError, HTTPError
from multiprocessing import Pool

from utils.dircheck import dircheck
from utils.get_time import get_time
from utils.get_file_length import get_file_length
from data.protein.PDB_extract.v3.complex import Complex


def get_arguments():
    parser = argparse.ArgumentParser(description='Data generation')

    parser.add_argument('-p', '--pocket_dir', dest='pocket_dir', type=str, required=True)
    parser.add_argument('-f', '--feature_dir', dest='feature_dir', type=str, required=True)
    parser.add_argument('-i', '--input', dest='input_file', type=str)
    parser.add_argument('-s', '--single_input', dest='single_input', nargs=2, 
        help='First argument: PDB ID or PDB file path.\
              Second argument: Ligand ID in PDB database'
    )
    parser.add_argument('--max_poc_node', dest='max_poc_node', type=int, default=100)
    parser.add_argument('--threads', dest='threads', type=int, default=1)
    parser.add_argument(
        '-u', '--using_chain_list', dest='using_chain_list', 
        default='/home/ailon26/00_working/01_dataset/rawdata/DB/Uniprot/pdb_uniprot_202012070957.csv'
    )
    parser.add_argument(
        '-t', '--id_format', 
        dest='id_format',
        default='pdb',
        help='If the id format is pdb,the first column in the input file will be pdb id\
              If the id format is path, the first column in the input file will be pdb file path.'
    )

    return parser.parse_args()


def get_pdb(pdb):
    pdb_dir = '/home/ailon26/00_working/01_dataset/rawdata/Protein/pdb'
    fpath = os.path.join(pdb_dir, f'{pdb.lower()}.pdb')
    if os.path.isfile(fpath):
        pass
    else:
        try:
            url = f'https://files.rcsb.org/download/{pdb}.pdb'
            urllib.request.urlretrieve(url, fpath)
        except HTTPError as e:
            sys.stderr.write(f'{pdb.lower()}, {e}\n')
            fpath = None

    return fpath


def generate_pocket_feature(data, fpath=False):
    pid = data[0]
    ligands = data[1]

    if fpath is False:
        fpath = get_pdb(pid)
    
    if fpath is not None:
        print(pid, get_time())
        try:
            p_complex = Complex(pid)
            p_complex.parse_pdb(fpath)
            protein = p_complex.set_protein_residues(pid)
            
            if protein:
                find_poc = False
                for cid in ligands:
                    p_complex.set_ligands(cid)
                    find_poc = True
                
                if find_poc:
                    p_complex.find_pocket()
            else:
                sys.stderr.write(
                    f'[NOTICE] {pid}: Does not have any protein instance.\n'
                )
                print(f'[NOTICE] {pid}: Does not have any protein instance.')    
        
        except ValueError as e:
            print(f'[{pid, ligands}]', e)
        except IndexError as e:
            print(f'[{pid, ligands}]', e)

def main(args):
    """
    If the id format is path, the file name must be started with pdb id.
    """
    if args.single_input:
        lig = args.single_input[1]
        if args.id_format == 'pdb':
            pid = args.single_input[0]
            generate_pocket_feature([pid, [lig]])
        elif args.id_format == 'path':
            fpath = args.single_input[0]
            #pid = os.path.basename(fpath).split('_')[0]
            pid = os.path.basename(fpath).replace('.pdb', '')
            generate_pocket_feature([pid, [lig]], fpath)

    elif args.input_file:
        fpath = args.input_file
        Fopen = open(fpath)
        cread = csv.reader(Fopen)
        next(cread)

        total = get_file_length(fpath)
        now = 0
        
        for line in cread:
            ligs = line[2].split(':')
            #ligs = line[1]
            
            if args.id_format == 'pdb':
                pid = line[0]
                now += 1
                print(get_time(f'{pid} ({now} / {total}): '))
                generate_pocket_feature([pid, ligs])

            elif args.id_format == 'path':
                #pid = os.path.basename(line[0]).split('_')[0]
                pid = os.path.basename(line[0]).replace('.pdb', '')
                fpath = line[0]
                now += 1
                print(get_time(f'{pid} ({now} / {total}): '))
                generate_pocket_feature([pid, ligs], fpath)


        Fopen.close()


def main_multi_thread(fpath, n_threads):
    Fopen = open(fpath)
    cread = csv.reader(Fopen)
    next(cread)
    data = []
    for line in cread:
        pid = line[0]
        ligs = line[2].split(':')
        data.append([pid, ligs])
    Fopen.close()

    p = Pool(n_threads)
    p.map(generate_pocket_feature, data)
    p.close()


if __name__ == '__main__':
    print('#### Generate Protein feature version 3 ####\n')
    start_time = get_time('Start: ')
    args = get_arguments()

    dircheck(args.pocket_dir)
    dircheck(args.feature_dir)

    Complex.POCKET_DIR = args.pocket_dir
    Complex.FEATURE_DIR = args.feature_dir
    Complex.max_poc_node = args.max_poc_node
    Complex.set_using_chain_list(args.using_chain_list)

    if args.threads == 1:
        main(args)
    elif 2 <= args.threads:
        main_multi_thread(args.input_file, args.threads)
    
    print(f'\n{start_time}')
    print(get_time('End: '))
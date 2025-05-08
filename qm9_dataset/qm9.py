import torch
from torch_geometric.datasets import QM9
import rdkit

qm9_path = './QM9'
dataset = QM9(root=qm9_path)

from tools.AtomAtom import AtomBlockDecomposer, AtomBlockReconstructor

import pyscf
import itertools

class MolMatrixDatapointPreprocesser:
    """
    Preprocesser for molecular matrix data points.
    """
    def __init__(self, basis: str) -> None:
        super().__init__()
        self.basis = basis
        
        # create dummy molecule to get the correct irreps of each element: atom_basis_dict
        dummy_mol = pyscf.gto.Mole()
        dummy_mol.atom = '''
            H       1.000000      0.000000      0.000000  
            H       2.000000      0.000000      0.000000    
            C       3.000000      0.000000      0.000000  
            O       4.000000      0.000000      0.000000
            N       5.000000      0.000000      0.000000  
            F       6.000000      0.000000      0.000000  
        '''
        dummy_mol.basis = basis
        dummy_mol.build()
        dummy_info = self._create_atom_block_info(dummy_mol)
        self.atom_basis_dict = {dummy_info[i]["type"]: dummy_info[i]["irreps"] for i in range(1, 6)}
        
        # create decomposers, covering all possible atom pairs, avoiding duplicates
        self.decomposers = self._create_ordered_decomposers(self.atom_basis_dict)
        
    def _create_ordered_decomposers(self, atom_basis_dict):
        """
        Create ordered atom pair decomposers, avoiding duplicates

        Parameters:
            atom_basis_dict: Dictionary containing atom basis information, keys are atom symbols. For ccpvdz, that is:
            {
                'H': '2x0e+1x1o',
                'C': '3x0e+2x1o+1x2e',
                'O': '3x0e+2x1o+1x2e',
                'N': '3x0e+2x1o+1x2e',
                'F': '3x0e+2x1o+1x2e'
            }

        Returns:
            decomposers: Dictionary of atom pair decomposers ordered by rules:
            {
                'HH': <tools.AtomAtom.AtomBlockDecomposer at 0x12e488670>,
                'CH': <tools.AtomAtom.AtomBlockDecomposer at 0x12e5be850>,
                'OH': <tools.AtomAtom.AtomBlockDecomposer at 0x12adea490>,
                'NH': <tools.AtomAtom.AtomBlockDecomposer at 0x12e6b9c40>,
                'FH': <tools.AtomAtom.AtomBlockDecomposer at 0x12ed990d0>,
                'CC': <tools.AtomAtom.AtomBlockDecomposer at 0x12e488280>,
                'CO': <tools.AtomAtom.AtomBlockDecomposer at 0x12e2d7e20>,
                'CN': <tools.AtomAtom.AtomBlockDecomposer at 0x149ae5220>,
                'CF': <tools.AtomAtom.AtomBlockDecomposer at 0x149a80820>,
                'OO': <tools.AtomAtom.AtomBlockDecomposer at 0x149ae5850>,
                'NO': <tools.AtomAtom.AtomBlockDecomposer at 0x12dfd3a00>,
                'OF': <tools.AtomAtom.AtomBlockDecomposer at 0x12ada7910>,
                'NN': <tools.AtomAtom.AtomBlockDecomposer at 0x149a67d30>,
                'NF': <tools.AtomAtom.AtomBlockDecomposer at 0x1499b1e80>,
                'FF': <tools.AtomAtom.AtomBlockDecomposer at 0x12ac68a90>
            }
        """
        # Atomic number mapping table
        atomic_numbers = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
            'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
        }
    
        # Get all atom types and sort them
        atom_types = list(atom_basis_dict.keys())

        # Define sorting function
        def sort_atoms(a, b):
            # H atom always comes last
            if a == 'H' and b != 'H':
                return (b, a)
            if b == 'H' and a != 'H':
                return (a, b)
            # For other atoms, sort by atomic number
            if atomic_numbers.get(a, 0) <= atomic_numbers.get(b, 0):
                return (a, b)
            else:
                return (b, a)
    
        # Create decomposers
        decomposers = {}
    
        # Add all possible atom pair combinations (without duplicates)
        for i, type1 in enumerate(atom_types):
            # Add self-pairs
            key1, key2 = type1, type1
            decomposers[key1+key2] = AtomBlockDecomposer(irreps_1=atom_basis_dict[key1], 
                                                     irreps_2=atom_basis_dict[key2])
        
            # Add pairs between different types
            for type2 in atom_types[i+1:]:
                key1, key2 = sort_atoms(type1, type2)
                decomposers[key1+key2] = AtomBlockDecomposer(irreps_1=atom_basis_dict[key1], 
                                                             irreps_2=atom_basis_dict[key2])

        return decomposers
        
    def _create_atom_block_info(self, mol: pyscf.gto.Mole):
        """
            Required function for def pre_process(self, mol: pyscf.gto.Mole, matrix: torch.Tensor):
            
            Create atom block information from a PySCF molecule object. A C2H5OH molecule is used as an example:
        
            INPUT:
                mol = gto.M(
                atom = '''
                C  0.000000  0.000000  0.000000
                O  0.000000  0.000000  1.000000
                C  0.000000  0.000000  2.000000
                H  0.000000  0.757000  0.586000
                H  0.000000 -0.757000  0.586000
                H  0.000000  0.000000  1.586000
                H  0.000000  0.000000  2.586000
                H  0.000000  0.000000  3.586000
                H  0.000000  0.000000  4.586000
                ''',
                basis = 'ccpvdz',  
                unit = 'angstrom', 
                )
        
            OUTPUT:
            {   
                0: {'type': 'C', 'start': 0, 'end': 14, 'irreps': '3x0e+2x1o+1x2e'},
                1: {'type': 'O', 'start': 14, 'end': 28, 'irreps': '3x0e+2x1o+1x2e'},
                2: {'type': 'C', 'start': 28, 'end': 42, 'irreps': '3x0e+2x1o+1x2e'},
                3: {'type': 'H', 'start': 42, 'end': 47, 'irreps': '2x0e+1x1o'},
                4: {'type': 'H', 'start': 47, 'end': 52, 'irreps': '2x0e+1x1o'},
                5: {'type': 'H', 'start': 52, 'end': 57, 'irreps': '2x0e+1x1o'},
                6: {'type': 'H', 'start': 57, 'end': 62, 'irreps': '2x0e+1x1o'},
                7: {'type': 'H', 'start': 62, 'end': 67, 'irreps': '2x0e+1x1o'},
                8: {'type': 'H', 'start': 67, 'end': 72, 'irreps': '2x0e+1x1o'}
            }
        """
        ao_labels = mol.ao_labels()
    
        # Initialize result dictionary
        atom_block_info = {}

        # Current orbital starting index
        start_idx = 0
    
        # Group by atoms
        atom_labels = {}
        for label in ao_labels:
            parts = label.split()
            atom_idx = int(parts[0])
            if atom_idx not in atom_labels:
                atom_labels[atom_idx] = []
            atom_labels[atom_idx].append(label)
    
        # Number of orbitals for each angular momentum
        am_orbitals = {'s': 1, 'p': 3, 'd': 5, 'f': 7, 'g': 9, 'h': 11}
    
        # e3nn representation of angular momentum
        am_irreps = {'s': '0e', 'p': '1o', 'd': '2e', 'f': '3o', 'g': '4e', 'h': '5o'}
    
        # Angular momentum order
        am_order = ['s', 'p', 'd', 'f', 'g', 'h']
    
        # Process each atom
        for atom_idx in sorted(atom_labels.keys()):
            # Get all orbital labels for current atom
            labels = atom_labels[atom_idx]
        
            # Calculate number of orbitals
            orbital_count = len(labels)
        
            # Determine atom type
            atom_type = mol.atom_symbol(atom_idx)
        
            # Calculate end index
            end_idx = start_idx + orbital_count
        
            # Count orbitals for each angular momentum type
            am_counts = {'s': 0, 'p': 0, 'd': 0, 'f': 0, 'g': 0, 'h': 0}
        
            for label in labels:
                parts = label.split()
                orbital_type = parts[2]
            
                # Find angular momentum character (s, p, d, f, g, h)
                for am_type in am_counts.keys():
                    if am_type in orbital_type:
                        am_counts[am_type] += 1
                        break
        
            # Calculate number of groups for each angular momentum
            am_groups = {}
            for am_type, count in am_counts.items():
                if count > 0:
                    # Divide by number of orbitals per group
                    am_groups[am_type] = count // am_orbitals[am_type]
        
            # Build irreps string, sorted by increasing angular momentum
            irreps_parts = []
            for am_type in am_order:  # Use predefined order
                if am_type in am_groups and am_groups[am_type] > 0:
                    irreps_parts.append(f"{am_groups[am_type]}x{am_irreps[am_type]}")
        
            irreps_str = "+".join(irreps_parts)
        
            # Add to result dictionary
            atom_block_info[atom_idx] = {
                "type": atom_type,
                "start": start_idx,
                "end": end_idx,
                "irreps": irreps_str
            }
        
            # Update starting index for next atom
            start_idx = end_idx
    
        return atom_block_info

    def _create_atom_pair_divide(self, atom_block_info):
        """
            Required function for def pre_process(self, mol: pyscf.gto.Mole, matrix: torch.Tensor):
            
            Create atom pair classification dictionary from atom information, sorted by atomic number (H atoms always placed last)

            INPUT:
            {
                0: {'type': 'C', 'start': 0, 'end': 14, 'irreps': '3x0e+2x1o+1x2e'},
                1: {'type': 'O', 'start': 14, 'end': 28, 'irreps': '3x0e+2x1o+1x2e'},
                2: {'type': 'C', 'start': 28, 'end': 42, 'irreps': '3x0e+2x1o+1x2e'},
                3: {'type': 'H', 'start': 42, 'end': 47, 'irreps': '2x0e+1x1o'},
                4: {'type': 'H', 'start': 47, 'end': 52, 'irreps': '2x0e+1x1o'},
                5: {'type': 'H', 'start': 52, 'end': 57, 'irreps': '2x0e+1x1o'},
                6: {'type': 'H', 'start': 57, 'end': 62, 'irreps': '2x0e+1x1o'},
                7: {'type': 'H', 'start': 62, 'end': 67, 'irreps': '2x0e+1x1o'},
                8: {'type': 'H', 'start': 67, 'end': 72, 'irreps': '2x0e+1x1o'}
            }
            
            OUTPUT:
            {   'CC-atom': [(0, 0), (2, 2)],
                'OO-atom': [(1, 1)],
                'HH-atom': [(3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)],
                'CC-bond': [(0, 2)],
                'CO-bond': [(0, 1), (2, 1)],
                'CH-bond': [(0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8)],
                'OO-bond': [],
                'OH-bond': [(1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8)],
                'HH-bond': [(3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (4, 5), (4, 6), (4, 7), (4, 8), (5, 6), (5, 7), (5, 8), (6, 7), (6, 8), (7, 8)]
            }
        """
        # Get indices and types of all atoms
        atom_types = {idx: info['type'] for idx, info in atom_block_info.items()}

        # Atomic number mapping table (common elements)
        atomic_numbers = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
            'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
            # More elements can be added as needed
        }

        # Group atoms by type
        atom_by_type = {}
        for idx, type_name in atom_types.items():
            if type_name not in atom_by_type:
                atom_by_type[type_name] = []
            atom_by_type[type_name].append(idx)
    
        # Initialize result dictionary
        atom_pair_divide = {}
    
        # Process all possible atom type combinations
        atom_types_set = set(atom_types.values())
    
        # Get sorted list of atom types (H placed last)
        def sort_key(atom):
            if atom == 'H':
                return float('inf')  # H atoms always placed last
            return atomic_numbers.get(atom, 0)
    
        sorted_types = sorted(atom_types_set, key=sort_key)

        # Process self-atom pairs (X-X-atom)
        for atom_type in sorted_types:
            key = f"{atom_type}{atom_type}-atom"
            atom_pair_divide[key] = [(idx, idx) for idx in atom_by_type.get(atom_type, [])]

        # Process bonds between different atoms (X-Y-bond)
        for type1, type2 in itertools.product(sorted_types, sorted_types):
            if type1 == type2:
                key = f"{type1}{type2}-bond"
                atom_pair_divide[key] = [(i, j) for i, j in itertools.combinations(atom_by_type.get(type1, []), 2)]
            else:
                # Determine atom order (H atoms last, others by atomic number)
                if type1 == 'H' or (type2 != 'H' and atomic_numbers.get(type1, 0) > atomic_numbers.get(type2, 0)):
                    type1, type2 = type2, type1

                key = f"{type1}{type2}-bond"
                atom_pair_divide[key] = [(i, j) for i in atom_by_type.get(type1, []) 
                                        for j in atom_by_type.get(type2, [])]

        return atom_pair_divide
    
    def pre_process(self, mol: pyscf.gto.Mole, matrix: torch.Tensor):
        
        # Get atom block information
        atom_block_info = self._create_atom_block_info(mol)
        atom_pair_divide = self._create_atom_pair_divide(atom_block_info)
        
        # Initialize result dictionary
        matrix_decomposed = {}
        
        # Loop through each atom pair and decompose the corresponding block
        for pair, block_idx in atom_pair_divide.items():
            pair_type = pair[:2]    # first two characters of the pair are the atom types
            parameters_array = []
            parameter_irrep_structure = None    # only one irreps structure for each pair type
            
            for idx in block_idx:
                
                block = matrix[atom_block_info[idx[0]]["start"]:atom_block_info[idx[0]]["end"], atom_block_info[idx[1]]["start"]:atom_block_info[idx[1]]["end"]]
                decomposition_result = self.decomposers[pair_type].decompose_atom_block(block)

                parameters_array.append(decomposition_result["parameters_array"])
                parameter_irrep_structure = decomposition_result["parameter_irrep_structure"]

            matrix_decomposed[pair] = {
                "parameters_array": parameters_array,
                "parameter_irrep_structure": parameter_irrep_structure
            }
        
        return matrix_decomposed

test_preprocesser = MolMatrixDatapointPreprocesser(basis="sto-3g")

from pyscf import gto, scf
import os
import numpy as np

save_path = "./QM9_pyscf"

if not os.path.exists(save_path):
    os.makedirs(save_path)

# 定义原子序数到元素符号的映射
atomic_num_to_symbol = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'
}

def create_pyscf_molecule(sample, basis):
    z_list = sample.z.numpy()  # 原子序数列表
    pos = sample.pos.numpy()   # 原子坐标
    
    # 构建分子的原子规格
    atom_specs = []
    for i in range(len(z_list)):
        z = int(z_list[i])
        element = atomic_num_to_symbol.get(z, 'X')  # 将原子序数转换为元素符号
        x_coord, y_coord, z_coord = pos[i]
        atom_specs.append([element, (x_coord, y_coord, z_coord)])
    
    # 创建PySCF分子对象
    mol = gto.Mole()
    mol.atom = atom_specs
    mol.basis = basis  # 使用相对标准的基组
    mol.charge = 0  # 假设分子是中性的
    mol.spin = 0    # 假设分子是闭壳层的
    mol.unit = 'Angstrom'  # 设置坐标单位为Angstrom
    mol.verbose = 0  # 设置输出级别为0，避免输出过多信息
    mol.build()
    
    return z_list, pos, mol
        

def process_molecule(sample, basis):

    z_list, pos, mol = create_pyscf_molecule(sample, basis)  # 创建PySCF分子对象
    
    # 执行Hartree-Fock计算
    mf = scf.RHF(mol)
    dm_sad = mf.init_guess_by_minao()
    
    mf.kernel()
    hf_dm = mf.make_rdm1()

    return {
        "z_list": np.array(z_list, dtype=np.int32),
        "pos": np.array(pos, dtype=np.float32),
        "sad": np.array(dm_sad, dtype=np.float32),
        "hf_dm": np.array(hf_dm, dtype=np.float32),
    }
    
from tqdm import tqdm
import pickle

for i in tqdm(range(0, 5000)):
    sample = dataset[i]
    processed_data = process_molecule(sample, "sto-3g")
    
    with open(os.path.join(save_path, f'molecule_{i}.pkl'), 'wb') as f:
        pickle.dump(processed_data, f)
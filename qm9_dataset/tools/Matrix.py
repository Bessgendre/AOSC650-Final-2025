import torch
from e3nn import o3
import itertools
from .AtomAtom import AtomBlockDecomposer, AtomBlockReconstructor

class AtomBasis:
    def __init__(self, atom_type: str, basis_list: str) -> None:
        super(AtomBasis, self).__init__()
        self.atom_type = atom_type
        self.basis_list = o3.Irreps(basis_list)

class MatrixDecomposer:
    """
        This MatrixDecomposer class is used to decompose the whole density matrix into its irreps. 
        
        Only upper triangular part of the density matrix is considered.
        
        One MatrixDecomposer for the whole dataset. When creating an instance, please specify all of the possible elements and the irreps their basis functions belong to (determined by the choice of basis when doing quantum chemistry calculation, like sto-3g or ccpvdz) with a a list of class AtomBasis. For example, if you are using ccpvdz:

            ccpvdz:
            
                C: 3x0e+2x1o+1x2e
                N: 3x0e+2x1o+1x2e
                O: 3x0e+2x1o+1x2e
                H: 2x0e+1x1o
                
            Then you can create the element table like this:
            
                >>>carbon = AtomBasis("C", "3x0e+2x1o+1x2e")
                >>>nitrogen = AtomBasis("N", "3x0e+2x1o+1x2e")
                >>>oxygen = AtomBasis("O", "3x0e+2x1o+1x2e")
                >>>hydrogen = AtomBasis("H", "2x0e+1x1o")
                
                >>>element_table = [carbon, nitrogen, oxygen, hydrogen]
        
        When doing the decomposition, please provide the atom_list of that molecule and its target matrix.
        
        The output will be parameters for atoms & bonds and their irrep structures. The irrep structures are crucial for the reconstruction. The irrep structures are organized like:
        
            {
                "atom_irrep_structures": [
                    [[atom 1 & channel 1], [atom 1 & channel 2], ...],
                    [[atom 2 & channel 1], [atom 2 & channel 2], ...],
                    ...
                ],
                "bond_irrep_structures": [
                    [[bond 1 & channel 1], [bond 1 & channel 2], ...],
                    [[bond 2 & channel 1], [bond 2 & channel 2], ...],
                    ...
                ]
            }
        
        The AtomBlockReconstructor is working on blocks, which in the irrep structures are things like:
        
            [[atom 1 & channel 1], [atom 1 & channel 2], ...]
        
        which is the first dimension of the irrep structures. This information will be used to reconstruct the whole matrix.

    """
    def __init__(self, element_table: list[AtomBasis]) -> None:
        super(MatrixDecomposer, self).__init__()
        # Get the elements possibly involved in the matrix
        self.element_table = element_table
        
        # Create one decomposer for each unique element pair to save time and memory. But consider C-N and N-C require different decomposers to enhance robustness (if the atom_list = ["O", "N", "N", "C", "H", "H", "H", "H", "H", "H", "H", "H", "H"]).
        self.unique_element_pair = list(itertools.product(element_table, repeat=2))
        self.element_block_dim = {
            element.atom_type: o3.Irreps(element.basis_list).dim for element in self.element_table
        }
        self.atom_atom_decomposers = {f"{pair[0].atom_type}-{pair[1].atom_type}": AtomBlockDecomposer(pair[0].basis_list, pair[1].basis_list) for pair in self.unique_element_pair}
        
        # get the irreps of all atom-atom blocks
        self.atom_atom_irreps = {f"{pair[0].atom_type}-{pair[1].atom_type}": self.atom_atom_decomposers[f"{pair[0].atom_type}-{pair[1].atom_type}"].all_decomposed_irreps for pair in self.unique_element_pair}
        
    
    def decompose_matrix(self, atom_list: list[str], target_matrix: torch.Tensor):
        
        # atom_info contains the start and end index of each atom's block in the density matrix
        atom_dim_separate = [self.element_block_dim[atom] for atom in atom_list]
        atom_info = {i: {"atom_type": atom_list[i], "start": sum(atom_dim_separate[:i]), "end": sum(atom_dim_separate[:i + 1])} for i in range(len(atom_dim_separate))}
        
        # add the irreps to each atom
        for i in atom_info.keys():
            atom_type = atom_info[i]["atom_type"]
            for element in self.element_table:
                if element.atom_type == atom_type:
                    atom_info[i]["irreps"] = element.basis_list
                    break
        
        # When doing decomposition, both parameters and the corresponding irreps are needed for reconstruction.        
        atom_irrep_parameters = []
        atom_irreps = []
        bond_irrep_parameters = []
        bond_irreps = []
        # From left to right, top to bottom. only upper triangular part of the density matrix is considered.
        for atom_idx_i in range(len(atom_list)):
            for atom_idx_j in range(atom_idx_i, len(atom_list)):
                
                # get the atom-atom block from the target matrix
                atom_atom_block = target_matrix[atom_info[atom_idx_i]["start"]:atom_info[atom_idx_i]["end"], atom_info[atom_idx_j]["start"]:atom_info[atom_idx_j]["end"]]
                
                # atom-atom block decomposition
                block_type = f"{atom_info[atom_idx_i]['atom_type']}-{atom_info[atom_idx_j]['atom_type']}"
                result = self.atom_atom_decomposers[block_type].decompose_atom_block(atom_atom_block)
                block_parameter_array = result["parameters_array"]
                block_irreps = result["parameter_irrep_structure"]
                
                # if the atom type is the same, then it is a atom target, otherwise it is a bond target
                if atom_idx_i == atom_idx_j:
                    atom_irrep_parameters.append(block_parameter_array)
                    atom_irreps.append(block_irreps)
                else:
                    bond_irrep_parameters.append(block_parameter_array)
                    bond_irreps.append(block_irreps)

        # concatenation will lose information, so we need irrep_structures to save the structure information for future reconstruction
        self.atom_irrep_parameters = torch.cat(atom_irrep_parameters, dim=0)
        self.bond_irrep_parameters = torch.cat(bond_irrep_parameters, dim=0)
        
        return {
            "parameters": {
                "atom_irrep_parameters": self.atom_irrep_parameters, "bond__irrep_parameters": self.bond_irrep_parameters
                },
            "irrep_structures": {
                "atom_irrep_structures": atom_irreps, 
                "bond_irrep_structures": bond_irreps
                }
        }

class MatrixReconstructor:
    """
        This MatrixReconstructor class is used to reconstruct the whole density matrix from its irrep array and irrep structure.
    """
    
    def __init__(self, element_table: list[AtomBasis]) -> None:
        super(MatrixReconstructor, self).__init__()
        # Get the elements possibly involved in the matrix
        self.element_table = element_table
        
        # Create one reconstructor for each unique element pair to save time and memory. But consider C-N and N-C require different reconstructors to enhance robustness (if the atom_list = ["O", "N", "N", "C", "H", "H", "H", "H", "H", "H", "H", "H", "H"]).
        self.unique_element_pair = list(itertools.product(element_table, repeat=2))
        self.element_block_dim = {
            element.atom_type: o3.Irreps(element.basis_list).dim for element in self.element_table
        }
        self.atom_atom_reconstructors = {f"{pair[0].atom_type}-{pair[1].atom_type}": AtomBlockReconstructor(pair[0].basis_list, pair[1].basis_list) for pair in self.unique_element_pair}
        
        # get the irreps of all atom-atom blocks
        self.atom_atom_irreps = {f"{pair[0].atom_type}-{pair[1].atom_type}": self.atom_atom_reconstructors[f"{pair[0].atom_type}-{pair[1].atom_type}"].all_decomposed_irreps for pair in self.unique_element_pair}
    
    def reconstruct_matrix(self, atom_list: list[str], parameters_and_structures_dict: dict):
        parameters = parameters_and_structures_dict["parameters"]
        atom_irrep_parameters = parameters["atom_irrep_parameters"]
        bond_irrep_parameters = parameters["bond__irrep_parameters"]
        
        atom_irrep_structures = parameters_and_structures_dict["irrep_structures"]["atom_irrep_structures"]
        bond_irrep_structures = parameters_and_structures_dict["irrep_structures"]["bond_irrep_structures"]
        
        # atom_info contains the start and end index of each atom's block in the density matrix. it denotes the correct place a block should be put in the matrix.
        atom_dim_separate = [self.element_block_dim[atom] for atom in atom_list]
        atom_info = {i: {"atom_type": atom_list[i], "start": sum(atom_dim_separate[:i]), "end": sum(atom_dim_separate[:i + 1])} for i in range(len(atom_dim_separate))}
        
        # add the irreps to each atom
        for i in atom_info.keys():
            atom_type = atom_info[i]["atom_type"]
            for element in self.element_table:
                if element.atom_type == atom_type:
                    atom_info[i]["irreps"] = element.basis_list
                    break
             
        # here we need to separate the array-like parameters. we depend on this to find the correct parameters and irrep structure for a block.
        atom_array_separate = []
        bond_array_separate = []
        for i in range(len(atom_list)):
            for j in range(i, len(atom_list)):
                if i == j:
                    atom_array_separate.append(atom_dim_separate[i]**2)
                else:
                    bond_array_separate.append(atom_dim_separate[i] * atom_dim_separate[j])    
        atom_array_info = {i: {"start": sum(atom_array_separate[:i]), "end": sum(atom_array_separate[:i + 1])} for i in range(len(atom_array_separate))}
        bond_array_info = {i: {"start": sum(bond_array_separate[:i]), "end": sum(bond_array_separate[:i + 1])} for i in range(len(bond_array_separate))}
        
        # reconstruct the matrix
        reconstructed_matrix = torch.zeros((sum(atom_dim_separate), sum(atom_dim_separate)), dtype=torch.float32)
        
        # count atom and bond separately
        current_atom_idx = 0
        current_bond_idx = 0
        for i in range(len(atom_list)):
            for j in range(i, len(atom_list)):
                
                # get the atom-atom block from the target matrix
                block_type = f"{atom_info[i]['atom_type']}-{atom_info[j]['atom_type']}"
        
                if i == j:
                    # atom block
                    atom_block_parameters = atom_irrep_parameters[atom_array_info[current_atom_idx]["start"]:atom_array_info[current_atom_idx]["end"]]
                    
                    atom_block_irreps = atom_irrep_structures[current_atom_idx]
                    
                    atom_block = self.atom_atom_reconstructors[block_type].reconstruct_atom_block(
                        {
                            "parameters_array": atom_block_parameters,
                            "parameter_irrep_structure": atom_block_irreps
                        })
                    
                    reconstructed_matrix[atom_info[i]["start"]:atom_info[i]["end"], atom_info[j]["start"]:atom_info[j]["end"]] = atom_block
                    
                    current_atom_idx += 1
                
                else:
                    # bond block
                    bond_block_parameters = bond_irrep_parameters[bond_array_info[current_bond_idx]["start"]:bond_array_info[current_bond_idx]["end"]]
                    
                    bond_block_irreps = bond_irrep_structures[current_bond_idx]
                    
                    bond_block = self.atom_atom_reconstructors[block_type].reconstruct_atom_block(
                        {
                            "parameters_array": bond_block_parameters,
                            "parameter_irrep_structure": bond_block_irreps
                        })
                    
                    # bond block is symmetric, so we need to put it in both places.
                    reconstructed_matrix[atom_info[i]["start"]:atom_info[i]["end"], atom_info[j]["start"]:atom_info[j]["end"]] = bond_block
                    
                    reconstructed_matrix[atom_info[j]["start"]:atom_info[j]["end"], atom_info[i]["start"]:atom_info[i]["end"]] = bond_block.t()
                    
                    current_bond_idx += 1
        
        return reconstructed_matrix

if __name__ == "__main__":
    carbon = AtomBasis("C", "3x0e+2x1o+1x2e")
    hydrogen = AtomBasis("H", "2x0e+1x1o")

    element_table = [carbon, hydrogen]
    
    # test on CH4, ccpvdz basis
    
    test_matrix = torch.randn(34, 34, dtype=torch.float32)

    test_matrix = test_matrix + test_matrix.t()
    
    test_matrix_decomposer = MatrixDecomposer(element_table)
    
    test_matrix_result = test_matrix_decomposer.decompose_matrix(["C", "H", "H", "H", "H"], test_matrix)
    
    test_matrix_reconstructor = MatrixReconstructor(element_table)
    
    test_matrix_reconstructed = test_matrix_reconstructor.reconstruct_matrix(["C", "H", "H", "H", "H"], test_matrix_result)
    
    if torch.allclose(test_matrix, test_matrix_reconstructed, atol=1e-6):
        print("Matrix reconstruction successful!")
    else:
        print("Matrix reconstruction failed!")
        print("Original matrix:\n", test_matrix)
        print("Reconstructed matrix:\n", test_matrix_reconstructed)
        print("Difference:\n", test_matrix - test_matrix_reconstructed)
    
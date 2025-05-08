import torch
from e3nn import o3
import itertools
from .AngularMomentum import AngularMomentumBlockDecomposer, AngularMomentumBlockReconstructor


class AtomBlockDecomposer:
    """
        This AtomBlockDecomposer class can deal with the decomposition of a atom-atom block in the density matrix. It will handle multiple angular momentum channels and return a list of parameters for each channel-channel pair. 
        
        The output will have a precise order of angular momentum channels depending on the real arrangement of atom-atom block: LEFT TO RIGHT, UP TO DOWN. This order should be the same when we preprocess the label matrix data into such irreps.
        
        Unique angular momentum channel couplings are used to save time and memory. The decomposer will only create one instance of the AngularMomentumBlockDecomposer for each unique angular momentum channel coupling, and reuse it for all atom-atom blocks.
        
        1ox2e and 2ex1o are different.
        
        NOTICE: If suit_model is True, the output will be a single array of parameters and the structure of it. If suit_model is False, he output will be a list of dicts, each dict containing the parameters for a specific channel-channel pair. This is used to bridge the gap between the model and the data preprocessing where in the model, we only need the tensor and specify the structure of it, but in the data preprocessing, we need to know the parameters for each channel-channel pair to decompose and reconstruct.
    """
    def __init__(self, irreps_1: str, irreps_2: str) -> None:
        super(AtomBlockDecomposer, self).__init__()
        self.irreps_1 = o3.Irreps(irreps_1)
        self.irreps_2 = o3.Irreps(irreps_2)
        
        # Preparation: create a list of unique angular momentum channel couplings and the corresponding unique decomposer
        self.unique_channels_1 = [o3.Irreps([(1, (l, p))]).__str__() for mul, (l, p) in self.irreps_1] 
        self.unique_channels_2 = [o3.Irreps([(1, (l, p))]).__str__() for mul, (l, p) in self.irreps_2]
        
        self.unique_angular_coupling_pair = list(itertools.product(self.unique_channels_1, self.unique_channels_2))
        self.unique_decomposer = {f"{pair[0]}-{pair[1]}": AngularMomentumBlockDecomposer(pair[0], pair[1]) for pair in self.unique_angular_coupling_pair}
        
        # Preparation: separate simplified irreps into individual channels
        self.row_channel_separate = [(1, (l, p)) for mul, (l, p) in self.irreps_1 for i in range(mul)]
        self.column_channel_separate = [(1, (l, p)) for mul, (l, p) in self.irreps_2 for i in range(mul)]
        
        # Preparation: find the dimension of individual channels
        self.row_dim_separate = [(2*l+1) for mul, (l, p) in self.irreps_1 for i in range(mul)]
        self.column_dim_separate = [(2*l+1) for mul, (l, p) in self.irreps_2 for i in range(mul)]
        
        # Create a mapping between irrep indices to their angular momentum and positions
        self.row_info = {i: {"irreps": o3.Irreps([self.row_channel_separate[i]]).__str__(), "start": sum(self.row_dim_separate[:i]), "end": sum(self.row_dim_separate[:i+1])} for i in range(len(self.row_dim_separate))}
        self.column_info = {i: {"irreps": o3.Irreps([self.column_channel_separate[i]]).__str__(), "start": sum(self.column_dim_separate[:i]), "end": sum(self.column_dim_separate[:i+1])} for i in range(len(self.column_dim_separate))}
        
        self.all_decomposed_irreps = self._decompose_irreps()
    
    def _decompose_irreps(self):
        
        all_decomposed_irreps = []
        
        for row_channel, row_channel_info in self.row_info.items():
            for column_channel, column_channel_info in self.column_info.items():
                full_tp = o3.FullTensorProduct(irreps_in1=row_channel_info["irreps"], irreps_in2=column_channel_info["irreps"])
                all_decomposed_irreps.append(full_tp.irreps_out.__str__())
        
        for i in range(len(all_decomposed_irreps)):
            all_decomposed_irreps[i] = all_decomposed_irreps[i].split('+')

        return all_decomposed_irreps 
            
        
        
        # # suit_model means we want to return a single array of parameters and the structure of it
        # if suit_model:
        #     parameters_array = torch.cat([torch.cat(list(channel.values())) for channel in all_parameters], dim=0)
        #     parameter_irrep_structure = [list(channel.keys()) for channel in all_parameters]
        #     return {
        #         "parameters_array": parameters_array,
        #         "parameter_irrep_structure": parameter_irrep_structure,
        #     }
        # else:
        #     return all_parameters
        
    def decompose_atom_block(self, atom_block: torch.Tensor, suit_model=True) -> dict:
        
        all_parameters = []
        
        for row_channel, row_channel_info in self.row_info.items():
            for column_channel, column_channel_info in self.column_info.items():
                
                # Extract the angular momentum block from the atom-atom block
                angular_block = atom_block[row_channel_info["start"]:row_channel_info["end"], column_channel_info["start"]:column_channel_info["end"]]
                
                # Decompose the angular momentum block into parameters
                parameterizer = self.unique_decomposer[f"{row_channel_info['irreps']}-{column_channel_info['irreps']}"]
                parameters = parameterizer.decompose_block(angular_block)
                all_parameters.append(parameters)
        
        # return all_parameters
        # suit_model means we want to return a single array of parameters and the structure of it
        if suit_model:
            parameters_array = torch.cat([torch.cat(list(channel.values())) for channel in all_parameters], dim=0)
            parameter_irrep_structure = [list(channel.keys()) for channel in all_parameters]
            return {
                "parameters_array": parameters_array,
                "parameter_irrep_structure": parameter_irrep_structure,
            }
        else:
            return all_parameters


class AtomBlockReconstructor:
    """
        This AtomBlockReconstructor class is the reverse of the AtomBlockDecomposer. It follows the same logic as the decomposer, but reconstructs the atom-atom block from the parameters. It will also handle multiple angular momentum channels and return a single atom-atom block.
        
        The error between an original atom-atom block and reconstructed block is atol=1e-7 if using torch.allclose.
    """
    
    def __init__(self, irreps_1: str, irreps_2: str) -> None:
        super(AtomBlockReconstructor, self).__init__()
        self.irreps_1 = o3.Irreps(irreps_1)
        self.irreps_2 = o3.Irreps(irreps_2)
        
        # Preparation: create a list of unique angular momentum channel couplings and the corresponding unique reconstructor
        self.unique_channels_1 = [o3.Irreps([(1, (l, p))]).__str__() for mul, (l, p) in self.irreps_1] 
        self.unique_channels_2 = [o3.Irreps([(1, (l, p))]).__str__() for mul, (l, p) in self.irreps_2]
        
        self.unique_angular_coupling_pair = list(itertools.product(self.unique_channels_1, self.unique_channels_2))
        self.unique_reconstructor = {f"{pair[0]}-{pair[1]}": AngularMomentumBlockReconstructor(pair[0], pair[1]) for pair in self.unique_angular_coupling_pair}
        
        # Preparation: separate simplified irreps into individual channels
        self.row_channel_separate = [(1, (l, p)) for mul, (l, p) in self.irreps_1 for i in range(mul)]
        self.column_channel_separate = [(1, (l, p)) for mul, (l, p) in self.irreps_2 for i in range(mul)]
        
        # Preparation: find the dimension of individual channels
        self.row_dim_separate = [(2*l+1) for mul, (l, p) in self.irreps_1 for i in range(mul)]
        self.column_dim_separate = [(2*l+1) for mul, (l, p) in self.irreps_2 for i in range(mul)]
        
        # Create a mapping between irrep indices to their angular momentum and positions
        self.row_info = {i: {"irreps": o3.Irreps([self.row_channel_separate[i]]).__str__(), "start": sum(self.row_dim_separate[:i]), "end": sum(self.row_dim_separate[:i+1])} for i in range(len(self.row_dim_separate))}
        self.column_info = {i: {"irreps": o3.Irreps([self.column_channel_separate[i]]).__str__(), "start": sum(self.column_dim_separate[:i]), "end": sum(self.column_dim_separate[:i+1])} for i in range(len(self.column_dim_separate))}
    
        self.all_decomposed_irreps = self._decompose_irreps()
    
    def _decompose_irreps(self):
        
        all_decomposed_irreps = []
        
        for row_channel, row_channel_info in self.row_info.items():
            for column_channel, column_channel_info in self.column_info.items():
                full_tp = o3.FullTensorProduct(irreps_in1=row_channel_info["irreps"], irreps_in2=column_channel_info["irreps"])
                all_decomposed_irreps.append(full_tp.irreps_out.__str__())
        
        for i in range(len(all_decomposed_irreps)):
            all_decomposed_irreps[i] = all_decomposed_irreps[i].split('+')
        
        return all_decomposed_irreps
        
    def reconstruct_atom_block(self, array_and_structure: dict) -> torch.Tensor:
        
        # We always specify suit_model=True in AtomBlockDecomposer. But this part is originally designed for suit_model=False, so we need to reshape the array_and_structure into all_parameters who is the standard output of AtomBlockDecomposer if suit_model=False.
        
        #-------------------------------------------
        all_parameters = []
        current_index = 0
        for coupled_channels in array_and_structure["parameter_irrep_structure"]:
            reshaped_channels = {}
            for channel in coupled_channels:
                reshaped_channels[channel] = array_and_structure["parameters_array"][current_index:current_index + 2 * o3.Irreps(channel).lmax + 1]
                current_index += 2 * o3.Irreps(channel).lmax + 1
            all_parameters.append(reshaped_channels)
        #-------------------------------------------
        
        
        # original version if suit_model=False in AtomBlockDecomposer
        
        # Initialize the atom-atom block with zeros
        atom_block = torch.zeros((sum(self.row_dim_separate), sum(self.column_dim_separate)), dtype=torch.float32)
        
        for row_channel, row_channel_info in self.row_info.items():
            for column_channel, column_channel_info in self.column_info.items():
                
                # Get the parameters for the current angular momentum block
                parameters = all_parameters[row_channel * len(self.column_info) + column_channel]
                
                # Reconstruct the angular momentum block from the parameters
                parameterizer = self.unique_reconstructor[f"{row_channel_info['irreps']}-{column_channel_info['irreps']}"]
                angular_block = parameterizer.reconstruct_block(parameters)
                
                # Place the angular momentum block back into the atom-atom block
                atom_block[row_channel_info["start"]:row_channel_info["end"], column_channel_info["start"]:column_channel_info["end"]] = angular_block
        
        return atom_block
    
if __name__ == "__main__":
    # for a carbon atom-atom block in ccpvdz basis
    atom_atom_decomposer = AtomBlockDecomposer("3x0e+2x1o+1x2e", "3x0e+2x1o+1x2e")
    atom_atom_reconstructor = AtomBlockReconstructor("3x0e+2x1o+1x2e", "3x0e+2x1o+1x2e")
    
    # create a dummy symmetric atom-atom block
    test_atom_atom_matrix = torch.randn(14, 14, dtype=torch.float32)
    test_atom_atom_matrix = test_atom_atom_matrix + test_atom_atom_matrix.T
    
    # test the decomposer and reconstructor reversibility
    test_decom = atom_atom_decomposer.decompose_atom_block(test_atom_atom_matrix)
    test_recon = atom_atom_reconstructor.reconstruct_atom_block(test_decom)
    
    if torch.allclose(test_atom_atom_matrix, test_recon):
        print("Successfully decomposed and reconstructed the atom-atom block!")
    else:
        print("Decomposition and reconstruction failed. The original and reconstructed atom-atom blocks are not close enough.")
        print(f"Original atom-atom block:\n{test_atom_atom_matrix}")
        print(f"Reconstructed atom-atom block:\n{test_recon}")
        print(f"Error: {torch.abs(test_atom_atom_matrix - test_recon).max()}")
    
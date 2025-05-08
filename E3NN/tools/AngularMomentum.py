import torch
import numpy as np
from sympy.physics.quantum.cg import CG

from e3nn import o3
from e3nn.o3 import Irreps

from itertools import product
from scipy.linalg import block_diag

class AngularMomentumBlockDecomposer:
    """
    The AngularMomentumBlockDecomposer class decomposes an angular momentum coupling block (P_block) of an equivariant matrix into irreducible representations.
    
    An matrix is said to be equivariant if it transforms under the action of the rotation group in a way that preserves the structure of the irreducible representations. Density matrices, for example, are equivariant under rotations. Nevertheless, it shows the equivariance property block-wisely since the row and column of the DM has certain angular momentum.
    
    Due to the special mathematical structure of DMs, this representation decomposition is named as Clebsch-Gordan decomposition (https://zhuanlan.zhihu.com/p/525617797) , which algebraically speaking, is a unitary similarity transformation that takes the Kronecker product of two wigner-D matrices D(l_1), D(l_2) into a direct sum of another set of wigner-D matrices D(|l_1-l_2|), D(|l_1-l_2+1|), ..., D(|l_1+l_2|):

        Clebsch-Gordan decomposition: There exists a unitary S such that
        S {D(l_1) ⊗ D(l_2)} S_dagger = D(|l_1-l_2|) ⊕ D(|l_1-l_2+1|) ⊕ ... ⊕ D(|l_1+l_2|)
    
    where S_dagger contains the Clebsch-Gordan coefficients, and S is a transformation matrix that takes the original flattened vector (P_block -> p, a long array) into another flattened vector (q = S @ p) that serves as the coefficients of the decomposed irreducible representations. By choosing a suitable phase convention, which is already done by Condon and Shotley in their book 'The Theory of Atomic Spectra', S_dagger happens to be real numbers. These numbers, called Clebsch-Gordan coefficients, are saved in math softwares and python packages, like 'sympy.physics.quantum.cg'. Since flatten operation and S are both reversible, the decomposition is reversible.
    
    However, the discussion above is in general case for standard complex spherical harmonics. So there are two problems:
    
        1. What computers are actually using is the real spherical harmonics, which is a linear recombination of the complex spherical harmonics (https://en.wikipedia.org/wiki/Spherical_harmonics).
        2. Further more, numbers passed through the computers are always real numbers, so the decomposed coefficients should be real.
    
    In short, the first problem requires us to answer "How does the basis change affect the S_dagger?", which is easier to solve , but the resulting new S_dagger violates the second problem. Therefore, it requires to find a proper phase convention for the S_dagger matrix so that all coefficients are real numbers. 
    
    The implementation below solves these two problems, and the detailed explanation can be found in notes.
    """
    
    def __init__(self, irrep_1: str, irrep_2: str) -> None:
        
        # only deal with channel - channel decomposition
        self.l1 = Irreps(irrep_1).lmax
        self.l2 = Irreps(irrep_2).lmax
        
        # get the irrep decomposition
        self.coupled_irreps = o3.FullTensorProduct(irreps_in1=irrep_1, irreps_in2=irrep_2).irreps_out
        self.coupled_info = [(mul, (l, p)) for mul, (l, p) in self.coupled_irreps]
        self.channel_separate = [2*l+1 for (_, (l, _)) in self.coupled_info]
        self.channel_start_end = [
            {
                "start": sum(self.channel_separate[:i]), 
                "end": sum(self.channel_separate[:i+1])
            }
            for i in range(len(self.channel_separate))
        ]
        
        # calculate S_dagger for complex spherical harmonics
        self.S_dagger_complex_SH, _, _ = self.cg_matrix(self.l1, self.l2)
        
        # create the transformation from complex to real spherical harmonics
        X_l1_l2_independent = self.create_kron_product(self.l1, self.l2)
        X_direct_sum_coupled = self.create_direct_sum(self.l1, self.l2)
        
        # calculate the S_dagger for real spherical harmonics
        self.S_dagger_real_SH = X_l1_l2_independent.conj().T @ self.S_dagger_complex_SH @ X_direct_sum_coupled
        
        # choose the correct phase convention
        self.S_matrix_transpose = self.S_dagger_real_SH.real + self.S_dagger_real_SH.imag
        self.S_matrix = self.S_matrix_transpose.T
        
        # change the type to torch tensor
        self.S_matrix = torch.tensor(self.S_matrix, dtype=torch.float32)
        self.S_matrix_transpose = torch.tensor(self.S_matrix_transpose, dtype=torch.float32)
        
    def complex_to_real_SH_transform_matrix(self, l):
        """
        Create the transformation matrix X from complex spherical harmonics to real spherical harmonics.
        """
        dim = int(2 * l + 1)
        X_matrix = np.zeros((dim, dim), dtype=np.complex128)
    
        def m_to_idx(m):
            return int(m + l)
    
        for m in range(-l, l + 1):
            idx_m = m_to_idx(m)
            if m < 0:
                # m < 0: |l,m⟩ = (i/√2)(|l,m⟩_C - (-1)^m |l,-m⟩_C)
                X_matrix[m_to_idx(m), idx_m] = 1j / np.sqrt(2)
                X_matrix[m_to_idx(-m), idx_m] = -1j * (-1)**m / np.sqrt(2)

            elif m == 0:
                # m = 0: |l,m⟩ = |l,m⟩_C
                X_matrix[m_to_idx(0), idx_m] = 1.0

            else:  # m > 0
                # m > 0: |l,m⟩ = (1/√2)(|l,-m⟩_C + (-1)^m |l,m⟩_C)
                X_matrix[m_to_idx(-m), idx_m] = 1 / np.sqrt(2)
                X_matrix[m_to_idx(m), idx_m] = (-1)**m / np.sqrt(2)
    
        return X_matrix

    def cg_matrix(self, l1, l2):
        """
        Create a matrix of Clebsch-Gordan coefficients.

        Parameters:
        l1, l2: Angular momenta to be coupled

        Returns:
        Matrix where rows are indexed by (m1, m2) and columns by (L, M)
        Dictionary mapping row indices to (m1, m2) pairs
        Dictionary mapping column indices to (L, M) pairs
        """
        # Possible values for m1, m2
        m1_values = np.arange(-l1, l1 + 1)
        m2_values = np.arange(-l2, l2 + 1)

        # Possible values for L
        L_values = np.arange(abs(l1 - l2), l1 + l2 + 1)

        # Create mappings for row and column indices
        row_indices = {}
        col_indices = {}

        row_idx = 0
        for m1, m2 in product(m1_values, m2_values):
            row_indices[row_idx] = (m1, m2)
            row_idx += 1

        col_idx = 0
        for L in L_values:
            for M in np.arange(-L, L + 1):
                col_indices[col_idx] = (L, M)
                col_idx += 1
    
        # Initialize matrix with zeros
        matrix = np.zeros((len(row_indices), len(col_indices)))
    
        # Fill the matrix with CG coefficients
        for row_idx, (m1, m2) in row_indices.items():
            for col_idx, (L, M) in col_indices.items():
                # CG coefficient is non-zero only if m1 + m2 = M
                if m1 + m2 == M:
                    # Calculate the CG coefficient using sympy
                    cg_value = float(CG(l1, m1, l2, m2, L, M).doit())
                    matrix[row_idx, col_idx] = cg_value

        return matrix, row_indices, col_indices
    
    def create_kron_product(self, l1, l2):
        return np.kron(self.complex_to_real_SH_transform_matrix(l1), self.complex_to_real_SH_transform_matrix(l2))
    
    def create_direct_sum(self, l1, l2):
        """
        Create a direct sum of the transformation matrices for all L values from |l1 - l2| to l1 + l2.
        """
        dump = None
        X_matrix_list = []
    
        for L_value in range(abs(l1 - l2), l1 + l2 + 1):
            X_matrix = self.complex_to_real_SH_transform_matrix(L_value)
        
            X_matrix_list.append(X_matrix)
    
        for matrix in X_matrix_list:
            if dump is None:
                dump = matrix
            else:
                dump = block_diag(dump, matrix)
            
        return dump

    def decompose_block(self, P_block: torch.tensor, irreps_divide=True) -> dict:
        """
        Decompose a block in equivariant matrix into irreducible representations.
        """
        
        p_vector = P_block.flatten()
        q_vector = self.S_matrix @ p_vector
        
        if irreps_divide:
        # divide the coefficients into irreducible representations
            coefficients = {}
            for i, irrep_info in enumerate(self.coupled_info):
                # get the irreducible representation info
                mul, (l, p) = irrep_info
                coefficients[Irreps([(mul, (l, p))]).__str__()] = q_vector[self.channel_start_end[i]["start"]:self.channel_start_end[i]["end"]]
        
            return coefficients
        
        else:
            # return the coefficients as a single vector
            return q_vector

class AngularMomentumBlockReconstructor(AngularMomentumBlockDecomposer):
    """
        The reverse process of AngularMomentumBlockDecomposer.
    """
    def __init__(self, irrep_1: str, irrep_2: str) -> None:
        super().__init__(irrep_1, irrep_2)
    
    def reconstruct_block(self, coefficients, irreps_divide=True):
        q_vector = torch.cat(list(coefficients.values())) if irreps_divide else coefficients
        
        p_vector = self.S_matrix_transpose @ q_vector
        P_block = p_vector.reshape(2*self.l1+1, 2*self.l2+1)
        
        return P_block

if __name__ == "__main__":
    
    # 1. Does the real S solve the problem of real CG decomposition with real spherical harmonics?
    test_decom = AngularMomentumBlockDecomposer("1x1o", "1x2e")

    irreps_1 = Irreps("1x1o")
    irreps_2 = Irreps("1x2e")
    irreps_3 = Irreps("1x3o")

    rot_matrix = o3.rand_matrix()
    wigner_1 = irreps_1.D_from_matrix(rot_matrix)
    wigner_2 = irreps_2.D_from_matrix(rot_matrix)
    wigner_3 = irreps_3.D_from_matrix(rot_matrix)

    test_block = torch.randn(3, 5)
    q_before = test_decom.decompose_block(test_block, False)

    block_after = wigner_1 @ test_block @ wigner_2.T
    q_after = test_decom.decompose_block(block_after, False)

    direct_sum = torch.tensor(block_diag(wigner_1, wigner_2, wigner_3))

    if torch.allclose(direct_sum @ q_before, q_after, atol=1e-6, rtol=1e-6) and torch.allclose(test_decom.S_matrix_transpose @ test_decom.S_matrix, torch.eye(test_decom.S_matrix.shape[0]), atol=1e-6, rtol=1e-6):
        print("The S is exactly what we want!")
    else:
        print("Wrong!")
        
    # 2. Is the reconstruction correct?
    
    test_block = torch.randn(5, 5)

    test_decom = AngularMomentumBlockDecomposer("1x2o", "1x2e")
    test_recon = AngularMomentumBlockReconstructor("1x2o", "1x2e")

    result = test_decom.decompose_block(test_block, irreps_divide=True)
    reconstructed_block = test_recon.reconstruct_block(result, irreps_divide=True)

    if torch.allclose(test_block, reconstructed_block, atol=1e-6, rtol=1e-6):
        print("Reversible decomposition and reconstruction!")
    else:
        print("Reconstruction failed!")
        
        print("Original Block:\n", test_block)
        print("Reconstructed Block:\n", reconstructed_block)
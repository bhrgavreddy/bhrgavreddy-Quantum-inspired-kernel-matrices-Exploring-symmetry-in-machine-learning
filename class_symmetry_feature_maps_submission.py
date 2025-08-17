import numpy as np
from scipy.linalg import expm
import func_sun

class SymmetryFeatureMaps:
    def __init__(self, num_features):
        self.num_features = num_features
        # Initialize the group sizes
        size_SO = find_so_group(num_features)
        size_SL = find_sl_group(num_features)  # SL and SU have the same size
        size_SU = find_su_group(num_features)  # SL and SU have the same size
        size_GL = find_gl_u_group(num_features)
        size_U = find_gl_u_group(num_features)
        size_O = find_o_group(num_features)
        size_T = find_translation_group(num_features)

        self.size_SO = size_SO
        self.size_SL = size_SL
        self.size_SU = size_SU
        self.size_GL = size_GL
        self.size_U = size_U
        self.size_O = size_O
        self.size_T = size_T

        # Generate the corresponding generators for each group
        self.group_generators_SO = func_sun.generate_SO(size_SO)
        self.group_generators_SL = func_sun.generate_SL_from_SU(size_SL)
        self.group_generators_GL = func_sun.generate_GL(size_GL)
        self.group_generators_O = func_sun.generate_O(size_O)
        self.group_generators_U = func_sun.generate_U(size_U)
        self.group_generators_SU = func_sun.generate_SU(size_SU)
        self.group_generators_T = func_sun.generate_T(size_T)

    def get_group_sizes(self):
        """
        Returns a dictionary containing the number of features and the number of generators for each group.
        """
        group_sizes = {
            "num_features": self.num_features,
            "size_SO": len(self.group_generators_SO),
            "size_SL": len(self.group_generators_SL),
            "size_SU": len(self.group_generators_SU),
            "size_GL": len(self.group_generators_GL),
            "size_U": len(self.group_generators_U),
            "size_O": len(self.group_generators_O),
            "size_T": len(self.group_generators_T)
        }
        return group_sizes

    def apply_feature_map(self, X, group_type, output_real=False, return_group_n=False):
        if group_type == "SO":
            if return_group_n: return self.SOn_feature_map(X, output_real=output_real), self.size_SO
            else: return self.SOn_feature_map(X, output_real=output_real)
        elif group_type == "SL":
            if return_group_n: return self.SLn_feature_map(X, output_real=output_real), self.size_SL
            else: return self.SLn_feature_map(X, output_real=output_real)
        elif group_type == "SU":
            if return_group_n: return self.SUn_feature_map(X, output_real=output_real), self.size_SU
            else: return self.SUn_feature_map(X, output_real=output_real)
        elif group_type == "GL":
            if return_group_n: return self.GLn_feature_map(X, output_real=output_real), self.size_GL
            else: return self.GLn_feature_map(X, output_real=output_real)
        elif group_type == "U":
            if return_group_n: return self.Un_feature_map(X, output_real=output_real), self.size_U
            else: return self.Un_feature_map(X, output_real=output_real)
        elif group_type == "O":
            if return_group_n: return self.On_feature_map(X, output_real=output_real), self.size_O
            else: return self.On_feature_map(X, output_real=output_real)
        elif group_type == "T":
            if return_group_n: return self.Tn_feature_map(X, output_real=output_real), self.size_T
            else: return self.Tn_feature_map(X, output_real=output_real)
        else:
            raise ValueError(f"Unknown group type: {group_type}")

    def SOn_feature_map(self, X, output_real=False):
        return self.generic_feature_map(X, self.group_generators_SO, output_real=output_real)

    def SLn_feature_map(self, X, output_real=False):
        return self.generic_feature_map(X, self.group_generators_SL, output_real=output_real)

    def SUn_feature_map(self, X, output_real=False):
        return self.generic_feature_map(X, self.group_generators_SU, output_real=output_real)

    def GLn_feature_map(self, X, output_real=False):
        return self.generic_feature_map(X, self.group_generators_GL, output_real=output_real)

    def Un_feature_map(self, X, output_real=False):
        return self.generic_feature_map(X, self.group_generators_U, output_real=output_real)

    def On_feature_map(self, X, output_real=False):
        return self.generic_feature_map(X, self.group_generators_O, output_real=output_real)

    def Tn_feature_map(self, X, output_real=False):
        return self.generic_feature_map(X, self.group_generators_T, output_real=output_real)

    def complex_to_real_vector(self, vector):
        """
        Convert a complex vector with n components into a real vector with 2n components.
        """
        real_part = vector.real
        imag_part = vector.imag
        return np.concatenate((real_part, imag_part))

    def generic_feature_map(self, X, generators, output_real=False):
        num_features = len(X)
        num_generators = len(generators)

        group_element = np.sum([X[i] * generators[i] for i in range(min(num_features, num_generators))], axis=0)

        group_element = expm(1j * group_element)

        dim = generators[0].shape[0]
        initial_vector = np.ones(dim) / np.sqrt(dim)

        transformed_vector = np.dot(group_element, initial_vector)

        # Convert to real vector if required
        if output_real:
            transformed_vector = self.complex_to_real_vector(transformed_vector)

        return transformed_vector

def find_su_group(num_features):
    """
    Find the smallest n for SU(n) that provides enough generators for the number of features.
    """
    n = 2  # Start from SU(2)
    while True:
        num_generators = n**2 - 1
        if num_generators >= num_features:
            # Ensure at least equal number of generators as features
            return max(n, int(np.ceil(np.sqrt(num_features + 1))))
        n += 1

def find_sl_group(num_features):
    return find_su_group(num_features)  # Same as SU(n)

def find_so_group(num_features):
    n = 1  # Start from SO(1)
    while True:
        num_generators = n * (n - 1) // 2
        if num_generators >= num_features:
            return n
        n += 1

def find_gl_u_group(num_features):
    n = 1  # Start from GL(1) or U(1)
    while True:
        num_generators = n**2
        if num_generators >= num_features:
            return n
        n += 1

def find_u_group(num_features):
    n = 1  # Start from GL(1) or U(1)
    while True:
        num_generators = n**2
        if num_generators >= num_features:
            return n
        n += 1

def find_o_group(num_features):
    return find_so_group(num_features)  # Same as SO(n)

def find_translation_group(num_features):
    """
    Find the smallest n for the translation group T(n) that provides enough generators for the number of features.
    In the case of the translation group, each dimension has a single generator.
    """
    n = 1  # Start from T(1)
    while True:
        num_generators = n  # Each dimension has one generator
        if num_generators >= num_features:
            return n
        n += 1

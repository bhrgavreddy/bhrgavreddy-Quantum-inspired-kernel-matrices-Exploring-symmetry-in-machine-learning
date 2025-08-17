import numpy as np
from scipy.linalg import expm

class ClassicalFeatureMap:
    def __init__(self):
        print('Quantum Feature Map initialized')
        # Pauli Matrices
        self.sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        # Hadamard Gate
        self.hadamard = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex)

    def apply_feature_map(self, X, feature_map_name):
        """
        Apply a quantum feature map to the input vector X, closely following the mathematical descriptions.

        Parameters:
        X (numpy.ndarray): An input vector of real numbers.
        feature_map_name (str): The name of the quantum feature map to apply ('Z', 'ZZ', or 'Pauli').
        Only Z implemented

        Returns:
        numpy.ndarray: The transformed feature vector, simulating the effect of the quantum feature map.
        """
        if feature_map_name == 'Z':
            return self.z_feature_map(X)
        else:
            raise ValueError(f"Unknown feature map type: {feature_map_name}")

    def z_feature_map(self, X):
        """
        Implements the Z feature map by applying a sequence of Hadamard gates followed by U1 phase rotations
        based on the input features. This feature map applies the phase rotations without entangling gates,
        focusing on encoding the data into the quantum states through phase modulation.

        The process involves applying a Hadamard gate to each qubit to create a superposition, followed by
        a U1 phase rotation gate that encodes the feature data into the phase of the qubit. This sequence
        captures the essence of the Z feature map, leveraging the principles of quantum superposition and
        phase encoding to represent classical data in a quantum state.

        Parameters:
        X (numpy.ndarray): A vector of real numbers representing classical data. Each element in X corresponds
                           to a feature to be encoded into the quantum state by a qubit.

        Returns:
        numpy.ndarray: A complex vector representing the final quantum state after applying the Z feature map.
                       This vector simulates the state space of the qubits with encoded data, illustrating how
                       classical information can be translated into quantum information.

        Example Usage:
        feature_map = ClassicalFeatureMap()
        data_vector = np.array([0.5, -1.3, 2.4])
        quantum_state = feature_map.z_feature_map(data_vector)
        print("Simulated Quantum State:", quantum_state)
        """
        # Prepare initial states with Hadamard gates
        states = [np.dot(self.hadamard, np.array([1, 0], dtype=complex)) for _ in X]

        # Encode data with Z rotations
        states = [np.dot(expm(-1j * x * self.sigma_z), state) for x, state in zip(X, states)]

        # Concatenate all qubit states to form the final combined state
        combined_state = np.hstack(states)

        return combined_state

    """
    Could be extended with e.g. ZZ feature map
    """






import numpy as np
from scipy.linalg import expm
import sys
from copy import deepcopy as dc
import numpy as np
from scipy.linalg import block_diag

def commutator(A, B):
    """
    Calculate the commutator of two matrices.
    """
    return np.dot(A, B) - np.dot(B, A)

def check_orthogonality(gens):
    """
    Check if the generators are orthogonal under the Hilbert-Schmidt inner product.

    Orthogonality of generators is a desirable property for computational purposes.
    Two matrices A and B are orthogonal under the Hilbert-Schmidt inner product if
    Tr(A† B) = 0, where Tr is the trace and † denotes the conjugate transpose.

    Parameters:
    gens : array-like
        Array of generator matrices of the SU(n) group.

    Returns:
    bool
        True if all generators are mutually orthogonal, False otherwise.
    """
    num_gens = len(gens)
    for i in range(num_gens):
        for j in range(i + 1, num_gens):
            if not np.isclose(np.trace(np.dot(gens[i].conj().T, gens[j])), 0):
                return False
    return True

def is_unitary(matrix):
    """
    Check if a matrix is unitary.

    A matrix U is unitary if its conjugate transpose U^dagger is equal to its inverse,
    i.e., U^dagger U = I, where I is the identity matrix.

    Parameters
    ----------
    matrix : array-like
        A square matrix.

    Returns
    -------
    bool
        True if the matrix is unitary, False otherwise.
    """
    identity = np.eye(matrix.shape[0])
    matrix_conj_transpose = np.conj(matrix).T
    return np.allclose(np.dot(matrix_conj_transpose, matrix), identity)

def check_algebra_closure(gens):
    """
    for SU matrices

    :param gens:
    :return:
    """
    dim = len(gens)
    nm = gens[0].shape[0]
    for i in range(dim):
        for j in range(dim):
            comm = commutator(gens[i], gens[j])
            # Solve for coefficients in the linear combination of generators that equals the commutator
            coeffs, _, _, _ = np.linalg.lstsq(gens.reshape(dim, -1).T, comm.ravel(), rcond=None)
            # Reconstruct the commutator from the linear combination of generators
            reconstructed_comm = sum(coeffs[k] * gens[k] for k in range(dim))
            if not np.allclose(comm, reconstructed_comm):
                return f"Algebra closure failed for generators {i} and {j}."
    return "Algebra closure verified."

def is_skew_hermitian(matrix):
    """
    Check if a matrix is skew-Hermitian.

    A matrix is skew-Hermitian if it is equal to the negation of its own
    conjugate transpose.

    Parameters
    ----------
    matrix : array-like
        A square matrix.

    Returns
    -------
    bool
        True if the matrix is skew-Hermitian, False otherwise.
    """
    matrix_conj_transpose = matrix.conj().T
    return np.allclose(matrix, -matrix_conj_transpose, atol=1e-10)

def is_orthogonal(matrix):
    """
    Check if a matrix is orthogonal.

    A matrix is orthogonal if its transpose is equal to its inverse.

    Parameters
    ----------
    matrix : array-like
        A square matrix.

    Returns
    -------
    bool
        True if the matrix is orthogonal, False otherwise.
    """
    return np.allclose(matrix @ matrix.T, np.eye(matrix.shape[0]))

def is_hermitian(matrix):
    """
    NOT TESTED
    Check if a matrix is Hermitian.

    A matrix is Hermitian if it is equal to its own conjugate transpose.
    In other words, the matrix is Hermitian if it is unchanged when
    replaced by its conjugate transpose.

    Parameters
    ----------
    matrix : array-like
        A square matrix.

    Returns
    -------
    bool
        True if the matrix is Hermitian, False otherwise.
    """
    # Conjugate transpose of the matrix
    matrix_conj_transpose = matrix.conj().T
    # Check if the original matrix is close to its conjugate transpose
    return np.allclose(matrix, matrix_conj_transpose)


def random_element(G):
    """
    Generate a random element of the group represented by the list of generators G.

    A random element is created by forming a linear combination of the generators
    with random coefficients, and then exponentiating the resulting matrix.

    Parameters
    ----------
    G : list of array-like
        List of generators of the group.

    Returns
    -------
    array-like
        A random element of the group.
    """
    # Get the dimension of the group from the first generator
    N = G[0].shape[0]
    # Get the number of generators
    num_generators = len(G)

    # Generate a set of random coefficients
    #coefficients = np.random.rand(num_generators)
    #coefficients = np.random.uniform(0, 2 * np.pi, num_generators)
    coefficients = np.random.uniform(np.pi-0.001,np.pi+0.001, num_generators)

    # Create a linear combination of the generators with the coefficients
    linear_combination = sum(coefficients[i] * G[i] for i in range(num_generators))

    # Apply the matrix exponential operation
    R = expm(linear_combination)

    return R

def generate_SU(n):
    """
    Generate the generators of the special unitary group SU(n).

    SU(n) is a Lie group of unitary matrices with determinant 1. It plays a significant role in both mathematics and physics, especially in quantum mechanics and quantum computing.
    The generators of SU(n) are n x n traceless Hermitian matrices, which form a basis for the Lie algebra su(n) of SU(n).

    These generators are crucial in representing infinitesimal rotations in a complex vector space, which is a key concept in quantum state transformations.
    SU(n) is closely related to the general unitary group U(n), which includes all unitary matrices without the determinant constraint.
    The difference is that SU(n) focuses on 'special' unitary matrices (with determinant 1), representing a 'pure' rotation without scaling, whereas U(n) includes both rotations and phase shifts.

    Comparatively, SU(n) is more restrictive than U(n) due to the determinant condition, making its algebraic structure and representation theory richer and more complex.
    For SU(n), the number of independent generators is n^2 - 1. These generators are used to parameterize the group elements via exponential mapping from the Lie algebra to the Lie group.

    Add, the construction we used had the problem, that sometimes it would produce hermitian and sometimes skew-hermitian matrices, however, all elements fulfilled the algebera closure, but the skew-hermitianess had to be guaranteed, see (1)

    Parameters
    ----------
    n : int
        The dimension of the group.

    Returns
    -------
    array-like
        A set of generators for SU(n).

    Note:
    The function utilizes a recursive approach for n > 2, extending the generators of SU(n-1) to SU(n) and adding new generators as needed.

    Reference:
    Hall, Brian C. "Lie Groups, Lie Algebras, and Representations: An Elementary Introduction."
    Graduate Texts in Mathematics, Springer, 2015.
    Accessed on 03.12.2023.
    """

    def traceless_diag(n):
        """
        Helper function to generate a traceless diagonal array.

        The diagonal contains -1's and a single (n - 1) on the last
        entry to ensure the trace is zero.

        Parameters
        ----------
        n : int
            The size of the diagonal.

        Returns
        -------
        array-like
            A traceless diagonal array of size n.
        """
        tot_arr = np.zeros(n)
        for i in range(n - 1):
            tot_arr[i] = -1
        tot_arr[n - 1] = n - 1
        return tot_arr

    # Check the dimension n
    if n < 2:
        return 'Choose n>=2'

    # Case n=2
    if n == 2:
        SU2_gens = np.zeros((3, 2, 2), dtype=np.complex128)
        SU2_gens[0] = np.array([[0., 1], [1, 0]])
        SU2_gens[1] = np.array([[0., -1j], [1j, 0]])
        SU2_gens[2] = np.array([[1., 0], [0, -1]])

        for i in range(len(SU2_gens)):
            if is_skew_hermitian(SU2_gens[i]):
                SU2_gens[i] = SU2_gens[i]
            else:
                SU2_gens[i] *= 1j/2

        return SU2_gens

    # Case n>2
    else:
        dim = n ** 2 - 1
        dim_m_1 = (n - 1) ** 2 - 1
        gens = np.zeros((dim, n, n), dtype=np.complex128)
        gens_m_1 = generate_SU(n - 1)

        # Extend generators of SU(n-1) to SU(n)
        for i in range(dim_m_1):
            gens[i] = np.append(np.append(gens_m_1[i], np.zeros((n - 1, 1)), axis=1),
                                np.zeros((1, n)), axis=0)

        # Generators with 1 entries
        for a in range(dim_m_1, dim_m_1 + (n - 1)):
            gens[a, a % (n - 1), n - 1] = 1
            gens[a, n - 1, a % (n - 1)] = 1

        # Generators with 1j entries
        for a in range(dim_m_1 + (n - 1), dim_m_1 + 2 * (n - 1)):
            gens[a, a % (n - 1), n - 1] = -1j
            gens[a, n - 1, a % (n - 1)] = 1j

        # Generator with diagonal entries
        gens[dim - 1] = (2 ** 0.5 / (n * (n - 1)) ** 0.5) * np.diag(traceless_diag(n))
        # Multiply each generator by 'j' to make them skew-Hermitian
        for i in range(dim): #(1)
            if is_skew_hermitian(gens[i]):
                gens[i] = gens[i]
            else:
                gens[i] *= 1j

        return gens

def is_skew_symmetric(matrix):
    """
    Check if a matrix is skew-symmetric.

    A matrix is skew-symmetric if the transpose of the matrix is the
    negative of the matrix itself. In mathematical terms, a matrix A
    is skew-symmetric if A^T = -A, where A^T is the transpose of A.

    Parameters
    ----------
    matrix : array-like
        A square matrix.

    Returns
    -------
    bool
        True if the matrix is skew-symmetric, False otherwise.
    """
    matrix_transpose = np.transpose(matrix)
    return np.allclose(matrix, -matrix_transpose)

def generate_SO(n):
    """
    Generate the generators of the special orthogonal group SO(n).

    The generators are skew-symmetric matrices that can be used to
    generate any element of the group through linear combinations and
    exponentiation.

    Parameters
    ----------
    n : int
        The dimension of the group.

    Returns
    -------
    array-like
        A set of generators for SO(n).
    """

    dim = int(n * (n - 1) / 2)
    gens = np.zeros((dim, n, n), dtype=np.complex128)
    ij_pair = [(i, j) for i in range(n) for j in range(n) if i < j]

    # Construct the generators
    for a, (i, j) in enumerate(ij_pair):
        gens[a, i, j] = 1.
        gens[a, j, i] = -1.

    return gens

def generate_SL_from_SU(n):
    """
    Generate the generators of the special linear group SL(n) from the generators of SU(n).

    SL(n) is a Lie group consisting of n x n matrices with determinant 1. It is a subgroup of the general linear group GL(n), which comprises all invertible n x n matrices. The significance of SL(n) lies in its property of 'volume-preserving' transformations, as the determinant condition ensures that transformations represented by SL(n) do not alter the volume in n-dimensional space.

    The relationship between SL(n) and SU(n) (the special unitary group) is rooted in their algebraic structures. SU(n) consists of n x n unitary matrices with determinant 1, while SL(n) includes n x n matrices (not necessarily unitary) with the same determinant condition. The generators of SU(n) are traceless Hermitian matrices, and by extending these generators with complex coefficients, we can obtain the generators for SL(n). This extension is possible because both groups share the determinant condition and have similar algebraic foundations, though SL(n) is more general as it does not require matrices to be unitary.

    In quantum mechanics, SU(n) is often more directly applicable due to its unitary nature (preserving quantum state probabilities), while SL(n) finds its applications in broader areas, including geometry and theoretical physics, where volume-preserving transformations are crucial.

    Parameters
    ----------
    n : int
        The dimension of the group.

    Returns
    -------
    array-like
        A set of generators for SL(n).

    Note:
    This function generates SL(n) generators by taking the generators of SU(n) and modifying them to suit the requirements of SL(n), demonstrating the close relationship between these two groups.

    Reference:
    Hall, Brian C. "Lie Groups, Lie Algebras, and Representations: An Elementary Introduction."
    Graduate Texts in Mathematics, Springer, 2015.
    https://link.springer.com/book/10.1007/978-3-319-13467-3
    Accessed on 3.12.2023.
    """

    gens = generate_SU(n)

    # Change generators with purely imaginary entries
    for i in range(len(gens)):
        if np.iscomplex(gens[i]).any():
            gens[i] = gens[i] * 1j

    return gens

def generate_T(n):
    """
    Generate the generators of the Translation Group T(n).

    T(n) represents the group of translations in n-dimensional Euclidean space.
    It is a fundamental group in physics and mathematics for describing translations.

    Parameters
    ----------
    n : int
        The dimension of the space.

    Returns
    -------
    array-like
        A set of generators for T(n), each being an n x n matrix. There are n generators,
        each corresponding to a translation along one of the axes in the n-dimensional space.
    """
    gens = np.zeros((n, n, n), dtype=np.complex128)
    for i in range(n):
        gens[i, i, -1] = 1.  # Assigning translation along the i-th axis

    return gens

def generate_GL(n):
    """
    TESTED PASSED
    Generate the generators of the General Linear Group GL(n).

    GL(n) consists of all n x n invertible matrices. The generators of GL(n) are matrices that
    span the space of all n x n matrices. Each generator has a single unit entry, with the
    rest of the entries being zero.

    Parameters:
    n (int): The dimension of the group.

    Returns:
    numpy.ndarray: An array of shape (n^2, n, n), representing the generators of GL(n).
    """
    dim = n * n
    gens = np.zeros((dim, n, n), dtype=np.complex128)
    ij_pair = [(i, j) for i in range(n) for j in range(n)]

    # Construct the generators
    for a, (i, j) in enumerate(ij_pair):
        gens[a, i, j] = 1.

    return gens

def generate_U(n):
    """
    Generate the generators of the unitary group U(n).

    The unitary group U(n) consists of all n x n unitary matrices, which are matrices U satisfying U†U = I, where U† is the conjugate transpose of U, and I is the identity matrix. Unitary matrices represent rotations and phase shifts in complex vector spaces, playing a pivotal role in quantum mechanics and quantum computing for describing symmetries and evolution of quantum states.

    Generators of U(n):
    The generators of U(n) are the basis elements of its Lie algebra, denoted as u(n), consisting of skew-Hermitian matrices. These generators can be exponentiated to produce the elements of U(n), facilitating the representation of continuous transformations.

    Relationship with SU(n):
    U(n) is closely related to the special unitary group SU(n), which is a subgroup of U(n) comprising unitary matrices with determinant 1. The generators of SU(n) are traceless skew-Hermitian matrices, forming the su(n) Lie algebra. While SU(n) captures 'pure' rotations in an n-dimensional complex vector space, U(n) extends this by incorporating an additional degree of freedom associated with global phase shifts, leading to n^2 generators in total.

    Construction of U(n) Generators:
    The first n^2 - 1 generators of U(n) are taken directly from SU(n), representing the traceless skew-Hermitian matrices. The nth^2 generator, distinguishing U(n) from SU(n), accounts for the global phase factor and is represented by a purely imaginary multiple of the identity matrix, specifically i * I. This additional generator reflects the U(1) subgroup within U(n), associated with phase transformations that do not alter the magnitude of quantum states.

    Parameters
    ----------
    n : int
        The dimension of the group, specifying the size of the unitary matrices.

    Returns
    -------
    array-like
        A set of n^2 generators for U(n), each of size n x n, where the first n^2 - 1 generators are traceless skew-Hermitian matrices from SU(n), and the final generator is i * I, introducing the global phase shift characteristic of U(n).

    Notes
    -----
    - The generators are constructed in a way that ensures skew-Hermitian properties, guaranteeing that their exponentiation results in unitary matrices.
    - The inclusion of the global phase factor through the additional generator i * I is a key aspect that differentiates U(n) from SU(n), reflecting the broader symmetries U(n) encompasses compared to SU(n).

    References
    ----------
    - Hall, Brian C. "Lie Groups, Lie Algebras, and Representations: An Elementary Introduction." Graduate Texts in Mathematics, Springer, 2015.
    - The concept of extending SU(n) to U(n) by adding a global phase factor is discussed in various mathematical physics literature, highlighting the fundamental role of U(n) in describing symmetries in quantum systems.
    """
    # First, generate the SU(n) generators
    SU_gens = generate_SU(n)  # Assuming generate_SU(n) returns n^2 - 1 generators

    # Initialize the U(n) generators array with n^2 generators
    U_gens = np.zeros((n**2, n, n), dtype=np.complex128)

    # Copy the SU(n) generators into the U(n) array
    U_gens[:n**2 - 1] = SU_gens

    # Add the additional U(1) generator
    U_gens[-1] = np.eye(n) * 1j  # i * I

    return U_gens

def generate_O(n):
    """
    Generate the generators of the orthogonal group O(n).

    The generators are n x n skew-symmetric matrices, which form a basis for
    the Lie algebra o(n) of O(n).

    Parameters
    ----------
    n : int
        The dimension of the group.

    Returns
    -------
    array-like
        A set of generators for O(n).
    """

    dim = int(n * (n - 1) / 2)
    gens = np.zeros((dim, n, n))
    ij_pair = [(i, j) for i in range(n) for j in range(n) if i < j]

    # Construct the skew-symmetric generators
    for a, (i, j) in enumerate(ij_pair):
        gens[a, i, j] = 1
        gens[a, j, i] = -1

    return gens









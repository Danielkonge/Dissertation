#!/usr/bin/env python3

import itertools
import numpy as np
from sympy import *
from sympy.matrices.normalforms import smith_normal_form

# Example of initialization for I in SL_3(Z_p):
# from sympy import *
# e_sl3 = np.array([1,2,3,4,5,6,7,8], dtype=np.int32)
# e_sort_3 = np.array([0,3,4,6,7,1,2,5], dtype=np.int32)
# # 1-->0, 2-->3, 3-->4, 4-->6, 5-->7, 6-->1, 7-->2, 8-->5


# NOTE: Throughout, we use a tuple to represent a wedge product.


def non_zero_wedge(tup, g):
    """
    Return boolean describing whether tup is a non-zero wedge.

    Input:
    tup = a tuple of integers,
    g = np.array([g1,g2,...]) with graded parts of the Lie algebra g
    where g1,g2,... are numbers 1,2,... corresponding
    to a basis e_1,e_2,... of g

    Output:
    Boolean

    Description:
    True if tup contains more k's than dim(g_k)
    for any k from range(1, len(g)+1),
    where g_k = g[k-1],
    which implies that the wedge product is zero.
    False otherwise.
    """
    return all(tup.count(k) <= len(g[k - 1]) for k in range(1, len(g) + 1))


def grj_nwedge_g(j, n, g):
    """
    Return gradings for grade j of n wedges of g.

    Input:
    j = grading (≥1),
    n = number of wedges (≥1),
    g = np.array([g1,g2,...]) with graded parts of the Lie algebra g
    where g1,g2,... are lists of numbers 1,2,... corresponding
    to a basis e_1,e_2,... of g

    Output:
    Returns an np.array of lists of which n wedges gives grading j,
    each inner list correspond to a direct summand.

    Description:
    Goes through all ordered combinations (with repeats)
    of n numbers from 1 to len(g),
    and checks whether they give a non-zero wedge,
    and whether they have grading j alltogether.
    Also, ignores trivial cases.
    """
    if n <= 0 or j < n:
        return []

    base_gradings = [
        i
        for i in itertools.combinations_with_replacement(range(1, len(g) + 1), n)
        if non_zero_wedge(i, g) and sum(i) == j
    ]
    return base_gradings


def flatten_basis(basis):
    """
    Flattens a list of lists.

    Input:
    A basis in the form of
    np.array([[basis-elements-1],[basis-elements-2],...]),
    where each inner list corresponds to a direct summand.

    Output:
    A flat basis in the form
    np.array([basis-elements-1,basis-elements-2,...])

    Description:
    Flattens bases of direct sums to a list of all basis elements
    without separate lists for the direct summands.
    """
    return [item for direct_summand in basis for item in direct_summand]


def grj_nwedge_g_basis(j, n, g):
    """
    Return basis for grade j of n wedges of g.

    Input:
    j = grading (≥1),
    n = number of wedges (≥1),
    g = np.array([g1,g2,...]) with graded parts of the Lie algebra g
    where g1,g2,... are lists of numbers 1,2,... corresponding
    to a basis e_1,e_2,... of g

    Output:
    Returns an np.array of lists of basis elements for
    n wedges of g with grading j.
    Each inner list correspond to a direct summand.

    Description:
    Goes through all gradings (grade j with n wedges),
    counts the number of g_i for each i,
    finds basis for wedges of g_i for each i,
    puts the above together to a basis of
    grade j of n wedges of g.
    Finally, prettify the basis.
    """
    base_gradings = grj_nwedge_g(j, n, g)
    basis = []
    for grading in base_gradings:
        grade_count = [grading.count(i) for i in range(1, len(g) + 1)]
        basis_grading_indices = [
            tuple(itertools.combinations(range(len(g[i])), grade_count[i]))
            for i in range(len(g))
        ]
        grading_basis_ugly = itertools.product(*basis_grading_indices)
        grading_basis_pretty = []
        for base in grading_basis_ugly:
            pretty_base_element = []
            for k in range(len(g)):
                for index in base[k]:
                    pretty_base_element.append(g[k][index])
            grading_basis_pretty.append(tuple(pretty_base_element))
        basis.append(grading_basis_pretty)
    return np.array(flatten_basis(basis), dtype=np.int32)


def store_bases(jmax, nmax, g):
    """
    Store the all bases in an np.array.

    Input:
    jmax = max j to try when finding bases
    nmax = max n to try when finding bases
    g = np.array([g1,g2,...]) with graded parts of the Lie algebra g
    where g1,g2,... are lists of numbers 1,2,... corresponding
    to a basis e_1,e_2,... of g

    Output:
    An np.array with a basis for grade j of n wedges of g
    in entry array[j][n].

    Description:
    Goes through j from 0 to jmax and n from 0 to nmax,
    and saves the bases of grade j of n wedges of g.
    """
    bases_array = np.empty((jmax, nmax), dtype=object)
    for j in range(jmax):
        for n in range(nmax):
            bases_array[j][n] = grj_nwedge_g_basis(j, n, g)
    return bases_array


# Make sure to implement commutator(a,b) in each example.
# E.g. for I in SL_3(Z_p):
# def commutator(a, b):
# sign = 1
# if a > b:
#     a_tmp = a
#     a = b
#     b = a_tmp
#     sign = -1
# # to make it easier later, we return lists with the answers, where a x+y = [x,y]
# if a==1 and b==6:
#     return [(-sign,2)]
# elif a==1 and b==7:
#     return [(sign,3)]
# elif a==1 and b==8:
#     return [(-sign,5),(-sign,4)] # for the sake of later code, we order these in reverse order
# elif a==2 and b==7:
#     return [(-sign,4)]
# elif a==3 and b==6:
#     return [(-sign,5)]
# elif a==6 and b==7:
#     return [(-sign,8)]
# else:
#     return [(0,0)]

# Here we use a list to represent a sum, which will work with our implementations.


def base_sort(g, eis, e_sort):
    """
    Sort our basis to the order we prefer.

    Input:
    g = np.array([g1,g2,...]) with graded parts of the Lie algebra g
    where g1,g2,... are lists of numbers 1,2,... corresponding
    to a basis e_1,e_2,... of g,
    eis = a tuple (wedge product) of e_i's (basis elements) out of order
    with the first entry being the coefficient,
    e_sort = an np.array describing our ordering of the e_i's

    Output:
    A tuple of the e_i's sorted.

    Description:
    We use that, when constructing the e_i's, we can only really
    mess up the order of one element, so we just linearly check
    the order of the elements, and compare to e_sort.
    Remembering to change signs, when moving elements past wedges.
    Recall that a tuple corresponds to a wedge product.
    """
    eis = list(eis)
    sign = 1
    for j in range(1, len(eis) - 1):
        if e_sort[eis[j] - 1] > e_sort[eis[j + 1] - 1]:
            eis[j], eis[j + 1] = eis[j + 1], eis[j]
            # Equivalent to:
            # tmp = eis[j]
            # eis[j] = eis[j+1]
            # eis[j+1] = tmp
            sign *= -1
    # Put coefficient on first entry
    eis[0] = sign * eis[0]
    return tuple(eis)


def d(g, eis, e_sort, commutator):
    """
    Calculate the d(eis) in the chain complex.

    Input:
    g = np.array([g1,g2,...]) with graded parts of the Lie algebra g
    where g1,g2,... are lists of numbers 1,2,... corresponding
    to a basis e_1,e_2,... of g,
    eis = a tuple (wedge product) of e_i's (basis elements)
    with the first entry being the coefficient,
    e_sort = an np.array describing our ordering of the e_i's

    Output:
    A list of tuples of the e_i's sorted.
    Elements of the list correspond to summands.
    We put the coefficient on the first element of each tuple.

    Description:
    Calculate d(eis) for the wedge product eis of e_i's,
    while making sure to place the coefficient in the first entry,
    and keeping the e_i's in our desired order.
    NOTE: Remember to implement commutator(a, b) first.
    """
    d_eis = []
    eis = np.array(eis)
    d_eis_elem = np.zeros(len(eis) - 1, dtype=np.int32)
    for j in range(2, len(eis)):
        for i in range(1, j):
            com = commutator(eis[i], eis[j])
            # Use that we put com[0][0] = 0, if the commutator is 0.
            if com[0][0] != 0:
                sign = (-1) ** (i + j)
                d_eis_elem_tmp = [
                    eis[k] for k in range(1, len(eis)) if k != i and k != j
                ]
                # Check if wedge product should be zero, because of repeat e_i
                for k in range(len(com)):
                    if not com[k][1] in d_eis_elem:
                        d_eis_elem[0] = sign * com[k][0] * eis[0]
                        d_eis_elem[1] = com[k][1]
                        d_eis_elem[2:] = d_eis_elem_tmp
                        d_eis.append(base_sort(g, d_eis_elem, e_sort))
    return d_eis


def d_coefs(g, eis, e_sort, codomain_basis, commutator):
    """
    Calculate the coefficients describing the map d in the chain complex.

    Input:
    g = np.array([g1,g2,...]) with graded parts of the Lie algebra g
    where g1,g2,... are lists of numbers 1,2,... corresponding
    to a basis e_1,e_2,... of g,
    eis = a tuple (wedge product) of e_i's (basis elements)
    with the first entry being the coefficient,
    e_sort = an np.array describing our ordering of the e_i's,
    codomain_basis = basis of codomain of d.

    Output:
    An np.array of coefficients for d(eis) in the basis of
    the codomain given by basis in the input.

    Description:
    Return an empty np.array if d(eis) = 0 (trivially).
    Otherwise, calculate d(eis), and find the coefficients
    of d(eis) in codomain_basis.
    """
    # len(eis) <= 2 instead of 1, since the first entry is just the coefficient
    if len(codomain_basis) == 0 or len(eis) <= 2:
        return np.array([], dtype=np.int32)
    d_eis = d(g, eis, e_sort, commutator)
    coefs = np.zeros(len(codomain_basis), dtype=np.int32)
    for d_ei in d_eis:
        coefs[np.where(np.all(codomain_basis == np.array(d_ei[1:]), axis=1))] = d_ei[0]
    return coefs


def add_one_coef(arr):
    """
    Add one (first) entry to arr with 1.

    Input:
    An np.array of size n.

    Output:
    An np.array of size n+1 with 1 in the first entry.

    Description:
    Construct a new array of size n+1 with all 1's.
    Change the last n entries to equal arr.
    """
    arr_with_one = np.ones(len(arr) + 1, dtype=np.int32)
    arr_with_one[1:] = arr
    return arr_with_one


def d_matrix_grj_nwedge_g(j, n, g, e_sort, bases, commutator):
    """
    Return transpose of matrix d out from grade j of n wedges of g.

    Input:
    j = grading (≥1),
    n = number of wedges (≥1),
    g = np.array([g1,g2,...]) with graded parts of the Lie algebra g
    where g1,g2,... are lists of numbers 1,2,... corresponding
    to a basis e_1,e_2,... of g,
    e_sort = an np.array describing our ordering of the e_i's,
    bases = np.array with all bases in any grade any wedges.

    Output:
    Returns an np.array for the transpose matrix of the map d from
    grade j of n wedges of g to grade j of n-1 wedges of g, or
    equivalently,
    the matrix describing the map d^T
    from grade -j of Hom(n-1 wedges of g,k)
    to grade -j of Hom(n wedges of g, k).

    Description:
    First compute the bases of the codomain and domain of d,
    then (unless trivial)
    """
    codomain_basis = bases[j][n - 1]
    domain_basis = bases[j][n]
    if len(codomain_basis) == 0 or len(domain_basis) == 0:
        return np.array([], dtype=np.int32)
    # Note, that our basis doesn't contain coefficients,
    # so we add the coefficient 1 to eis
    matrix = [
        d_coefs(g, add_one_coef(eis), e_sort, codomain_basis, commutator)
        for eis in domain_basis
        if len(eis) > 1
    ]
    return np.array(matrix, dtype=np.int32)


def store_d_matrices(jmax, nmax, g, e_sort, bases, commutator):
    """
    Store the all the matrices describing d in an np.array.

    Input:
    jmax = max j to try when finding bases,
    nmax = max n to try when finding bases,
    g = np.array([g1,g2,...]) with graded parts of the Lie algebra g
    where g1,g2,... are lists of numbers 1,2,... corresponding
    to a basis e_1,e_2,... of g,
    e_sort = an np.array describing our ordering of the e_i's,
    bases = np.array with all bases in any grade any wedges.

    Output:
    An np.array with a matrix to grade -j of Hom(n wedges of g, k)
    in entry array[j][n].

    Description:
    Goes through j from 0 to jmax and n from 0 to nmax,
    and saves the matrix d to grade -j of Hom(n wedges of g, k).
    """
    d_matrices_array = np.empty((jmax, nmax), dtype=object)
    d_matrices_array[0][0] = np.array([0], dtype=np.int32)
    for n in range(1, nmax):
        d_matrices_array[0][n] = np.array([], dtype=np.int32)
    for j in range(1, jmax):
        d_matrices_array[j][0] = np.array([], dtype=np.int32)
        for n in range(1, nmax):
            d_matrices_array[j][n] = d_matrix_grj_nwedge_g(
                j, n, g, e_sort, bases, commutator
            )
    return d_matrices_array


def smith_form_matrices(matrices):
    """
    Calculate Smith Normal Form for each matrix in a rectangular np.array of matrices.

    Input:
    A rectangular np.array of matrices.

    Output:
    A rectangular np.array of corresponding matrices in Smith Normal Form.

    Description:
    Go through each entry and calculate the Smith Normal Form for non-empty
    matrices. For empty matrices, just return them.
    """
    smith_matrices = np.empty(matrices.shape, dtype=object)
    m, n = matrices.shape
    for i in range(m):
        for j in range(n):
            if len(matrices[i][j]) != 0:
                smith_matrices[i][j] = smith_normal_form(
                    Matrix(matrices[i][j]), domain=ZZ
                )
            else:
                smith_matrices[i][j] = Matrix(np.array([], dtype=np.int32))
    return smith_matrices

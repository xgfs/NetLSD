import numpy as np
import scipy.linalg as spl
import scipy.sparse as sps
import scipy.sparse.linalg as spsl


def check_1d(inp):
    """
    Check input to be a vector. Converts lists to np.ndarray.

    Parameters
    ----------
    inp : obj
        Input vector

    Returns
    -------
    numpy.ndarray or None
        Input vector or None

    Examples
    --------
    >>> check_1d([0, 1, 2, 3])
    [0, 1, 2, 3]

    >>> check_1d('test')
    None

    """
    if isinstance(inp, list):
        return check_1d(np.array(inp))
    if isinstance(inp, np.ndarray):
        if inp.ndim == 1: # input is a vector
            return inp


def check_2d(inp):
    """
    Check input to be a matrix. Converts lists of lists to np.ndarray.

    Also allows the input to be a scipy sparse matrix.
    
    Parameters
    ----------
    inp : obj
        Input matrix

    Returns
    -------
    numpy.ndarray, scipy.sparse or None
        Input matrix or None

    Examples
    --------
    >>> check_2d([[0, 1], [2, 3]])
    [[0, 1], [2, 3]]

    >>> check_2d('test')
    None

    """
    if isinstance(inp, list):
        return check_2d(np.array(inp))
    if isinstance(inp, (np.ndarray, np.matrixlib.defmatrix.matrix)):
        if inp.ndim == 2: # input is a dense matrix
            return inp
    if sps.issparse(inp):
        if inp.ndim == 2: # input is a sparse matrix
            return inp


def graph_to_laplacian(G, normalized=True):
    """
    Converts a graph from popular Python packages to Laplacian representation.

    Currently support NetworkX, graph_tool and igraph.
    
    Parameters
    ----------
    G : obj
        Input graph
    normalized : bool
        Whether to use normalized Laplacian.
        Normalized and unnormalized Laplacians capture different properties of graphs, e.g. normalized Laplacian spectrum can determine whether a graph is bipartite, but not the number of its edges. We recommend using normalized Laplacian.

    Returns
    -------
    scipy.sparse
        Laplacian matrix of the input graph

    Examples
    --------
    >>> graph_to_laplacian(nx.complete_graph(3), 'unnormalized').todense()
    [[ 2, -1, -1], [-1,  2, -1], [-1, -1,  2]]

    >>> graph_to_laplacian('test')
    None

    """
    try:
        import networkx as nx
        if isinstance(G, nx.Graph):
            if normalized:
                return nx.normalized_laplacian_matrix(G)
            else:
                return nx.laplacian_matrix(G)
    except ImportError:
        pass
    try:
        import graph_tool.all as gt
        if isinstance(G, gt.Graph):
            if normalized:
                return gt.laplacian_type(G, normalized=True)
            else:
                return gt.laplacian(G)
    except ImportError:
        pass
    try:
        import igraph as ig
        if isinstance(G, ig.Graph):
            if normalized:
                return np.array(G.laplacian(normalized=True))
            else:
                return np.array(G.laplacian())
    except ImportError:
        pass


def mat_to_laplacian(mat, normalized):
    """
    Converts a sparse or dence adjacency matrix to Laplacian.
    
    Parameters
    ----------
    mat : obj
        Input adjacency matrix. If it is a Laplacian matrix already, return it.
    normalized : bool
        Whether to use normalized Laplacian.
        Normalized and unnormalized Laplacians capture different properties of graphs, e.g. normalized Laplacian spectrum can determine whether a graph is bipartite, but not the number of its edges. We recommend using normalized Laplacian.

    Returns
    -------
    obj
        Laplacian of the input adjacency matrix

    Examples
    --------
    >>> mat_to_laplacian(numpy.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]), False)
    [[ 2, -1, -1], [-1,  2, -1], [-1, -1,  2]]

    """
    if sps.issparse(mat):
        if np.all(mat.diagonal()>=0): # Check diagonal
            if np.all((mat-sps.diags(mat.diagonal())).data <= 0): # Check off-diagonal elements
                return mat
    else:
        if np.all(np.diag(mat)>=0): # Check diagonal
            if np.all(mat - np.diag(mat) <= 0): # Check off-diagonal elements
                return mat
    deg = np.squeeze(np.asarray(mat.sum(axis=1)))
    if sps.issparse(mat):
        L = sps.diags(deg) - mat
    else:
        L = np.diag(deg) - mat
    if not normalized:
        return L
    with np.errstate(divide='ignore'):
        sqrt_deg = 1.0 / np.sqrt(deg)
    sqrt_deg[sqrt_deg==np.inf] = 0
    if sps.issparse(mat):
        sqrt_deg_mat = sps.diags(sqrt_deg)
    else:
        sqrt_deg_mat = np.diag(sqrt_deg)
    return sqrt_deg_mat.dot(L).dot(sqrt_deg_mat)


def updown_linear_approx(eigvals_lower, eigvals_upper, nv):
    """
    Approximates Laplacian spectrum using upper and lower parts of the eigenspectrum.
    
    Parameters
    ----------
    eigvals_lower : numpy.ndarray
        Lower part of the spectrum, sorted
    eigvals_upper : numpy.ndarray
        Upper part of the spectrum, sorted
    nv : int
        Total number of nodes (eigenvalues) in the graph.

    Returns
    -------
    numpy.ndarray
        Vector of approximated eigenvalues

    Examples
    --------
    >>> updown_linear_approx([1, 2, 3], [7, 8, 9], 9)
    array([1,  2,  3,  4,  5,  6,  7,  8,  9])

    """
    nal = len(eigvals_lower)
    nau = len(eigvals_upper)
    if nv < nal + nau:
        raise ValueError('Number of supplied eigenvalues ({0} lower and {1} upper) is higher than number of nodes ({2})!'.format(nal, nau, nv))
    ret = np.zeros(nv)
    ret[:nal] = eigvals_lower
    ret[-nau:] = eigvals_upper
    ret[nal-1:-nau+1] = np.linspace(eigvals_lower[-1], eigvals_upper[0], nv-nal-nau+2)
    return ret


def eigenvalues_auto(mat, n_eivals='auto'):
    """
    Automatically computes the spectrum of a given Laplacian matrix.
    
    Parameters
    ----------
    mat : numpy.ndarray or scipy.sparse
        Laplacian matrix
    n_eivals : string or int or tuple
        Number of eigenvalues to compute / use for approximation.
        If string, we expect either 'full' or 'auto', otherwise error will be raised. 'auto' lets the program decide based on the faithful usage. 'full' computes all eigenvalues.
        If int, compute n_eivals eigenvalues from each side and approximate using linear growth approximation.
        If tuple, we expect two ints, first for lower part of approximation, and second for the upper part.

    Returns
    -------
    np.ndarray
        Vector of approximated eigenvalues

    Examples
    --------
    >>> eigenvalues_auto(numpy.array([[ 2, -1, -1], [-1,  2, -1], [-1, -1,  2]]), 'auto')
    array([0, 3, 3])

    """
    do_full = True
    n_lower = 150
    n_upper = 150
    nv = mat.shape[0]
    if n_eivals == 'auto':
        if mat.shape[0] > 1024:
            do_full = False
    if n_eivals == 'full':
        do_full = True
    if isinstance(n_eivals, int):
        n_lower = n_upper = n_eivals
        do_full = False
    if isinstance(n_eivals, tuple):
        n_lower, n_upper = n_eivals
        do_full = False
    if do_full and sps.issparse(mat):
        mat = mat.todense()
    if sps.issparse(mat):
        if n_lower == n_upper:
            tr_eivals = spsl.eigsh(mat, 2*n_lower, which='BE', return_eigenvectors=False)
            return updown_linear_approx(tr_eivals[:n_upper], tr_eivals[n_upper:], nv)
        else:
            lo_eivals = spsl.eigsh(mat, n_lower, which='SM', return_eigenvectors=False)[::-1]
            up_eivals = spsl.eigsh(mat, n_upper, which='LM', return_eigenvectors=False)
            return updown_linear_approx(lo_eivals, up_eivals, nv)
    else:
        if do_full:
            return spl.eigvalsh(mat)
        else:
            lo_eivals = spl.eigvalsh(mat, eigvals=(0, n_lower-1))
            up_eivals = spl.eigvalsh(mat, eigvals=(nv-n_upper-1, nv-1))
            return updown_linear_approx(lo_eivals, up_eivals, nv)
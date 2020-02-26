import numpy as np
from .util import check_1d, check_2d, eigenvalues_auto, graph_to_laplacian, mat_to_laplacian


def compare(descriptor1, descriptor2):
    """
    Computes the distance between two NetLSD representations.
    
    Parameters
    ----------
    descriptor1: numpy.ndarray
        First signature to compare
    descriptor2: numpy.ndarray
        Second signature to compare

    Returns
    -------
    float
        NetLSD distance

    """
    return np.linalg.norm(descriptor1-descriptor2)


def netlsd(inp, timescales=np.logspace(-2, 2, 250), kernel='heat', eigenvalues='auto', normalization='empty', normalized_laplacian=True):
    """
    Computes NetLSD signature from some given input, timescales, and normalization.

    Accepts matrices, common Python graph libraries' graphs, or vectors of eigenvalues. 
    For precise definition, please refer to "NetLSD: Hearing the Shape of a Graph" by A. Tsitsulin, D. Mottin, P. Karras, A. Bronstein, E. Müller. Published at KDD'18.
    
    Parameters
    ----------
    inp: obj
        2D numpy/scipy matrix, common Python graph libraries' graph, or vector of eigenvalues
    timescales : numpy.ndarray
        Vector of discrete timesteps for the kernel computation
    kernel : str
        Either 'heat' or 'wave'. Type of a kernel to use for computation.
    eigenvalues : str
        Either string or int or tuple
        Number of eigenvalues to compute / use for approximation.
        If string, we expect either 'full' or 'auto', otherwise error will be raised. 'auto' lets the program decide based on the faithful usage. 'full' computes all eigenvalues.
        If int, compute n_eivals eigenvalues from each side and approximate using linear growth approximation.
        If tuple, we expect two ints, first for lower part of approximation, and second for the upper part.
    normalization : str or numpy.ndarray
        Either 'empty', 'complete' or None.
        If None or any ther value, return unnormalized heat kernel trace.
        For the details how 'empty' and 'complete' are computed, please refer to the paper.
        If np.ndarray, they are treated as exact normalization constants
    normalized_laplacian: bool
        Defines whether the eigenvalues came from the normalized Laplacian. It only affects 'complete' normalization.

    Returns
    -------
    numpy.ndarray
        NetLSD signature

    """
    if kernel not in {'heat', 'wave'}:
        raise AttributeError('Unirecognized kernel type: expected one of [\'heat\', \'wave\'], got {0}'.format(kernel))
    if not isinstance(normalized_laplacian, bool):
        raise AttributeError('Unknown Laplacian type: expected bool, got {0}'.format(normalized_laplacian))
    if not isinstance(eigenvalues, (int, tuple, str)):
        raise AttributeError('Unirecognized requested eigenvalue number: expected type of [\'str\', \'tuple\', or \'int\'], got {0}'.format(type(eigenvalues)))
    if not isinstance(timescales, np.ndarray):
        raise AttributeError('Unirecognized timescales data type: expected np.ndarray, got {0}'.format(type(timescales)))
    if timescales.ndim != 1:
        raise AttributeError('Unirecognized timescales dimensionality: expected a vector, got {0}-d array'.format(timescales.ndim))
    if normalization not in {'complete', 'empty', 'none', True, False, None}:
        if not isinstance(normalization, np.ndarray):
            raise AttributeError('Unirecognized normalization type: expected one of [\'complete\', \'empty\', None or np.ndarray], got {0}'.format(normalization))
        if normalization.ndim != 1:
            raise AttributeError('Unirecognized normalization dimensionality: expected a vector, got {0}-d array'.format(normalization.ndim))
        if timescales.shape[0] != normalization.shape[0]:
            raise AttributeError('Unirecognized normalization dimensionality: expected {0}-length vector, got length {1}'.format(timescales.shape[0], normalization.shape[0]))

    eivals = check_1d(inp)
    if eivals is None:
        mat = check_2d(inp)
        if mat is None:
            mat = graph_to_laplacian(inp, normalized_laplacian)
            if mat is None:
                raise ValueError('Unirecognized input type: expected one of [\'np.ndarray\', \'scipy.sparse\', \'networkx.Graph\',\' graph_tool.Graph,\' or \'igraph.Graph\'], got {0}'.format(type(inp)))
        else:
            mat = mat_to_laplacian(inp, normalized_laplacian)
        eivals = eigenvalues_auto(mat, eigenvalues)
    if kernel == 'heat':
        return _hkt(eivals, timescales, normalization, normalized_laplacian)
    else:
        return _wkt(eivals, timescales, normalization, normalized_laplacian)


def heat(inp, timescales=np.logspace(-2, 2, 250), eigenvalues='auto', normalization='empty', normalized_laplacian=True):
    """
    Computes heat kernel trace from some given input, timescales, and normalization.

    Accepts matrices, common Python graph libraries' graphs, or vectors of eigenvalues. 
    For precise definition, please refer to "NetLSD: Hearing the Shape of a Graph" by A. Tsitsulin, D. Mottin, P. Karras, A. Bronstein, E. Müller. Published at KDD'18.
    
    Parameters
    ----------
    inp: obj
        2D numpy/scipy matrix, common Python graph libraries' graph, or vector of eigenvalues
    timescales : numpy.ndarray
        Vector of discrete timesteps for the kernel computation
    eigenvalues : str
        Either string or int or tuple
        Number of eigenvalues to compute / use for approximation.
        If string, we expect either 'full' or 'auto', otherwise error will be raised. 'auto' lets the program decide based on the faithful usage. 'full' computes all eigenvalues.
        If int, compute n_eivals eigenvalues from each side and approximate using linear growth approximation.
        If tuple, we expect two ints, first for lower part of approximation, and second for the upper part.
    normalization : str or numpy.ndarray
        Either 'empty', 'complete' or None.
        If None or any ther value, return unnormalized heat kernel trace.
        For the details how 'empty' and 'complete' are computed, please refer to the paper.
        If np.ndarray, they are treated as exact normalization constants
    normalized_laplacian: bool
        Defines whether the eigenvalues came from the normalized Laplacian. It only affects 'complete' normalization.

    Returns
    -------
    numpy.ndarray
        Heat kernel trace signature

    """
    return netlsd(inp, timescales, 'heat', eigenvalues, normalization, normalized_laplacian)


def wave(inp, timescales=np.linspace(0, 2*np.pi, 250), eigenvalues='auto', normalization='empty', normalized_laplacian=True):
    """
    Computes wave kernel trace from some given input, timescales, and normalization.

    Accepts matrices, common Python graph libraries' graphs, or vectors of eigenvalues. 
    For precise definition, please refer to "NetLSD: Hearing the Shape of a Graph" by A. Tsitsulin, D. Mottin, P. Karras, A. Bronstein, E. Müller. Published at KDD'18.
    
    Parameters
    ----------
    inp: obj
        2D numpy/scipy matrix, common Python graph libraries' graph, or vector of eigenvalues
    timescales : numpy.ndarray
        Vector of discrete timesteps for the kernel computation
    eigenvalues : str
        Either string or int or tuple
        Number of eigenvalues to compute / use for approximation.
        If string, we expect either 'full' or 'auto', otherwise error will be raised. 'auto' lets the program decide based on the faithful usage. 'full' computes all eigenvalues.
        If int, compute n_eivals eigenvalues from each side and approximate using linear growth approximation.
        If tuple, we expect two ints, first for lower part of approximation, and second for the upper part.
    normalization : str or numpy.ndarray
        Either 'empty', 'complete' or None.
        If None or any ther value, return unnormalized wave kernel trace.
        For the details how 'empty' and 'complete' are computed, please refer to the paper.
        If np.ndarray, they are treated as exact normalization constants
    normalized_laplacian: bool
        Defines whether the eigenvalues came from the normalized Laplacian. It only affects 'complete' normalization.

    Returns
    -------
    numpy.ndarray
        Wave kernel trace signature

    """
    return netlsd(inp, timescales, 'wave', eigenvalues, normalization, normalized_laplacian)


def _hkt(eivals, timescales, normalization, normalized_laplacian):
    """
    Computes heat kernel trace from given eigenvalues, timescales, and normalization.

    For precise definition, please refer to "NetLSD: Hearing the Shape of a Graph" by A. Tsitsulin, D. Mottin, P. Karras, A. Bronstein, E. Müller. Published at KDD'18.
    
    Parameters
    ----------
    eivals : numpy.ndarray
        Eigenvalue vector
    timescales : numpy.ndarray
        Vector of discrete timesteps for the kernel computation
    normalization : str or numpy.ndarray
        Either 'empty', 'complete' or None.
        If None or any ther value, return unnormalized heat kernel trace.
        For the details how 'empty' and 'complete' are computed, please refer to the paper.
        If np.ndarray, they are treated as exact normalization constants
    normalized_laplacian: bool
        Defines whether the eigenvalues came from the normalized Laplacian. It only affects 'complete' normalization.

    Returns
    -------
    numpy.ndarray
        Heat kernel trace signature

    """
    nv = eivals.shape[0]
    hkt = np.zeros(timescales.shape)
    for idx, t in enumerate(timescales):
        hkt[idx] = np.sum(np.exp(-t * eivals))
    if isinstance(normalization, np.ndarray):
        return hkt / normalization
    if normalization == 'empty' or normalization == True:
        return hkt / nv
    if normalization == 'complete':
        if normalized_laplacian:
            return hkt / (1 + (nv - 1) * np.exp(-timescales))
        else:
            return hkt / (1 + nv * np.exp(-nv * timescales))
    return hkt


def _wkt(eivals, timescales, normalization, normalized_laplacian):
    """
    Computes wave kernel trace from given eigenvalues, timescales, and normalization.

    For precise definition, please refer to "NetLSD: Hearing the Shape of a Graph" by A. Tsitsulin, D. Mottin, P. Karras, A. Bronstein, E. Müller. Published at KDD'18.
    
    Parameters
    ----------
    eivals : numpy.ndarray
        Eigenvalue vector
    timescales : numpy.ndarray
        Vector of discrete timesteps for the kernel computation
    normalization : str or numpy.ndarray
        Either 'empty', 'complete' or None.
        If None or any ther value, return unnormalized wave kernel trace.
        For the details how 'empty' and 'complete' are computed, please refer to the paper.
        If np.ndarray, they are treated as exact normalization constants
    normalized_laplacian: bool
        Defines whether the eigenvalues came from the normalized Laplacian. It only affects 'complete' normalization.

    Returns
    -------
    numpy.ndarray
        Wave kernel trace signature

    """
    nv = eivals.shape[0]
    wkt = np.zeros(timescales.shape)
    for idx, t in enumerate(timescales):
        wkt[idx] = np.sum(np.exp(-1j * t * eivals))
    if isinstance(normalization, np.ndarray):
        return wkt / normalization
    if normalization == 'empty' or normalization == True:
        return wkt / nv
    if normalization == 'complete':
        if normalized_laplacian:
            return wkt / (1 + (nv - 1) * np.cos(timescales))
        else:
            return wkt / (1 + (nv - 1) * np.cos(nv * timescales))
    return wkt

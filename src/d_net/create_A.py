import numpy as np
from scipy import optimize, stats


def distance_A(A, D):
    """Takes distance matrix D and material attribute/category
    matrix A and finds the sum of the pairwise distances of
    each entry compared to D. This function should be used
    in an optimizer to generate an optimal A matrix.

    Implements first term of Equation 4

    Parameters:
    D: (k x k) matrix
        The D matrix generated using the d-cnn.
    A: (k x m) matrix
        The A matrix generated in the process of
        executing create_A. This is not necessarily
        the optimal version of A we are trying to seek.
        
    Returns:
    dist: float
        The sum of L2-norm distances between the rows of 
        the A matrix and the D matrix.
    """
    k    = D.shape[0]
    dist = 0.0
    
    for m in range(k):
        for n in range(k):
            l2_A = np.linalg.norm(A[m] - A[n])       
            dist += (l2_A - D[m][n])**2
            
    return dist



# TODO Need to look over this and rework the function to better fit the definition.
def kld_A(A, beta_a, beta_b):
    """Takes optimized material attribute/category matrix A and 
    finds the KL divergence between the A matrix distribution 
    and a beta distribution.

    Implements second term (gamma-weighted) Equation 4,
    and term for Equation 5

    Parameters:
        A: (k x m) matrix
            The optimal A matrix generated in the process of
            executing create_A.
        beta_a: float
            The beta distribution parameter a in (4)
        beta_b: float
            The beta distribution parameter b in (4)
            
    Returns:
        kld: float
            The KL-divergence between the Gaussian KDE of A
            and the beta distribution with a, b parameters.
    """
    
    # Use a grid of points to approximate the Beta distribution.
    # Points in range [0, 1] because that is the Beta distribution's range.
    # Start at 0.02, end at 0.98 to avoid infinite values
    points = np.mgrid[0.02: 0.98: 0.04]
    # points = np.vstack(np.meshgrid(x, x, x)).reshape(3, -1).T
  
    akde = stats.gaussian_kde(A)      # Gaussian kernel density estimate for A -> Eq. 5
    beta = stats.beta(beta_a, beta_b) # Beta distribution to compare akde to
    
    beta_pts = [p for p in points] # Beta distribution is one-dimensional
    beta_pts = [beta.pdf(p) for p in beta_pts]
    akde_pts = [akde(p) for p in points]
    akde_pts = np.squeeze(akde_pts)
    
    kld = stats.entropy(beta_pts, akde_pts) # KL-divergence -> Eq. 4 term 2
    return kld
    


def min_A(A, D, w_kld, beta_a, beta_b):
    """Uses distance_A and kld_A to implement the
    minimization objetive introduced in equation 4

    Parameters:
        D: (k x k) matrix
            The D matrix generated using the d-cnn.
        w_kld: float
            The hyperparameter / weight gamma in (4)
        beta_a: float
            The beta distribution parameter a in (4)
        beta_b: float
            The beta distribution parameter b in (4)
            
    Returns:
        dist: float
            The distance to be minimized in the minimization
            objective.
    """
    return distance_A(A,D) + w_kld * kld_A(A, beta_a, beta_b)



def create_A(D, m, w_kld = 1e-1, beta_a = 0.5, beta_b = 0.5):
    """Takes a material distance matrix D and runs it
    through the L-BFGS algorithm to create an optimal
    A matrix.

    Implements Equation 4

    Parameters:
        D: (k x k) matrix
            The D matrix generated using the d-cnn.
        m: int
            The number of material attribute categories desired
        w_kld: float
            The hyperparameter / weight gamma in (4)
        beta_a: float
            The beta distribution parameter a in (4)
            Default = 0.5 b/c it exhibits suitable distribution for
            widely fitting or widely not fitting material category
        beta_b: float
            The beta distribution parameter b in (4)
            Default = 0.5 b/c it exhibits suitable distribution for
            widely fitting or widely not fitting material category
            
    Returns:
        A: (k x m) matrix
            The **optimized** material attribute/category matrix 
            to be utilized by the MMAC-CNN.
    """
    
    k = D.shape[0]
    
    # Unoptimized A function constrained to the range [0, 1]
    A = np.random.random((k, m))
    
    print(f'\n--- Before optim: ---')
    print(f'D matrix :\n{D}')
    print(f'\nA matrix :\n{A}')
    print(f'\nd(D; A)  : {distance_A(A, D)}')
    
    # NOTE Sometimes the KL divergence blows up to inf and
    #      the resulting A matrix is useless. Especially with less M classes.
    #      If this happens, try to run the optimization code again.
    #
    #      Is there an issue with the A matrix random initialization?
    result = optimize.minimize(min_A, A, args=(D, w_kld, beta_a, beta_b), 
                               method = 'L-BFGS-B', callback = None)
    dist   = result.fun
    A      = result.x
    A = np.reshape(A, (k, m))
    
    # Normalize A matrix values to the range [0, 1] after optimization
    A = (A - np.min(A))/np.ptp(A)
    
    print(f'\n--- After optim: ---')
    print(f'A matrix :\n{A}')
    print(f'\nd(D; A) : {dist}')

    return A
    
   
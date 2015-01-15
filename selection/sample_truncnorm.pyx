import numpy as np, cython
cimport numpy as np

from libc.math cimport pow, sqrt, log, exp # sin, cos, acos, asin, sqrt, fabs
from scipy.special import ndtr, ndtri

cdef double PI = np.pi

"""
This module has a code to sample from a truncated normal distribution
specified by a set of affine constraints.
"""

DTYPE_float = np.float
ctypedef np.float_t DTYPE_float_t
DTYPE_int = np.int
ctypedef np.int_t DTYPE_int_t

@cython.boundscheck(False)
@cython.cdivision(True)
def sample_truncnorm_white(np.ndarray[DTYPE_float_t, ndim=2] A, 
                           np.ndarray[DTYPE_float_t, ndim=1] b, 
                           np.ndarray[DTYPE_float_t, ndim=1] initial, 
                           np.ndarray[DTYPE_float_t, ndim=1] bias_direction, #eta
                           DTYPE_int_t how_often=1000,
                           DTYPE_float_t sigma=1.,
                           DTYPE_int_t burnin=500,
                           DTYPE_int_t ndraw=1000,
			   int use_A=1
                           ):
    """
    Sample from a truncated normal with covariance
    equal to sigma**2 I.

    Constraint is $Ax \leq b$ where `A` has shape
    `(q,n)` with `q` the number of constraints and
    `n` the number of random variables.


    Parameters
    ----------

    A : np.float((q,n))
        Linear part of affine constraints.

    b : np.float(q)
        Offset part of affine constraints.

    initial : np.float(n)
        Initial point for Gibbs draws.
        Assumed to satisfy the constraints.

    bias_direction : np.float (optional)
        Which projection is of most interest?

    how_often : int (optional)
        How often should the sampler make a move along `direction_of_interest`?
        If negative, defaults to ndraw+burnin (so it will never be used).

    sigma : float
        Variance parameter.

    burnin : int
        How many iterations until we start
        recording samples?

    ndraw : int
        How many samples should we return?

    Returns
    -------

    trunc_sample : np.float((ndraw, n))

    """

    cdef int nvar = A.shape[1]
    cdef int nconstraint = A.shape[0]
    cdef np.ndarray[DTYPE_float_t, ndim=2] trunc_sample = \
            np.empty((ndraw, nvar), np.float)
    cdef np.ndarray[DTYPE_float_t, ndim=1] state = initial.copy()
    cdef int idx, iter_count, irow, ivar
    cdef double lower_bound, upper_bound, V
    cdef double cdfL, cdfU, unif, tnorm, val, alpha

    cdef double tol = 1.e-7

    cdef np.ndarray[DTYPE_float_t, ndim=1] U = np.dot(A, state) - b

    cdef np.ndarray[DTYPE_float_t, ndim=1] usample = \
        np.random.sample(burnin + ndraw)

    # directions not parallel to coordinate axes

    if use_A:
        _dirs = [A, 
                 np.random.standard_normal((int(nvar/5),nvar))]
    else:
        _dirs = [np.random.standard_normal((nvar,nvar))]

    cdef np.ndarray[DTYPE_float_t, ndim=2] directions = \
        np.vstack(_dirs)
        
    directions[-1][:] = bias_direction

    directions /= np.sqrt((directions**2).sum(1))[:,None]

    cdef int ndir = directions.shape[0]

    cdef np.ndarray[DTYPE_float_t, ndim=2] alphas_dir = \
        np.dot(A, directions.T)

    cdef np.ndarray[DTYPE_float_t, ndim=2] alphas_coord = A
        
    cdef np.ndarray[DTYPE_float_t, ndim=1] alphas_max_dir = \
        np.fabs(alphas_dir).max(0) * tol    

    cdef np.ndarray[DTYPE_float_t, ndim=1] alphas_max_coord = \
        np.fabs(alphas_coord).max(0) * tol 

    # choose the order of sampling (randomly)

    cdef np.ndarray[DTYPE_int_t, ndim=1] random_idx_dir = \
        np.random.random_integers(0, ndir-1, size=(burnin+ndraw,))

    cdef np.ndarray[DTYPE_int_t, ndim=1] random_idx_coord = \
        np.random.random_integers(0, nvar-1, size=(burnin+ndraw,))

    # for switching between coordinate updates and
    # other directions

    cdef int invperiod = 13
    cdef int docoord = 0
    cdef int iperiod = 0
    cdef int ibias = 0
    cdef int dobias = 0

    for iter_count in range(ndraw + burnin):

        docoord = 1
        iperiod = iperiod + 1
        ibias = ibias + 1

        if iperiod == invperiod: 
            docoord = 0
            iperiod = 0
            dobias = 0

        if ibias == how_often:
            docoord = 0
            ibias = 0
            dobias = 1
        
        if docoord == 1:
            idx = random_idx_coord[iter_count]
            V = state[idx]
        else:
            if not dobias:
                idx = random_idx_dir[iter_count]
            else:
                idx = directions.shape[0]-1 # last row of directions is bias_direction
            V = 0
            for ivar in range(nvar):
                V = V + directions[idx, ivar] * state[ivar]

        lower_bound = -1e12
        upper_bound = 1e12
        for irow in range(nconstraint):
            if docoord == 1:
                alpha = alphas_coord[irow,idx]
                val = -U[irow] / alpha + V
                if alpha > alphas_max_coord[idx] and (val < upper_bound):
                    upper_bound = val
                elif alpha < -alphas_max_coord[idx] and (val > lower_bound):
                    lower_bound = val
            else:
                alpha = alphas_dir[irow,idx]
                val = -U[irow] / alpha + V
                if alpha > alphas_max_dir[idx] and (val < upper_bound):
                    upper_bound = val
                elif alpha < -alphas_max_dir[idx] and (val > lower_bound):
                    lower_bound = val
        if lower_bound > V:
            lower_bound = V - tol * sigma
        elif upper_bound < V:
            upper_bound = V + tol * sigma

        lower_bound = lower_bound / sigma
        upper_bound = upper_bound / sigma

        if lower_bound > upper_bound:
            raise ValueError('bound violation')

        if upper_bound < -10: # use Exp approximation
            unif = usample[iter_count] * (1 - exp((
                        lower_bound - upper_bound) * upper_bound))
            tnorm = upper_bound - log(1 - unif) / upper_bound 
        elif lower_bound > 10:
            unif = usample[iter_count] * (1 - exp(-(
                        upper_bound - lower_bound) * lower_bound))
            tnorm = -log(1 - unif) / lower_bound + lower_bound
            #if tnorm < lower_bound:
            #    tnorm = lower_bound + 0.001 * (upper_bound - lower_bound)
        elif lower_bound < 0:
            cdfL = ndtr(lower_bound)
            cdfU = ndtr(upper_bound)
            unif = usample[iter_count] * (cdfU - cdfL) + cdfL
            if unif < 0.5:
                tnorm = ndtri(unif) * sigma
            else:
                tnorm = -ndtri(1-unif) * sigma
        else:
            cdfL = ndtr(-lower_bound)
            cdfU = ndtr(-upper_bound)
            unif = usample[iter_count] * (cdfL - cdfU) + cdfU
            if unif < 0.5:
                tnorm = -ndtri(unif) * sigma
            else:
                tnorm = ndtri(1-unif) * sigma
            
        if docoord == 1:
            state[idx] = tnorm
            tnorm = tnorm - V
            for irow in range(nconstraint):
                U[irow] = U[irow] + tnorm * A[irow, idx]
        else:
            tnorm = tnorm - V
            for ivar in range(nvar):
                state[ivar] = state[ivar] + tnorm * directions[idx,ivar]
                for irow in range(nconstraint):
                    U[irow] = (U[irow] + A[irow, ivar] * 
                               tnorm * directions[idx,ivar])

        if iter_count >= burnin:
            for ivar in range(nvar):
                trunc_sample[iter_count - burnin, ivar] = state[ivar]
        
    return trunc_sample

@cython.boundscheck(False)
@cython.cdivision(True)
def sample_truncnorm_white_ball(np.ndarray[DTYPE_float_t, ndim=2] A, 
                                np.ndarray[DTYPE_float_t, ndim=1] b, 
                                np.ndarray[DTYPE_float_t, ndim=1] initial, 
                                np.ndarray[DTYPE_float_t, ndim=1] bias_direction, 
                                sample_squared_radius, 
                                DTYPE_int_t how_often=1000,
                                DTYPE_int_t burnin=500,
                                DTYPE_int_t ndraw=1000,
                                ):
    """
    Sample from a ball of radius `np.linalg.norm(initial)`
    intersected with a constraint.

    Constraint is $Ax \leq b$ where `A` has shape
    `(q,n)` with `q` the number of constraints and
    `n` the number of random variables.

    Parameters
    ----------

    A : np.float((q,n))
        Linear part of affine constraints.

    b : np.float(q)
        Offset part of affine constraints.

    initial : np.float(n)
        Initial point for Gibbs draws.
        Assumed to satisfy the constraints.

    sample_squared_radius : callable
        A callable that takes argument the state and 
        returns a draw from distribution of the 
        square of the radius. 

    bias_direction : np.float (optional)
        Which projection is of most interest?

    how_often : int (optional)
        How often should the sampler make a move along `direction_of_interest`?
        If negative, defaults to ndraw+burnin (so it will never be used).

    sigma : float
        Variance parameter.

    burnin : int
        How many iterations until we start
        recording samples?

    ndraw : int
        How many samples should we return?

    Returns
    -------

    trunc_sample : np.float((ndraw, n))

    """

    cdef int nvar = A.shape[1]
    cdef int nconstraint = A.shape[0]
    cdef np.ndarray[DTYPE_float_t, ndim=2] trunc_sample = \
            np.empty((ndraw, nvar), np.float)
    cdef np.ndarray[DTYPE_float_t, ndim=1] state = initial.copy()
    cdef int idx, iter_count, irow, ivar
    cdef double lower_bound, upper_bound, V
    cdef double cdfL, cdfU, unif, tval, val, alpha
    cdef double norm_state_bound = sample_squared_radius(state)
    cdef double norm_state_sq = norm_state_bound

    cdef double tol = 1.e-7

    cdef np.ndarray[DTYPE_float_t, ndim=1] U = np.dot(A, state) - b

    cdef np.ndarray[DTYPE_float_t, ndim=1] usample = \
        np.random.sample(burnin + ndraw)

    # directions not parallel to coordinate axes

    cdef np.ndarray[DTYPE_float_t, ndim=2] directions = \
        np.vstack([A, 
                   np.random.standard_normal((int(nvar/5),nvar))])
    directions[-1][:] = bias_direction

    directions /= np.sqrt((directions**2).sum(1))[:,None]

    cdef int ndir = directions.shape[0]

    cdef np.ndarray[DTYPE_float_t, ndim=2] alphas_dir = \
        np.dot(A, directions.T)

    cdef np.ndarray[DTYPE_float_t, ndim=2] alphas_coord = A
        
    cdef np.ndarray[DTYPE_float_t, ndim=1] alphas_max_dir = \
        np.fabs(alphas_dir).max(0) * tol    

    cdef np.ndarray[DTYPE_float_t, ndim=1] alphas_max_coord = \
        np.fabs(alphas_coord).max(0) * tol 

    # choose the order of sampling (randomly)

    cdef np.ndarray[DTYPE_int_t, ndim=1] random_idx_dir = \
        np.random.random_integers(0, ndir-1, size=(burnin+ndraw,))

    cdef np.ndarray[DTYPE_int_t, ndim=1] random_idx_coord = \
        np.random.random_integers(0, nvar-1, size=(burnin+ndraw,))

    # for switching between coordinate updates and
    # other directions

    cdef int invperiod = 13
    cdef int docoord = 0
    cdef int iperiod = 0
    cdef int ibias = 0
    cdef int dobias = 0
    cdef double discriminant, multiplier

    for iter_count in range(ndraw + burnin):

        docoord = 1
        iperiod = iperiod + 1
        ibias = ibias + 1

        if iperiod == invperiod: 
            docoord = 0
            iperiod = 0
            dobias = 0

        if ibias == how_often:
            docoord = 0
            ibias = 0
            dobias = 1
        
        if docoord == 1:
            idx = random_idx_coord[iter_count]
            V = state[idx]
        else:
            if not dobias:
                idx = random_idx_dir[iter_count]
            else:
                idx = directions.shape[0]-1 # last row of directions is bias_direction
            V = 0
            for ivar in range(nvar):
                V = V + directions[idx, ivar] * state[ivar]

        lower_bound = -1e12
        upper_bound = 1e12
        for irow in range(nconstraint):
            if docoord == 1:
                alpha = alphas_coord[irow,idx]
                val = -U[irow] / alpha + V
                if alpha > alphas_max_coord[idx] and (val < upper_bound):
                    upper_bound = val
                elif alpha < -alphas_max_coord[idx] and (val > lower_bound):
                    lower_bound = val
            else:
                alpha = alphas_dir[irow,idx]
                val = -U[irow] / alpha + V
                if alpha > alphas_max_dir[idx] and (val < upper_bound):
                    upper_bound = val
                elif alpha < -alphas_max_dir[idx] and (val > lower_bound):
                    lower_bound = val
        if lower_bound > V:
            lower_bound = V - tol 
        elif upper_bound < V:
            upper_bound = V + tol 

        discriminant = sqrt(V*V-(norm_state_sq-norm_state_bound))
        if np.isnan(discriminant):
            upper_bound = V
            lower_bound = V
        else:
            if upper_bound > discriminant:
                upper_bound = discriminant
            if lower_bound < - discriminant:
                lower_bound = - discriminant

        tval = lower_bound + usample[iter_count] * (upper_bound - lower_bound)
            
        if docoord == 1:
            state[idx] = tval
            tval = tval - V
            for irow in range(nconstraint):
                U[irow] = U[irow] + tval * A[irow, idx]
        else:
            tval = tval - V
            for ivar in range(nvar):
                state[ivar] = state[ivar] + tval * directions[idx,ivar]
                for irow in range(nconstraint):
                    U[irow] = (U[irow] + A[irow, ivar] * 
                               tval * directions[idx,ivar])

        if iter_count >= burnin:
            for ivar in range(nvar):
                trunc_sample[iter_count - burnin, ivar] = state[ivar]
        
        norm_state_sq = 0
        for ivar in range(nvar):
            norm_state_sq = norm_state_sq + state[ivar]*state[ivar]
        if norm_state_sq > norm_state_bound:
            multiplier = np.sqrt(0.999 * norm_state_bound / norm_state_sq)
            for ivar in range(nvar):
                state[ivar] = state[ivar] * multiplier
            norm_state_sq = 0.999 * norm_state_bound

        norm_state_bound = sample_squared_radius(state)

    return trunc_sample

@cython.boundscheck(False)
@cython.cdivision(True)
def sample_truncnorm_white_sphere(np.ndarray[DTYPE_float_t, ndim=2] A, 
                                  np.ndarray[DTYPE_float_t, ndim=1] b, 
                                  np.ndarray[DTYPE_float_t, ndim=1] initial, 
                                  np.ndarray[DTYPE_float_t, ndim=1] bias_direction, 
                                  DTYPE_int_t how_often=1000,
                                  DTYPE_int_t burnin=500,
                                  DTYPE_int_t ndraw=1000,
                                  ):
    """
    Sample from a ball of radius `np.linalg.norm(initial)`
    intersected with a constraint.

    Constraint is $Ax \leq b$ where `A` has shape
    `(q,n)` with `q` the number of constraints and
    `n` the number of random variables.

    Parameters
    ----------

    A : np.float((q,n))
        Linear part of affine constraints.

    b : np.float(q)
        Offset part of affine constraints.

    initial : np.float(n)
        Initial point for Gibbs draws.
        Assumed to satisfy the constraints.

    bias_direction : np.float (optional)
        Which projection is of most interest?

    how_often : int (optional)
        How often should the sampler make a move along `direction_of_interest`?
        If negative, defaults to ndraw+burnin (so it will never be used).

    sigma : float
        Variance parameter.

    burnin : int
        How many iterations until we start
        recording samples?

    ndraw : int
        How many samples should we return?

    Returns
    -------

    trunc_sample : np.float((ndraw, n))

    """

    cdef int nvar = A.shape[1]
    cdef int nconstraint = A.shape[0]
    cdef np.ndarray[DTYPE_float_t, ndim=2] trunc_sample = \
            np.empty((ndraw, nvar), np.float)
    cdef np.ndarray[DTYPE_float_t, ndim=1] weight_sample = \
            np.empty((ndraw,), np.float)
    cdef np.ndarray[DTYPE_float_t, ndim=1] state = initial.copy()
    cdef int idx, iter_count, irow, ivar
    cdef double lower_bound, upper_bound, V
    cdef double cdfL, cdfU, unif, tval, val, alpha
    cdef double norm_state_bound = np.linalg.norm(state)**2
    cdef double norm_state_sq = norm_state_bound

    cdef double tol = 1.e-7

    cdef np.ndarray[DTYPE_float_t, ndim=1] Astate = np.dot(A, state) 

    cdef np.ndarray[DTYPE_float_t, ndim=1] usample = \
        np.random.sample(burnin + ndraw)

    # directions not parallel to coordinate axes

    cdef np.ndarray[DTYPE_float_t, ndim=2] directions = \
        np.vstack([A, 
                   np.random.standard_normal((int(nvar/5),nvar))])
    directions[-1][:] = bias_direction

    directions /= np.sqrt((directions**2).sum(1))[:,None]

    cdef int ndir = directions.shape[0]

    cdef np.ndarray[DTYPE_float_t, ndim=2] alphas_dir = \
        np.dot(A, directions.T)

    cdef np.ndarray[DTYPE_float_t, ndim=2] alphas_coord = A
        
    cdef np.ndarray[DTYPE_float_t, ndim=1] alphas_max_dir = \
        np.fabs(alphas_dir).max(0) * tol    

    cdef np.ndarray[DTYPE_float_t, ndim=1] alphas_max_coord = \
        np.fabs(alphas_coord).max(0) * tol 

    # choose the order of sampling (randomly)

    cdef np.ndarray[DTYPE_int_t, ndim=1] random_idx_dir = \
        np.random.random_integers(0, ndir-1, size=(burnin+ndraw,))

    cdef np.ndarray[DTYPE_int_t, ndim=1] random_idx_coord = \
        np.random.random_integers(0, nvar-1, size=(burnin+ndraw,))

    # for switching between coordinate updates and
    # other directions

    cdef int invperiod = 13
    cdef int docoord = 0
    cdef int iperiod = 0
    cdef int ibias = 0
    cdef int dobias = 0
    cdef double discriminant, multiplier
    cdef int sample_count = 0
    cdef int in_event = 0
    cdef int numout = 0
    cdef double min_multiple = 0.

    iter_count = 0

    while True:

        # sample from the ball

        docoord = 1
        iperiod = iperiod + 1
        ibias = ibias + 1

        if iperiod == invperiod: 
            docoord = 0
            iperiod = 0
            dobias = 0

        if ibias == how_often:
            docoord = 0
            ibias = 0
            dobias = 1
        
        if docoord == 1:
            idx = random_idx_coord[iter_count  % (ndraw + burnin)]
            V = state[idx]
        else:
            if not dobias:
                idx = random_idx_dir[iter_count  % (ndraw + burnin)]
            else:
                idx = directions.shape[0]-1 # last row of directions is bias_direction
            V = 0
            for ivar in range(nvar):
                V = V + directions[idx, ivar] * state[ivar]

        lower_bound = -1e12
        upper_bound = 1e12
        for irow in range(nconstraint):
            if docoord == 1:
                alpha = alphas_coord[irow,idx]
                val = (-Astate[irow] + b[irow]) / alpha + V
                if alpha > alphas_max_coord[idx] and (val < upper_bound):
                    upper_bound = val
                elif alpha < -alphas_max_coord[idx] and (val > lower_bound):
                    lower_bound = val
            else:
                alpha = alphas_dir[irow,idx]
                val = (-Astate[irow] + b[irow]) / alpha + V
                if alpha > alphas_max_dir[idx] and (val < upper_bound):
                    upper_bound = val
                elif alpha < -alphas_max_dir[idx] and (val > lower_bound):
                    lower_bound = val

        if lower_bound > V:
            lower_bound = V - tol 
        elif upper_bound < V:
            upper_bound = V + tol 

        discriminant = sqrt(V*V-(norm_state_sq-norm_state_bound))
        if np.isnan(discriminant):
            upper_bound = V
            lower_bound = V
        else:
            if upper_bound > discriminant:
                upper_bound = discriminant
            if lower_bound < - discriminant:
                lower_bound = - discriminant

        tval = lower_bound + usample[iter_count % (ndraw + burnin)] * (upper_bound - lower_bound)
            
        if docoord == 1:
            state[idx] = tval
            tval = tval - V
            for irow in range(nconstraint):
                Astate[irow] = Astate[irow] + tval * A[irow, idx]
        else:
            tval = tval - V
            for ivar in range(nvar):
                state[ivar] = state[ivar] + tval * directions[idx,ivar]
                for irow in range(nconstraint):
                    Astate[irow] = (Astate[irow] + A[irow, ivar] * 
                                    tval * directions[idx,ivar])

        norm_state_sq = 0
        for ivar in range(nvar):
            norm_state_sq = norm_state_sq + state[ivar]*state[ivar]
        if norm_state_sq > norm_state_bound:
            multiplier = np.sqrt(0.999 * norm_state_bound / norm_state_sq)
            for ivar in range(nvar):
                state[ivar] = state[ivar] * multiplier
            norm_state_sq = 0.999 * norm_state_bound

        # check constraints

        in_event = 1
        multiplier = sqrt(norm_state_bound / norm_state_sq)
        for irow in range(nconstraint):
            if Astate[irow] * multiplier > b[irow]:
                in_event = 0

        if in_event == 1:
            # store the sample

            if sample_count >= burnin:
                for ivar in range(nvar):
                    trunc_sample[sample_count-burnin, ivar] = state[ivar] * multiplier

                # now compute the smallest multiple M of state that is in the event
            
                min_multiple = 0
                for irow in range(nconstraint):
                    if Astate[irow] < 0:
                        val = b[irow] / Astate[irow] 
                        if min_multiple <  val:
                            min_multiple = val

                # the weight for this sample is 1/(1-M^n)

                weight_sample[sample_count-burnin] = 1. / (1 - pow(min_multiple, nvar))

            sample_count = sample_count + 1
        else:
            numout = numout + 1

        iter_count = iter_count + 1

        if sample_count >= ndraw + burnin:
            break

    return trunc_sample, weight_sample

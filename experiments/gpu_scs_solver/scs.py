import numpy as np
import scipy.linalg
import scipy.optimize
import cupy as cp
import cupyx.scipy.sparse, cupyx.scipy.sparse.linalg
import time

def linprog10(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, tolcheck=10, eps_pri=1e-6, eps_dual=1e-4, eps_gap=1e-4, eps_ubdd=1e-4, eps_infeas=1e-4, max_itr=100000, dtype=cp.float32):
    #move data over to GPU. This really isn't fair to this method
    start = time.time()
    b = np.concatenate((b_eq, b_ub))
    b_norm = np.linalg.norm(b)
    c_norm = np.linalg.norm(c)
    h = np.concatenate((c, b))
    h = cp.asarray(h, dtype=dtype)
    c = h[:c.shape[0]]
    b = h[c.shape[0]:]
    if type(A_ub) == np.ndarray and type(A_eq) == np.ndarray:
        A_ub, A_eq = cp.asarray(A_ub, dtype=dtype), cp.asarray(A_eq, dtype=dtype)
    elif scipy.sparse.issparse(A_ub) and scipy.sparse.issparse(A_eq):
        A_ub, A_eq = cupyx.scipy.sparse.csr_matrix(A_ub, dtype=dtype), cupyx.scipy.sparse.csr_matrix(A_eq, dtype=dtype)
    else:
        assert False, "Matrix format not recognized"
    print(f"Took {time.time() - start} seconds to move problem data to GPU")
    apply_A_out = cp.empty(A_eq.shape[0] + A_ub.shape[0], dtype=dtype) #prevents an allocation when matmuling
    def apply_A(x, out=apply_A_out):
        out[:A_eq.shape[0]] = A_eq.dot(x)
        out[A_eq.shape[0]:] = A_ub.dot(x)
        return out
    apply_At_out = cp.empty(c.shape[0], dtype=dtype) #prevent an allocation when matmuling
    def apply_At(y, out=apply_At_out):
        out[:] = A_eq.T.dot(y[:A_eq.shape[0]])
        out[:] += A_ub.T.dot(y[A_eq.shape[0]:])
        return out
    def apply_IpAtA(x, out=apply_At_out):
        apply_At(apply_A(x)) #populate apply_At_out
        out[:] += x #add identity
        return out
    IpAtA = cupyx.scipy.sparse.linalg.LinearOperator((A_ub.shape[1], A_ub.shape[1]), matvec=apply_IpAtA)
    Minvh = cp.empty(h.shape[0], dtype=dtype)
    Minvh[:c.shape[0]], _ = cupyx.scipy.sparse.linalg.cg(IpAtA, h[:c.shape[0]] - apply_At(h[c.shape[0]:]))
    Minvh[c.shape[0]:] = h[c.shape[0]:] + apply_A(Minvh[:c.shape[0]])
    u = cp.zeros(h.shape[0]+1, dtype=dtype)
    #u[-1] = 1.0
    v = cp.zeros(h.shape[0]+1, dtype=dtype)
    v[-1] = 1.0
    utilde = cp.empty(h.shape[0]+1, dtype=dtype)
    w = cp.empty(h.shape[0]+1, dtype=dtype) #prevent an allocation in the loop
    rhs = cp.empty(h.shape[0], dtype=dtype) #prevent an allocation in the loop
    vstep = cp.empty(h.shape[0]+1, dtype=dtype)
    itr=0
    while itr < max_itr:
        itr += 1
        cp.add(u, v, out=w)
        cp.add(w[:-1], cp.multiply(-1.0*w[-1], h, out=rhs), out=rhs)
        utilde[:c.shape[0]], _ = cupyx.scipy.sparse.linalg.cg(IpAtA, rhs[:c.shape[0]] - apply_At(rhs[c.shape[0]:]))
        utilde[c.shape[0]:-1] = rhs[c.shape[0]:] + apply_A(utilde[:c.shape[0]])
        utilde[:-1] -= (h.T@utilde[:-1])/(1 + h.T @ Minvh)*Minvh
        ###
        utilde[-1] = w[-1] + h.T @ utilde[:-1] #this is the equation preceeding (28). I expressed it w/ h instead
        u[:(A_eq.shape[1]+A_eq.shape[0])] = utilde[:(A_eq.shape[1]+A_eq.shape[0])] \
                                            - v[:(A_eq.shape[1]+A_eq.shape[0])]
        u[(A_eq.shape[1]+A_eq.shape[0]):] = cp.maximum(0,\
                                             utilde[(A_eq.shape[1]+A_eq.shape[0]):] \
                                                - v[(A_eq.shape[1]+A_eq.shape[0]):])
        vstep[:] = u - utilde
        v += vstep
        #stopping condition check every tolcheck
        if itr%tolcheck==0:
            if u[-1] > 0: #we're optimal
                x = u[:c.shape[0]]/u[-1]
                s = v[c.shape[0]:c.shape[0]+b.shape[0]]/u[-1]
                y = u[c.shape[0]:(c.shape[0]+b.shape[0])]/u[-1]
                p_feas = cp.linalg.norm(apply_A(x) + s - b)/(1+b_norm)
                d_feas = cp.linalg.norm(apply_At(y) + c)/(1+c_norm)
                dual_gap = cp.abs(c.T@x + b.T@y)/(1 + cp.abs(c.T@x) + cp.abs(b.T@y))
                print("| itr | primal_feas |  dual_feas  | primal/dual gap | ")
                print(itr, p_feas, d_feas, dual_gap)
                if p_feas < eps_pri and d_feas < eps_dual and dual_gap < eps_gap:
                    print("We're optimal. Terminating...")
                    break
            unbdd_chk = cp.linalg.norm(apply_A(u[:c.shape[0]])\
                + v[c.shape[0]:c.shape[0]+b.shape[0]]) \
                    <= (-c.T@u[:c.shape[0]]/c_norm)*eps_ubdd
            if unbdd_chk:
                print("Problem is unbounded. Terminating...")
                break
            infeas_check = cp.linalg.norm(apply_At(u[c.shape[0]:c.shape[0]+b.shape[0]]))\
                <= (-b.T@u[c.shape[0]:c.shape[0]+b.shape[0]]/b_norm)*eps_infeas
            if infeas_check:
                print("Problem is infeasible. Terminating...")
                break
    if itr == max_itr:
        print("Iteration limit hit")
    return u[:A_eq.shape[1]]/u[-1] #the x component of u. See (8)

import numpy as np
import torch
from pyFM.mesh import TriMesh
from pyFM.functional import FunctionalMapping
from scipy.optimize import fmin_l_bfgs_b, linear_sum_assignment
import time
import os

def compute_surface_map(mesh1_t, mesh2_t, c1, c2, n_ev=50, compute_extra=False, optimizer="fmin_l_bfgs_b", descr_type="neural", maxiter=100000, optimize_p2p=False, fit_params=None):
    '''
    Returns:
        mapping_2to1: torch.Tensor, shape (N2,) closest vertex on mesh 1 for each vertex on mesh 2
        mapping_1to2: torch.Tensor, shape (N1,) closest vertex on mesh 2 for each vertex on mesh 1
    '''
    # make sure the TriMesh is initialized correctly. Loading with o3d and pytorch3d produces different results.
    assert descr_type in ["neural", "HKS", "WKS"]
    mesh1 = TriMesh(mesh1_t.verts_list()[0].cpu(), mesh1_t.faces_list()[0].cpu())
    mesh2 = TriMesh(mesh2_t.verts_list()[0].cpu(), mesh2_t.faces_list()[0].cpu())
    if descr_type == "HKS":
        process_params = {
        'n_ev': (n_ev,n_ev),  # Number of eigenvalues on source and Target
        'landmarks': None,
        'n_descr': 16, 
        'descr_type': 'HKS',
        'subsample_step': 1,
        }
    elif descr_type == "WKS":
        process_params = {
        'n_ev': (n_ev,n_ev),  # Number of eigenvalues on source and Target
        'n_descr': 2048,
        'landmarks': None,
        'subsample_step': 1,  # In order not to use too many descriptors
        'descr_type': 'WKS',  # WKS or HKS
        }
    elif descr_type == "neural":
        process_params = {
        'n_ev': (n_ev,n_ev),  # Number of eigenvalues on source and Target
        'n_descr': c1.shape[1],
        'landmarks': None,
        'descr1': c1,
        'descr2': c2,
        'subsample_step': 1,
        }
    model = FunctionalMapping(mesh1, mesh2, partial=False, optimizer=optimizer)
    model.preprocess(**process_params,verbose=True)
    fit_params = fit_params
    model.fit(**fit_params, verbose=True)
    p2p_21_adjoint, p2p_12_adjoint = model.get_p2p(n_jobs=1) # sets model.mapped_indicator
    p2p_21 = (model.mapped_indicator * model.eta[..., None]).argmax(axis=1) # override the above
    p2p_12 = (model.mapped_indicator * model.eta[..., None]).argmax(axis=0)
    
    timing = os.environ.get("TIMEIT", False)
    if timing:
        compute_extra = True

    start_s = time.time()
    hungarian = linear_sum_assignment(model.mapped_indicator * model.eta[..., None] - 1000 * (1 - model.eta[..., None]), maximize=True) if compute_extra else None 
    if timing: print("Hungarian for vanilla took", time.time() - start_s, "seconds")

    start_s = time.time()
    # get precise stuff
    mapped_indicator_precise = model.get_precise_map().toarray() if compute_extra else None
    if timing: print("getting precise map took", time.time() - start_s, "seconds")
    
    start_s = time.time()
    hungarian_precise = linear_sum_assignment(mapped_indicator_precise * model.eta[..., None] - 1000 * (1 - model.eta[..., None]), maximize=True) if compute_extra else None
    if timing: print("Hungarian for precise took", time.time() - start_s, "seconds")
    
    start_s = time.time()            
    # get icp stuff
    model.icp_refine()
    if timing: print("ICP refinement took", time.time() - start_s, "seconds")
    
    start_s = time.time()
    p2p_21_icp_adjoint, p2p_12_icp_adjoint = model.get_p2p(n_jobs=1)
    p2p_21_icp = (model.mapped_indicator * model.eta[..., None]).argmax(axis=1) # override the above
    p2p_12_icp = (model.mapped_indicator * model.eta[..., None]).argmax(axis=0)
    hungarian_icp = linear_sum_assignment(model.mapped_indicator * model.eta[..., None] - 1000 * (1 - model.eta[..., None]), maximize=True)
    if timing: print("Hungarian for icp took", time.time() - start_s, "seconds")
    
    return p2p_21, p2p_12, hungarian, hungarian_precise, p2p_21_icp, p2p_12_icp, hungarian_icp, model, model.mesh1, model.mesh2, p2p_21_adjoint, p2p_12_adjoint, p2p_21_icp_adjoint, p2p_12_icp_adjoint

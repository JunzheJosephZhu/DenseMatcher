import numpy as np
import torch
import copy
import os

can_op1 = None
can_op2 = None
descr1_red = None
descr2_red = None
commute_left = None
commute_right = None

def reset_globals():
    global can_op1
    can_op1 = None
    global can_op2
    can_op2 = None
    global descr1_red
    descr1_red = None
    global descr2_red
    descr2_red = None
    global commute_left
    commute_left = None
    global commute_right
    commute_right = None

def verbose_print(*args):
    if os.environ.get("VERBOSE", False):
        print(*args)

def descr_preservation(C, v, descr1_red, descr2_red, ctx):
    """
    Compute the descriptor preservation constraint

    Parameters
    ---------------------
    C      :
        (K2,K1) Functional map
    descr1 :
        (K1,p) descriptors on first basis
    descr2 :
        (K2,p) descriptros on second basis

    Returns
    ---------------------
    energy : float
        descriptor preservation squared norm
    """
    loss = 0.5 * torch.square(C @ descr1_red - descr2_red).sum()
    loss.backward()
    ctx["descr_preservation_grad"] = C.grad.clone()
    C.grad.zero_()
    if v is not None:
        ctx["descr_preservation_grad_v"] = v.grad.clone()
        v.grad.zero_()
    return loss

def descr_preservation_grad(C, descr1_red, descr2_red):
    """
    Compute the gradient of the descriptor preservation constraint

    Parameters
    ---------------------
    C      :
        (K2,K1) Functional map
    descr1 :
        (K1,p) descriptors on first basis
    descr2 :
        (K2,p) descriptros on second basis

    Returns
    ---------------------
    gradient : np.ndarray
        gradient of the descriptor preservation squared norm
    """
    return (C @ descr1_red - descr2_red) @ descr1_red.T


def LB_commutation(C, v, ev_sqdiff, ctx):
    """
    Compute the LB commutativity constraint

    Parameters
    ---------------------
    C      :
        (K2,K1) Functional map
    ev_sqdiff :
        (K2,K1) [normalized] matrix of squared eigenvalue differences

    Returns
    ---------------------
    energy : float
        (float) LB commutativity squared norm
    """
    loss = 0.5 * (torch.square(C) * ev_sqdiff).sum()
    loss.backward()
    ctx["LB_commutation_grad"] = C.grad.clone()
    C.grad.zero_()
    if v is not None:
        ctx["LB_commutation_grad_v"] = v.grad.clone()
        v.grad.zero_()
    return loss


def LB_commutation_grad(C, ev_sqdiff):
    """
    Compute the gradient of the LB commutativity constraint

    Parameters
    ---------------------
    C         :
        (K2,K1) Functional map
    ev_sqdiff :
        (K2,K1) [normalized] matrix of squared eigenvalue differences

    Returns
    ---------------------
    gradient : np.ndarray
        (K2,K1) gradient of the LB commutativity squared norm
    """
    return C * ev_sqdiff


def op_commutation(C, op1, op2):
    """
    Compute the operator commutativity constraint.
    Can be used with descriptor multiplication operator

    Parameters
    ---------------------
    C   :
        (K2,K1) Functional map
    op1 :
        (K1,K1) operator on first basis
    op2 :
        (K2,K2) descriptros on second basis

    Returns
    ---------------------
    energy : float
        (float) operator commutativity squared norm
    """
    return 0.5 * torch.square(C @ op1 - op2 @ C).sum()


def op_commutation_grad(C, op1, op2):
    """
    Compute the gradient of the operator commutativity constraint.
    Can be used with descriptor multiplication operator

    Parameters
    ---------------------
    C   :
        (K2,K1) Functional map
    op1 :
        (K1,K1) operator on first basis
    op2 :
        (K2,K2) descriptros on second basis

    Returns
    ---------------------
    gardient : np.ndarray
        (K2,K1) gradient of the operator commutativity squared norm
    """
    return op2.T @ (op2 @ C - C @ op1) - (op2 @ C - C @ op1) @ op1.T


def oplist_commutation(C, v, op_list, compute_backward=True):
    """
    Compute the operator commutativity constraint for a list of pairs of operators
    Can be used with a list of descriptor multiplication operator

    Parameters
    ---------------------
    C   :
        (K2,K1) Functional map
    op_list :
        list of tuple( (K1,K1), (K2,K2) ) operators on first and second basis

    Returns
    ---------------------
    energy : float
        (float) sum of operators commutativity squared norm
    """
    energy = 0
    for (op1, op2) in op_list:
        energy += op_commutation(C, op1, op2)
        
    # if just evaluating w_orient before the optimization loop, don't do backward
    if not compute_backward:
        return energy
    
    energy.backward()
    grad_C = C.grad.clone()
    C.grad.zero_()
    if v is not None:
        grad_v = v.grad.clone()
        v.grad.zero_()
    else:
        grad_v = None
    return energy, grad_C, grad_v


def oplist_commutation_grad(C, op_list):
    """
    Compute the gradient of the operator commutativity constraint for a list of pairs of operators
    Can be used with a list of descriptor multiplication operator

    Parameters
    ---------------------
    C   :
        (K2,K1) Functional map
    op_list :
        list of tuple( (K1,K1), (K2,K2) ) operators on first and second basis

    Returns
    ---------------------
    gradient : np.ndarray
        (K2,K1) gradient of the sum of operators commutativity squared norm
    """
    gradient = 0
    for (op1, op2) in op_list:
        gradient += op_commutation_grad(C, op1, op2)
    return gradient

#################################################################################################################
#add
def area(C):
    """
    Return the area shape difference computed from a functional map.

    Parameters
    ---------------------------
    C : (k2,k1) functional map between two meshes

    Output
    ----------------------------
    SD : (k1,k1) - Area based shape difference operator
    """
    k2, k1 = C.shape
    return 0.5 * torch.square(C.T @ C - torch.eye(k1, device=C.device)).sum()

def area_grad(C):
    """
    Return the area shape difference computed from a functional map.

    Parameters
    ---------------------------
    C : (k2,k1) functional map between two meshes

    Output
    ----------------------------
    SD : (k1,k1) - Area based shape difference operator
    """
    return 2 * C @ C.T @ C - 2 * C

def conformal(C, evals1, evals2):
    """
    Return the conformal shape difference operator computed from a functional map.

    Parameters
    ---------------------------
    C     : (k2,k1) functional map between two meshes
    evals1 : eigenvalues of the LBO on the source mesh (at least k1)
    evals2 : eigenvalues of the LBO on the target mesh (at least k2)

    Output
    ----------------------------
    SD : (k1,k1) - Conformal shape difference operator
    """
    k2,k1 = C.shape

    # SD = np.linalg.pinv(np.diag(evals1[:k1])) @ FM.T @ (evals2[:k2,None] * FM)
    scale = max(evals1.max(), evals2.max())
    return 0.5 * torch.square(C.T @ torch.diag(evals2 / scale) @ C - torch.diag(evals1 / scale)).sum()


def conformal_grad(C, evals1, evals2):
    """
    Return the conformal shape difference operator computed from a functional map.

    Parameters
    ---------------------------
    C     : (k2,k1) functional map between two meshes
    evals1 : eigenvalues of the LBO on the source mesh (at least k1)
    evals2 : eigenvalues of the LBO on the target mesh (at least k2)

    Output
    ----------------------------
    SD : (k1,k1) - Conformal shape difference operator
    """
    scale = max(evals1.max(), evals2.max())
    grad = 2 * torch.diag(evals2 / scale) @ C @ C.T @ torch.diag(evals2 / scale) @ C - 2 * torch.diag(evals2 / scale) @ C @ torch.diag(evals1 / scale)
    return grad

def p2p(C, v, evects1, evects2, A1, ctx):
    """
    Return the point 2 point loss:
    See section 5 of http://www.lix.polytechnique.fr/~maks/papers/fundescEG17.pdf
    
    Parameters
    ---------------------------
    C       : (k2,k1) functional map between two meshes
    evects1 : (n1,k1) eigenvectors of the LBO on the source mesh
    evects2 : (n2,k2) eigenvectors of the LBO on the target mesh
    ctx: dictionary
    
    Output
    ----------------------------
    loss    : If we map an identity matrix(indicator func for each vertex)
                from mesh1 to mesh2, the output should ideally be {0, 1}
                So its square should be equal the itself
    """
    mapped_id = evects2 @ C @ evects1.T @ A1 # [n2, n1]
    loss = torch.sum((mapped_id ** 2 - mapped_id) ** 2)
    loss.backward()
    ctx["p2p_grad"] = C.grad.clone()
    C.grad.zero_()
    if v is not None:
        ctx["p2p_grad_v"] = v.grad.clone()
        v.grad.zero_()
    return loss

def doubly_stochastic(C, v, evects1, evects2, A1, ctx):
    """
    Make sure the matrix maps all ones to all ones
    The p2p constraint indicates that indicator functions can be mapped to [0, 1]
    Its possible that everything winds up being 0, we dont want that
    Specifically we want to make sure the squared version still sums to 1
    
    Parameters
    ---------------------------
    C       : (k2,k1) functional map between two meshes
    evects1 : (n1,k1) eigenvectors of the LBO on the source mesh
    evects2 : (n2,k2) eigenvectors of the LBO on the target mesh
    ctx: dictionary
    
    Output
    ----------------------------
    loss    : If we map an all 1 vector(indicator func for each vertex)
                from mesh1 to mesh2, the output should be all 1
    """
    n1, k1 = evects1.shape
    n2, k2 = evects2.shape

    # mapped_ones = evects2 @ (C @ evects1.T.sum(dim=0, keepdim=True))  # [n2, 1]
    # gt_ones = torch.ones_like(mapped_ones)
    # loss = torch.sum((mapped_ones - gt_ones) ** 2)
    # we desire each output logit to be 1. So total output will be n2 while total input will be n1
    mapped_id = evects2 @ C @ evects1.T @ A1 # [n2, n1]
    mapped_id_sq = mapped_id ** 2
    loss_col = torch.sum((mapped_id_sq.sum(dim=0) - n2 / n1) ** 2) # [n1]
    loss_row = torch.sum((mapped_id_sq.sum(dim=1) - 1) ** 2) # [n2]
    loss = loss_col + loss_row
    loss.backward()
    ctx["doubly_stochastic_grad"] = C.grad.clone()
    C.grad.zero_()
    if v is not None:
        ctx["doubly_stochastic_grad_v"] = v.grad.clone()
        v.grad.zero_()
    return loss
    
def entropy(C, v, evects1, evects2, A1, ctx):
    mapped_id = evects2 @ C @ evects1.T @ A1 # [n2, n1]
    loss = torch.sum(-mapped_id.clamp(0, 1) * torch.log(mapped_id.clamp(0, 1) + 1e-10))
    loss.backward()
    ctx["entropy_grad"] = C.grad.clone()
    C.grad.zero_()
    if v is not None:
        ctx["entropy_grad_v"] = v.grad.clone()
        v.grad.zero_()
    return loss

def range01(C, v, evects1, evects2, A1, ctx):
    mapped_id = evects2 @ C @ evects1.T @ A1 # [n2, n1]
    loss_sub0 = torch.square((-mapped_id).clamp(0, max=None)).sum()
    loss_super1 = torch.square((mapped_id - 1).clamp(0, max=None)).sum()
    loss = loss_sub0 + loss_super1
    loss.backward()
    ctx["range01_grad"] = C.grad.clone()
    C.grad.zero_()
    if v is not None:
        ctx["range01_grad_v"] = v.grad.clone()
        v.grad.zero_()
    return loss

def sumto1(C, v, evects1, evects2, A1, ctx):
    """
    Make sure the matrix maps all ones to all ones
    The p2p constraint indicates that indicator functions can be mapped to [0, 1]
    Its possible that everything winds up being 0, we dont want that
    Specifically we want to make sure the squared version still sums to 1
    
    Parameters
    ---------------------------
    C       : (k2,k1) functional map between two meshes
    evects1 : (n1,k1) eigenvectors of the LBO on the source mesh
    evects2 : (n2,k2) eigenvectors of the LBO on the target mesh
    ctx: dictionary
    
    Output
    ----------------------------
    loss    : If we map an all 1 vector(indicator func for each vertex)
                from mesh1 to mesh2, the output should be all 1
    """
    n1, k1 = evects1.shape
    n2, k2 = evects2.shape

    # mapped_ones = evects2 @ (C @ evects1.T.sum(dim=0, keepdim=True))  # [n2, 1]
    # gt_ones = torch.ones_like(mapped_ones)
    # loss = torch.sum((mapped_ones - gt_ones) ** 2)
    mapped_id = evects2 @ C @ evects1.T @ A1 # [n2, n1]
    if v is not None:
        # we desire each output logit corresponding to an input to be 1
        eta = torch.sigmoid(v)
        loss_col = torch.sum((mapped_id.sum(0) - 1) ** 2) # [n1]. Each input should sum to 1
        loss_row = torch.sum((mapped_id.sum(1) - eta) ** 2) # [n2]. Each output should sum to eta
    else:
        loss_col = torch.sum((mapped_id.sum(dim=0) - mapped_id.sum(dim=0).mean()) ** 2)  # [n1] # -n2 / n1
        loss_row = torch.sum((mapped_id.sum(dim=1) - mapped_id.sum(dim=1).mean()) ** 2) # [n2]
    loss = loss_col + loss_row
    loss.backward()
    ctx["sumto1_grad"] = C.grad.clone()
    C.grad.zero_()
    if v is not None:
        ctx["sumto1_grad_v"] = v.grad.clone()
        v.grad.zero_()
    return loss

def orientation_op_torch(grad_field, vertices, faces, normals, per_vert_area):
    '''
    copied from pyFM.mesh.geometry.get_orientation_op
    modified for parallelization
    grad_field: (f, 3, ndescr)
    normals: (f, 3)
    return (f, f, n_descr)
    '''
    n_faces = faces.shape[0]
    n_verts = vertices.shape[0]
    n_descr = grad_field.shape[2]
    v1 = vertices[faces[:,0]].float()  # (n_f,3)
    v2 = vertices[faces[:,1]].float()  # (n_f,3)
    v3 = vertices[faces[:,2]].float()  # (n_f,3)
    
    # gradient in barycentric coordinate axes(how fast the cartesian coordinates change wrt barycentric coordinate)
    # So if we call barycentric coordinate as b, cartesian coordinate as x, then Jc  = db/dx, assuming we have some function of F(b), so dF/dx = dF/db * db/dx.
    Jc1 = torch.cross(normals.float(), v3-v2)/2
    Jc2 = torch.cross(normals.float(), v1-v3)/2
    Jc3 = torch.cross(normals.float(), v2-v1)/2
    
    rot_field = torch.cross(normals[..., None].float(), grad_field.float(), dim=1)  # (f, 3, ndescr)

    I = torch.cat([faces[:,0], faces[:,1], faces[:,2]], dim=0) # [3 * f]
    J = torch.cat([faces[:,1], faces[:,2], faces[:,0]], dim=0)
    
    # projection onto three axes
    Sij = 1/3*torch.cat([(Jc2[..., None] * rot_field).sum(dim=1), 
                         (Jc3[..., None] * rot_field).sum(dim=1),
                         (Jc1[..., None] * rot_field).sum(dim=1)], dim=0) # (3 * f, ndescr)
    Sji = 1/3*torch.cat([(Jc1[..., None] * rot_field).sum(dim=1),
                         (Jc2[..., None] * rot_field).sum(dim=1),
                         (Jc3[..., None] * rot_field).sum(dim=1)], dim=0) # (3 * f, ndescr)
    # # This is for computing dot(rot_field, db/dx), so its the multiplication operator with dF/db on vertices, which would be Sij-Sii or Sji-sjj  
    In = torch.cat([I,J,I,J])
    Jn = torch.cat([J,I,I,J])
    # Kn = torch.arange(n_descr, device=I.device).repeat(4 * 3 * n_faces)
    Sn = torch.cat([Sij,Sji,-Sij,-Sji])
    
    # hybrid COO tensor with last dimesnion occupied
    W = torch.sparse_coo_tensor(torch.stack([In,Jn]), Sn, (n_verts,n_verts, n_descr), device=vertices.device)
    # W = torch.zeros((n_verts, n_verts, n_descr), device=vertices.device, dtype=Sn.dtype)
    # W[In, Jn] = Sn

    # inv_area = torch.sparse.spdiags(1/per_vert_area.cpu(), offsets=torch.tensor(0, dtype=int), shape=(n_verts,n_verts)).to(vertices)    
    return (1 / per_vert_area)[:, None, None] * W.to(vertices.dtype)



#################################################################################################################
def energy_func_std(x, partial, descr_mu, lap_mu, descr_comm_mu, orient_mu, area_mu, conformal_mu, p2p_mu, stochastic_mu, ent_mu, range01_mu, sumto1_mu, area_difference_mu, mumford_shah_mu, mumford_shah_var, eta_entropy_mu, descr1, descr2, gradmat1, gradmat2, orient_op, ev_sqdiff, evals1, evals2, evects1, evects2, A1, A2, pinv1, pinv2, verts1, verts2, faces1, faces2, normals1, normals2, ctx):
    """
    Evaluation of the energy for standard FM computation

    Parameters:
    ----------------------
    C               : (K2*K1) or (K2,K1) Functional map
    descr_mu        : scaling of the descriptor preservation term
    lap_mu          : scaling of the laplacian commutativity term
    descr_comm_mu   : scaling of the descriptor commutativity term
    orient_mu       : scaling of the orientation preservation term
    descr1          : (n1,n_descriptors) descriptors on first basis
    descr2          : (n2,n_descriptors) descriptros on second basis
    list_descr      : p-uple( (K1,K1), (K2,K2) ) operators on first and second basis
                      related to descriptors.
    orient_op       : p-uple( (K1,K1), (K2,K2) ) operators on first and second basis
                      related to orientation preservation operators.
    ev_sqdiff       : (K2,K1) [normalized] matrix of squared eigenvalue differences
    evects1         : (n1, k1) eigenvectors of the LBO on the source mesh
    evects2         : (n2, k2) eigenvectors of the LBO on the source mesh

    ctx is for passing shit from here to grad_func_std, while global variables are for shit that only need to be computed once per pair of mesh
    Output
    ------------------------
    energy : float - value of the energy
    """
    # print("forward")
    keys = list(ctx.keys())
    for key in keys:
        ctx.pop(key)
        verbose_print(f"Warning: popped {key} from ctx. This shouldnt happen under LBFGS but is normal under SLSQP and at end of CG")
    n1, k1 = evects1.shape
    n2, k2 = evects2.shape
    _, ndescr = descr1.shape
    f1, f2 = faces1.shape[0], faces2.shape[0]
    C = x[:k1 * k2].reshape((k2,k1))
    C = torch.tensor(C).to(evects1)
    C.requires_grad_(True)
    assert C.is_leaf
    if partial:
        raise NotImplementedError()
    else:
        v = None
        eta = torch.ones((n2, )).to(evects1)
        
    # project descriptors to spectral domain
    global descr1_red
    if descr1_red is None:
        descr1_red = evects1.T @ A1 @ descr1
    global descr2_red
    if descr2_red is None:
        descr2_red = evects2.T @ A2 @ (eta[..., None] * descr2)
    ctx["descr2_red"] = descr2_red
    
    energy = 0

    if descr_mu > 0:
        descr_loss = descr_mu * descr_preservation(C, v, descr1_red, descr2_red, ctx)
        verbose_print("descr loss:", descr_loss.item())
        energy += descr_loss

    if lap_mu > 0:
        lap_loss = lap_mu * LB_commutation(C, v, ev_sqdiff, ctx)
        verbose_print("lap loss:", lap_loss.item())
        energy += lap_loss

    if descr_comm_mu > 0:
        # compute descriptors opt
        global commute_left
        if commute_left is None:
            verbose_print("computing commute_left")
            commute_left = [pinv1@(descr1[:, i, None] * evects1) for i in range(descr1.shape[1])]
        global commute_right
        if commute_right is None:
            verbose_print("computing commute_right")
            commute_right = [pinv2@(eta[..., None] * descr2[:, i, None] * evects2) for i in range(descr2.shape[1])]
        
        # ctx["commute_right"] = commute_right
        
        list_descr = list(zip(commute_left, commute_right))
        descr_comm_loss, ctx["descr_comm_grad"], grad_v = oplist_commutation(C, v, list_descr, ctx)
        if partial:
            raise NotImplementedError()
        descr_comm_loss *= descr_comm_mu
        verbose_print("descr comm loss:", descr_comm_loss.item())
        energy += descr_comm_loss

    if orient_mu > 0:
        # only compute this once
        global can_op1
        if can_op1 is None:
            verbose_print("computing can_op1")
            grads1 = (gradmat1.float() @ descr1.float()).reshape(f1, 3, ndescr).to(descr1.dtype)
            orient_op_hat1 = orientation_op_torch(grads1, verts1, faces1, normals1, A1[torch.arange(n1, device=A1.device), torch.arange(n1, device=A1.device)])
            pinv1_batched = pinv1.unsqueeze(0).expand(ndescr, -1, -1)  # Shape (ndescr, F, F)
            evects1_batched = evects1.unsqueeze(0).expand(ndescr, -1, -1)  # Shape (ndescr, F, F)
            orient_op_hat1_batched = orient_op_hat1.to_dense().permute(2, 0, 1)  # Shape (ndescr, F, F)
            can_op1 = torch.bmm(pinv1_batched, orient_op_hat1_batched)  # Shape (ndescr, F, F)
            can_op1 = torch.bmm(can_op1, evects1_batched)  # Shape (ndescr, F, F)
            del orient_op_hat1_batched
        
        global can_op2
        if can_op2 is None:
            verbose_print("computing can_op2")
            grads2 = (gradmat2.float() @ (eta[..., None] * descr2.float())).reshape(f2, 3, ndescr).to(descr2.dtype)
            orient_op_hat2 = orientation_op_torch(grads2, verts2, faces2, normals2, A2[torch.arange(n2, device=A2.device), torch.arange(n2, device=A2.device)])
            pinv2_batched = pinv2.unsqueeze(0).expand(ndescr, -1, -1)  # Shape (ndescr, F, F)
            evects2_batched = evects2.unsqueeze(0).expand(ndescr, -1, -1)  # Shape (ndescr, F, F)
            orient_op_hat2_batched = orient_op_hat2.to_dense().permute(2, 0, 1)  # Shape (ndescr, F, F)
            can_op2 = torch.bmm(pinv2_batched, orient_op_hat2_batched)  # Shape (ndescr, F, F)
            can_op2 = torch.bmm(can_op2, evects2_batched)  # Shape (ndescr, F, F)
            del orient_op_hat2_batched
            
        # ctx["can_op2"] = can_op2
        
        orient_op = [(can_op1[i], can_op2[i]) for i in range(ndescr)]

        orient_loss, ctx["orient_comm_grad"], grad_v = oplist_commutation(C, v, orient_op)
        if partial:
            raise NotImplementedError()
        orient_loss *= orient_mu
        verbose_print("orient loss:", orient_loss.item())
        energy += orient_loss

    if area_mu > 0:
        area_difference_loss = area_mu * area(C)
        verbose_print("area loss:", area_difference_loss.item())
        energy += area_difference_loss

    if conformal_mu > 0:
        conformal_loss = conformal_mu * conformal(C, evals1, evals2)
        verbose_print("conformal loss:", conformal_loss.item())
        energy += conformal_loss

    if p2p_mu > 0:
        p2p_loss = p2p_mu * p2p(C, v, evects1, evects2, A1, ctx)
        verbose_print("p2p loss:", p2p_loss.item())
        energy += p2p_loss

    if stochastic_mu > 0:
        stochastic_loss = stochastic_mu * doubly_stochastic(C, v, evects1, evects2, A1, ctx)
        verbose_print("stochastic loss:", stochastic_loss.item())
        energy += stochastic_loss

    if ent_mu > 0:
        ent_loss = ent_mu * entropy(C, v, evects1, evects2, A1, ctx)
        verbose_print("entropy loss:", ent_loss.item())
        energy += ent_loss
    
    if range01_mu > 0:
        range01_loss = range01_mu * range01(C, v, evects1, evects2, A1, ctx)
        verbose_print("range01 loss:", range01_loss.item())
        energy += range01_loss
        
    if sumto1_mu > 0:
        sumto1_loss = sumto1_mu * sumto1(C, v, evects1, evects2, A1, ctx)
        verbose_print("sumto1 loss:", sumto1_loss.item())
        energy += sumto1_loss

    return energy.item()


def grad_energy_std(x, partial, descr_mu, lap_mu, descr_comm_mu, orient_mu, area_mu, conformal_mu, p2p_mu, stochastic_mu, ent_mu, range01_mu, sumto1_mu, area_difference_mu, mumford_shah_mu, mumford_shah_var, eta_entropy_mu, descr1, descr2, gradmat1, gradmat2, orient_op, ev_sqdiff, evals1, evals2, evects1, evects2, A1, A2, pinv1, pinv2, verts1, verts2, faces1, faces2, normals1, normals2, ctx):
    """
    Evaluation of the gradient of the energy for standard FM computation

    Parameters:
    ----------------------
    C               :
        (K2*K1) or (K2,K1) Functional map
    descr_mu        :
        scaling of the descriptor preservation term
    lap_mu          :
        scaling of the laplacian commutativity term
    descr_comm_mu   :
        scaling of the descriptor commutativity term
    orient_mu       :
        scaling of the orientation preservation term
    descr1          :
        (K1,p) descriptors on first basis
    descr2          :
        (K2,p) descriptros on second basis
    list_descr      :
        p-uple( (K1,K1), (K2,K2) ) operators on first and second basis
                      related to descriptors.
    orient_op       :
        p-uple( (K1,K1), (K2,K2) ) operators on first and second basis
                      related to orientation preservation operators.
    ev_sqdiff       :
        (K2,K1) [normalized] matrix of squared eigenvalue differences

    Returns
    ------------------------
    gradient : float
        (K2*K1) - value of the energy
    """
    # print("backward")
    n1, k1 = evects1.shape
    n2, k2 = evects2.shape
    _, ndescr = descr1.shape
    f1, f2 = faces1.shape[0], faces2.shape[0]
    C = x[:k1 * k2].reshape((k2,k1))
    C = torch.tensor(C).to(evects1)
    if partial:
        raise NotImplementedError()
    else:
        eta = torch.ones((n2, )).to(evects1)

    # project descriptors to spectral domain
    global descr1_red
    if descr1_red is None:
        descr1_red = evects1.T @ A1 @ descr1
    descr2_red = ctx.pop("descr2_red", None)
    if descr2_red is None:
        verbose_print("Warning: computing descr2_red in backward pass")
        descr2_red = evects2.T @ A2 @ (eta[..., None] * descr2)
    
    gradient = torch.zeros_like(C)

    if descr_mu > 0:
        gradient += descr_mu * ctx.pop("descr_preservation_grad") # descr_preservation_grad(C, descr1_red, descr2_red)
        if partial:
            raise NotImplementedError()

    if lap_mu > 0:
        gradient += lap_mu * ctx.pop("LB_commutation_grad") # LB_commutation_grad(C, ev_sqdiff)
        if partial:
            raise NotImplementedError()

    if descr_comm_mu > 0:
        gradient += descr_comm_mu * ctx.pop("descr_comm_grad") #oplist_commutation_grad(C, list_descr)
        if partial:
            raise NotImplementedError()

    if orient_mu > 0:
        gradient += orient_mu * ctx.pop("orient_comm_grad") # oplist_commutation_grad(C, orient_op)
        if partial:
            raise NotImplementedError()

    if area_mu > 0:
        gradient += area_mu * area_grad(C)

    if conformal_mu > 0:
        grad_conformal = conformal_grad(C, evals1, evals2)
        gradient += conformal_mu * grad_conformal

    if p2p_mu > 0:
        p2p_grad = ctx.pop("p2p_grad")
        gradient += p2p_mu * p2p_grad
        if partial:
            raise NotImplementedError()
        
    if stochastic_mu > 0:
        stochastic_grad = ctx.pop("doubly_stochastic_grad")
        gradient += stochastic_mu * stochastic_grad
        if partial:
            raise NotImplementedError()

    if ent_mu > 0:
        ent_grad = ctx.pop("entropy_grad")
        gradient += ent_mu * ent_grad
        if partial:
            raise NotImplementedError()
    
    if range01_mu > 0:
        range01_grad = ctx.pop("range01_grad")
        gradient += range01_mu * range01_grad
        if partial:
            raise NotImplementedError()
        
    if sumto1_mu > 0:
        sumto1_grad = ctx.pop("sumto1_grad")
        gradient += sumto1_mu * sumto1_grad
        if partial:
            raise NotImplementedError()

    if partial:
        raise NotImplementedError()

    gradient[:,0] = 0
    if partial:
        raise NotImplementedError()
    else:
        return gradient.reshape(-1).float().cpu().numpy()
"""
2D Information

Improve peak fitting
    - show difference between fit and data
    - Plot breadths and asymetry

General
    - combine redundant functions
        - slice_refls
        - get_scattering_vectors
        - min_separation_check

    - variable rename
        - s1 to s1_lab
        - s1_norm to s1
"""
from cctbx import sgtbx
from cctbx import uctbx
from cctbx.crystal import symmetry
import cctbx.miller
from dials.array_family import flex
from dxtbx.model.experiment_list import ExperimentList
from dxtbx import flumpy
from itertools import combinations
from itertools import permutations
import matplotlib.pyplot as plt
import numpy as np
import numba
import os
import pandas as pd
import scipy.signal
import scipy.spatial.distance
from scipy.spatial import KDTree
import sklearn.metrics
import sklearn.cluster
import subprocess
import tqdm


def slice_refls(q2_obs, s1, start, refl_counts, refl_mask, mask):
    if mask:
        expt_refl_mask = refl_mask[start: start + refl_counts]
        expt_q2_obs = q2_obs[start: start + refl_counts][expt_refl_mask]
        expt_s1 = s1[start: start + refl_counts][expt_refl_mask]
    else:
        expt_q2_obs = q2_obs[start: start + refl_counts]
        expt_s1 = s1[start: start + refl_counts]
    start += refl_counts
    return expt_q2_obs, expt_s1, start


def get_scattering_vectors(s0, s1):
    wavelength = 1 / np.linalg.norm(s0)
    s1_normed = s1 / (wavelength * np.linalg.norm(s1, axis=1)[:, np.newaxis])
    q = s1_normed - s0
    return q


def min_separation_check(q, min_separation):
    if min_separation is None:
        return True
    else:
        q2_diff_all = np.linalg.norm(q[:, np.newaxis, :] - q[np.newaxis, :, :], axis=2)**2
        indices = np.triu_indices(q.shape[0], k=1)
        q2_diff_lattice = q2_diff_all[indices[0], indices[1]]
        if np.min(q2_diff_lattice) > min_separation:
            return True
        else:
            return False


def calc_pairwise_diff(q, q_ref=None, metric='euclidean'):
    """
    Calculate pairwise differences between vectors
    
    Parameters:
    q : array of shape (N, 3)
        Array of N 3D vectors
    metric : str
        'euclidean' for Euclidean distance
        'angular' for angular distance in radians
    
    Returns:
    distances : array of shape (N*(N-1)/2,)
        Condensed distance matrix
    """
    if metric == 'euclidean':
        return scipy.spatial.distance.pdist(q, metric='euclidean')
    
    elif metric == 'angular':
        if q_ref is None:
            assert False
        # Normalize vectors
        q_norm = q / np.linalg.norm(q, axis=1)[:, np.newaxis]
        
        # Calculate cosine distances and convert to angles
        #cos_angles = scipy.spatial.distance.pdist(q_norm, metric='cosine')
        i, j = np.triu_indices(len(q), k=1)
        cos_angles = np.dot(q_norm, q_norm.T)
        valid = np.logical_and(cos_angles >= -1, cos_angles <= 1)
        angles = np.full(cos_angles.shape, np.nan)
        angles[valid] = np.arccos(cos_angles[valid])
        angles = angles[i, j]
        
        # Calculate cross products for handedness
        # Convert to full matrix indices
        cross_products = np.cross(q_norm[i], q_norm[j])
        
        # Use z-component sign for handedness
        signs = np.sign(np.matmul(q_ref, cross_products.T))
        return signs * angles


def law_of_cosines(q2_0, q2_1, q2_diff, clip=False):
    """
    Calculate angle between vectors using law of cosines
    
    Parameters:
    q2_0 : array_like
        Magnitude squared of first vectors
    q2_1 : array_like
        Magnitude squared of second vectors
    q2_diff : array_like
        Magnitude squared of difference vectors
    clip : bool
        If True, clip cos_theta to [-1,1]
        If False, return nans for values outside [-1,1]
    
    Returns:
    angle : array_like
        Angles between vectors in radians
    """
    cos_theta = (q2_0 + q2_1 - q2_diff) / (2 * np.sqrt(q2_0 * q2_1))
    
    if clip:
        return np.arccos(np.clip(cos_theta, -1, 1))
    else:
        angles = np.full(cos_theta.shape, np.nan)
        valid = (cos_theta >= -1) & (cos_theta <= 1)
        angles[valid] = np.arccos(cos_theta[valid])
        return angles


def apply_simple_symmetry(angles):
    angles_sym = angles.copy()

    invert_and_rotate = angles_sym > np.pi/2
    if np.sum(invert_and_rotate) > 0:
        angles_sym[invert_and_rotate] = np.pi - angles_sym[invert_and_rotate]

    rotate = np.logical_and(-np.pi/2 < angles_sym, angles_sym < 0)
    if np.sum(rotate) > 0:
        angles_sym[rotate] = -angles_sym[rotate]

    invert = angles_sym < -np.pi/2
    if np.sum(invert) > 0:
        angles_sym[invert] = np.pi + angles_sym[invert]
    return angles_sym


def manual_cross(q1, q2):
    # This is actually much faster than np.cross
    return np.array([
        q1[1] * q2[2] - q1[2] * q2[1],
        q1[2] * q2[0] - q1[0] * q2[2],
        q1[0] * q2[1] - q1[1] * q2[0]
    ])


@numba.njit
def get_R_point_to_ref(q_move, q_ref):
    """
    Rotates q_move onto q_ref using Numba for acceleration.
    
    Parameters:
    q_move, q_ref: 3D vectors
    
    Returns:
    R: 3x3 rotation matrix
    """
    # Calculate norms directly
    q_move_len = np.sqrt(q_move[0]**2 + q_move[1]**2 + q_move[2]**2)
    q_ref_len = np.sqrt(q_ref[0]**2 + q_ref[1]**2 + q_ref[2]**2)
    
    # Normalize vectors
    q_move_norm = np.empty(3)
    q_ref_norm = np.empty(3)
    
    for i in range(3):
        q_move_norm[i] = q_move[i] / q_move_len
        q_ref_norm[i] = q_ref[i] / q_ref_len
    
    # Calculate cross product
    rotation_axis = np.empty(3)
    rotation_axis[0] = q_move_norm[1] * q_ref_norm[2] - q_move_norm[2] * q_ref_norm[1]
    rotation_axis[1] = q_move_norm[2] * q_ref_norm[0] - q_move_norm[0] * q_ref_norm[2]
    rotation_axis[2] = q_move_norm[0] * q_ref_norm[1] - q_move_norm[1] * q_ref_norm[0]
    
    # Check if vectors are parallel or anti-parallel
    axis_len_sq = rotation_axis[0]**2 + rotation_axis[1]**2 + rotation_axis[2]**2
    
    if axis_len_sq < 1e-10:
        # Vectors are parallel or anti-parallel
        dot_product = q_move_norm[0]*q_ref_norm[0] + q_move_norm[1]*q_ref_norm[1] + q_move_norm[2]*q_ref_norm[2]
        
        if dot_product < 0:  # Anti-parallel
            # Find perpendicular vector
            perp = np.empty(3)
            
            if abs(q_move_norm[0]) < abs(q_move_norm[1]):
                perp[0] = 1.0
                perp[1] = 0.0
                perp[2] = 0.0
            else:
                perp[0] = 0.0
                perp[1] = 1.0
                perp[2] = 0.0
            
            # Make perp orthogonal to q_move_norm
            dot_perp_move = perp[0]*q_move_norm[0] + perp[1]*q_move_norm[1] + perp[2]*q_move_norm[2]
            
            for i in range(3):
                perp[i] = perp[i] - dot_perp_move * q_move_norm[i]
            
            # Normalize perp
            perp_len = np.sqrt(perp[0]**2 + perp[1]**2 + perp[2]**2)
            for i in range(3):
                perp[i] /= perp_len
            
            # Create rotation matrix for 180° rotation around perp
            # For 180° rotation: sin(π)=0, 1-cos(π)=-2
            # R = I - 2 * (perp ⊗ perp) = I - 2 * (perp * perp^T)
            R = np.empty((3, 3))
            
            # Diagonal elements: 1 - 2 * perp[i]^2
            R[0, 0] = 1.0 - 2.0 * perp[0]**2
            R[1, 1] = 1.0 - 2.0 * perp[1]**2
            R[2, 2] = 1.0 - 2.0 * perp[2]**2
            
            # Off-diagonal elements: -2 * perp[i] * perp[j]
            R[0, 1] = -2.0 * perp[0] * perp[1]
            R[0, 2] = -2.0 * perp[0] * perp[2]
            R[1, 0] = -2.0 * perp[1] * perp[0]
            R[1, 2] = -2.0 * perp[1] * perp[2]
            R[2, 0] = -2.0 * perp[2] * perp[0]
            R[2, 1] = -2.0 * perp[2] * perp[1]
        else:
            # Vectors are parallel, return identity matrix
            R = np.eye(3)
    else:
        # Normalize rotation axis
        axis_len = np.sqrt(axis_len_sq)
        for i in range(3):
            rotation_axis[i] /= axis_len
        
        # Calculate angle
        dot_product = q_move_norm[0]*q_ref_norm[0] + q_move_norm[1]*q_ref_norm[1] + q_move_norm[2]*q_ref_norm[2]
        dot_product = max(min(dot_product, 1.0), -1.0)  # Clip to [-1, 1]
        angle = np.arccos(dot_product)
        
        # Compute sin and cos once
        sin_angle = np.sin(angle)
        cos_angle = np.cos(angle)
        one_minus_cos = 1.0 - cos_angle
        
        # Create rotation matrix using Rodrigues formula
        R = np.empty((3, 3))
        
        # Diagonal elements
        R[0, 0] = cos_angle + one_minus_cos * rotation_axis[0]**2
        R[1, 1] = cos_angle + one_minus_cos * rotation_axis[1]**2
        R[2, 2] = cos_angle + one_minus_cos * rotation_axis[2]**2
        
        # Off-diagonal elements
        R[0, 1] = one_minus_cos * rotation_axis[0] * rotation_axis[1] - sin_angle * rotation_axis[2]
        R[0, 2] = one_minus_cos * rotation_axis[0] * rotation_axis[2] + sin_angle * rotation_axis[1]
        R[1, 0] = one_minus_cos * rotation_axis[0] * rotation_axis[1] + sin_angle * rotation_axis[2]
        R[1, 2] = one_minus_cos * rotation_axis[1] * rotation_axis[2] - sin_angle * rotation_axis[0]
        R[2, 0] = one_minus_cos * rotation_axis[0] * rotation_axis[2] - sin_angle * rotation_axis[1]
        R[2, 1] = one_minus_cos * rotation_axis[1] * rotation_axis[2] + sin_angle * rotation_axis[0]
    
    return R


@numba.njit
def get_R_point_to_ref_along_axis(q_move, q_ref, rotation_axis):
    # Pre-compute norms
    q_move_len = np.sqrt(q_move[0]**2 + q_move[1]**2 + q_move[2]**2)
    q_ref_len = np.sqrt(q_ref[0]**2 + q_ref[1]**2 + q_ref[2]**2)
    axis_len = np.sqrt(rotation_axis[0]**2 + rotation_axis[1]**2 + rotation_axis[2]**2)
    
    # Normalize vectors
    q_move_norm = np.empty(3)
    q_ref_norm = np.empty(3)
    axis_norm = np.empty(3)
    
    for i in range(3):
        q_move_norm[i] = q_move[i] / q_move_len
        q_ref_norm[i] = q_ref[i] / q_ref_len
        axis_norm[i] = rotation_axis[i] / axis_len
    
    # Compute dot products
    dot_move_axis = q_move_norm[0]*axis_norm[0] + q_move_norm[1]*axis_norm[1] + q_move_norm[2]*axis_norm[2]
    dot_ref_axis = q_ref_norm[0]*axis_norm[0] + q_ref_norm[1]*axis_norm[1] + q_ref_norm[2]*axis_norm[2]
    
    # Project vectors
    q_move_proj = np.empty(3)
    q_ref_proj = np.empty(3)
    
    for i in range(3):
        q_move_proj[i] = q_move_norm[i] - dot_move_axis * axis_norm[i]
        q_ref_proj[i] = q_ref_norm[i] - dot_ref_axis * axis_norm[i]
    
    # Check projection norms
    move_proj_len_sq = q_move_proj[0]**2 + q_move_proj[1]**2 + q_move_proj[2]**2
    ref_proj_len_sq = q_ref_proj[0]**2 + q_ref_proj[1]**2 + q_ref_proj[2]**2
    
    if move_proj_len_sq < 1e-10 or ref_proj_len_sq < 1e-10:
        return np.eye(3)
    
    # Normalize projections
    move_proj_len = np.sqrt(move_proj_len_sq)
    ref_proj_len = np.sqrt(ref_proj_len_sq)
    
    for i in range(3):
        q_move_proj[i] /= move_proj_len
        q_ref_proj[i] /= ref_proj_len
    
    # Calculate angle
    dot_proj = q_move_proj[0]*q_ref_proj[0] + q_move_proj[1]*q_ref_proj[1] + q_move_proj[2]*q_ref_proj[2]
    dot_proj = max(min(dot_proj, 1.0), -1.0)
    angle = np.arccos(dot_proj)
    
    # Determine rotation direction with cross product
    cross = np.empty(3)
    cross[0] = q_move_proj[1]*q_ref_proj[2] - q_move_proj[2]*q_ref_proj[1]
    cross[1] = q_move_proj[2]*q_ref_proj[0] - q_move_proj[0]*q_ref_proj[2]
    cross[2] = q_move_proj[0]*q_ref_proj[1] - q_move_proj[1]*q_ref_proj[0]
    
    dot_cross_axis = cross[0]*axis_norm[0] + cross[1]*axis_norm[1] + cross[2]*axis_norm[2]
    if dot_cross_axis < 0:
        angle = -angle
    
    # Compute rotation matrix
    sin_angle = np.sin(angle)
    cos_angle = np.cos(angle)
    one_minus_cos = 1.0 - cos_angle
    
    R = np.empty((3, 3))
    
    # Diagonal elements
    R[0, 0] = cos_angle + one_minus_cos * axis_norm[0]**2
    R[1, 1] = cos_angle + one_minus_cos * axis_norm[1]**2
    R[2, 2] = cos_angle + one_minus_cos * axis_norm[2]**2
    
    # Off-diagonal elements
    R[0, 1] = one_minus_cos * axis_norm[0] * axis_norm[1] - sin_angle * axis_norm[2]
    R[0, 2] = one_minus_cos * axis_norm[0] * axis_norm[2] + sin_angle * axis_norm[1]
    R[1, 0] = one_minus_cos * axis_norm[0] * axis_norm[1] + sin_angle * axis_norm[2]
    R[1, 2] = one_minus_cos * axis_norm[1] * axis_norm[2] - sin_angle * axis_norm[0]
    R[2, 0] = one_minus_cos * axis_norm[0] * axis_norm[2] - sin_angle * axis_norm[1]
    R[2, 1] = one_minus_cos * axis_norm[1] * axis_norm[2] + sin_angle * axis_norm[0]
    
    return R


def remove_duplicate_rows_kdtree(arr, red_tolerance=0.01):
    if len(arr) <= 1:
        return arr
    
    # Build KD-Tree
    tree = KDTree(arr)
    
    # Keep track of which rows to keep
    keep = np.ones(len(arr), dtype=bool)
    
    for i in range(len(arr)):
        if keep[i]:
            # Find all points within tolerance of this point
            # Exclude the point itself (which has distance 0)
            neighbors = tree.query_ball_point(arr[i], red_tolerance)
            
            # Remove duplicates (keep the first occurrence)
            for j in neighbors:
                if j > i:  # Only affect points after current one
                    keep[j] = False
    
    return arr[keep]


def combine_graphs_sub_func(graph_ref, graph_add, i_ref, j_ref, i_add, j_add, comb_tolerance, red_tolerance):
    peak_indices_ref = graph_ref[:, 3]
    peak_indices_add = graph_add[:, 3]
    q_ref = graph_ref[:, :3]
    q_add = graph_add[:, :3]

    # Step 1: Get rotation matrix that rotates q_add_i onto q_ref_i:
    R1 = get_R_point_to_ref(q_add[i_add], q_ref[i_ref])
    q_ref_1 = (R1 @ q_ref.T).T
    q_add_1 = (R1 @ q_add.T).T

    # Step 2: Rotate q_add_j onto q_ref_j while keeping q_ref_i in place.
    R2 = get_R_point_to_ref_along_axis(q_add[j_add], q_ref[j_ref], q_ref[i_ref])
    q_ref_2 = (R2 @ q_ref_1.T).T
    q_add_2 = (R2 @ q_add_1.T).T

    # Step 3: Determine in the tolerance is okay
    i_err = np.linalg.norm(q_ref_2[i_ref] - q_add_2[i_add])
    j_err = np.linalg.norm(q_ref_2[j_ref] - q_add_2[j_add])

    if i_err < comb_tolerance and j_err < comb_tolerance:
        # Step 4: Combine q_add and q_ref
        #   Recombine the indices
        graph_ref_2 = np.concatenate((q_ref_2, peak_indices_ref[:, np.newaxis]), axis=1)
        graph_add_2 = np.concatenate((q_add_2, peak_indices_add[:, np.newaxis]), axis=1)
        #   Add delete the two peaks that were overlapped
        graph_add_2 = np.delete(graph_add_2, [i_add, j_add], axis=0)
        #   Then stack the peaks
        graph_combined = np.concatenate([graph_ref_2, graph_add_2], axis=0)
        #   Remove redundancies
        graph_combined = remove_duplicate_rows_kdtree(graph_combined, red_tolerance)
        # Only return a new graph if this algorithm expands the graph
        if graph_combined.shape[0] > graph_ref.shape[0]:
            return [graph_combined]
        else:
            return []
    else:
        return []


def combine_graphs(graph_ref, graph_add, comb_tolerance, red_tolerance):
    peak_indices_ref = graph_ref[:, 3]
    peak_indices_add = graph_add[:, 3]
    # attempt to add each possible combination of peaks
    graphs_new = []
    for index0_ref in range(peak_indices_ref.size - 1):
        peak_index0_ref = peak_indices_ref[index0_ref]
        for index1_ref in range(index0_ref + 1, peak_indices_ref.size):
            peak_index1_ref = peak_indices_ref[index1_ref]
            for index0_add in range(peak_indices_add.size - 1):
                peak_index0_add = peak_indices_add[index0_add]
                for index1_add in range(index0_add + 1, peak_indices_add.size):
                    peak_index1_add = peak_indices_add[index1_add]
                    check0 = (peak_index0_ref == peak_index0_add) and (peak_index1_ref == peak_index1_add)
                    check1 = (peak_index0_ref == peak_index1_add) and (peak_index1_ref == peak_index0_add)
                    if check0:
                        graphs_new += combine_graphs_sub_func(
                            graph_ref, graph_add,
                            index0_ref, index1_ref,
                            index0_add, index1_add,
                            comb_tolerance, red_tolerance
                            )
                    if check1:
                        graphs_new += combine_graphs_sub_func(
                            graph_ref, graph_add,
                            index0_ref, index1_ref,
                            index1_add, index0_add,
                            comb_tolerance, red_tolerance
                            )
    return graphs_new


def kabsch_align(P, Q):
    """
    Align points in P to points in Q using the Kabsch algorithm.
    
    Parameters:
    P: numpy array of shape (N, 3) - points to be aligned
    Q: numpy array of shape (N, 3) - reference points
    
    Returns:
    P_aligned: numpy array of shape (N, 3) - aligned points
    R: numpy array of shape (3, 3) - rotation matrix
    t: numpy array of shape (3,) - translation vector
    rmsd: float - root-mean-square deviation after alignment
    """
    # Ensure inputs are numpy arrays
    P = np.array(P, dtype=np.float64)
    Q = np.array(Q, dtype=np.float64)
    
    if P.shape != Q.shape:
        raise ValueError("Input point sets must have the same shape")
    
    # Center the points
    P_centroid = np.mean(P, axis=0)
    Q_centroid = np.mean(Q, axis=0)
    
    P_centered = P - P_centroid
    Q_centered = Q - Q_centroid
    
    # Compute the covariance matrix
    H = P_centered.T @ Q_centered
    
    # Singular value decomposition
    U, _, Vt = np.linalg.svd(H)
    
    # Ensure proper rotation (handle reflection case)
    # Determinant of R should be 1 for a proper rotation
    V = Vt.T
    det = np.linalg.det(V @ U.T)
    
    if det < 0:
        V[:, 2] = -V[:, 2]
    
    # Calculate rotation matrix
    R = V @ U.T
    
    # Calculate translation
    t = Q_centroid - R @ P_centroid
    
    # Apply rotation and translation to P
    P_aligned = (R @ P.T).T + t
    
    # Calculate RMSD
    rmsd = np.sqrt(np.mean(np.sum((P_aligned - Q)**2, axis=1)))
    
    return P_aligned, R, t, rmsd


def arrays_equivalent(arr1, arr2, red_tolerance=0.01):
    """
    Check if two arrays are equivalent up to row permutation within tolerance.
    
    Args:
        arr1, arr2: Arrays of shape (N1 x 4) and (N2 x 4)
        tolerance: Maximum allowed difference between corresponding elements
        
    Returns:
        bool: True if arrays are equivalent, False otherwise
    """
    # Quick check: different number of rows means not equivalent
    if arr1.shape[0] != arr2.shape[0]:
        return False
    
    n_rows = arr1.shape[0]
    arr1_sorted = arr1[np.argsort(arr1[:, 3])]
    arr2_sorted = arr2[np.argsort(arr2[:, 3])]
    for index in range(3):
        arr1_sorted_index = arr1_sorted[np.argsort(arr1_sorted[:, index])]
        arr2_sorted_index = arr2_sorted[np.argsort(arr2_sorted[:, index])]
        if np.allclose(arr1_sorted_index, arr2_sorted_index, atol=red_tolerance):
            return True
        arr1_sorted_index_aligned, _, _, _ = kabsch_align(arr1_sorted_index[:, :3], arr2_sorted_index[:, :3])
        arr1_sorted_index_aligned = np.concatenate([
            arr1_sorted_index_aligned,
            arr1_sorted_index[:, 3][:, np.newaxis]
            ], axis=1
            )
        if np.allclose(arr1_sorted_index_aligned, arr2_sorted_index, atol=red_tolerance):
            return True
    return False


def remove_redundant_arrays_slow(array_list, red_tolerance=0.01):
    """
    Remove redundant arrays from a list, considering row permutations.
    
    Args:
        array_list: List of numpy arrays, each with shape (N x 4)
        tolerance: Tolerance for numerical comparison
        
    Returns:
        List of unique arrays
    """
    if not array_list:
        return []
    
    unique_arrays = [array_list[0]]
    
    for i in range(1, len(array_list)):
        current_array = array_list[i]
        is_redundant = False
        
        for unique_array in unique_arrays:
            if arrays_equivalent(current_array, unique_array, red_tolerance):
                is_redundant = True
                break
                
        if not is_redundant:
            unique_arrays.append(current_array)
            
    return unique_arrays


def remove_redundant_arrays(array_list, red_tolerance=0.01):
    """
    Remove redundant arrays using a hash-based approach.
    """
    if not array_list:
        return []
    
    unique_arrays = []
    # Dictionary to store sorted arrays for quick lookup
    array_dict = {}
    
    for arr in array_list:
        # Sort the array and create a hashable representation
        sorted_arr = arr[np.lexsort(arr.T)]
        # Round to handle floating point tolerance
        rounded_arr = np.round(sorted_arr / red_tolerance) * red_tolerance
        # Convert to tuple for hashing
        arr_key = tuple(map(tuple, rounded_arr))
        
        if arr_key not in array_dict:
            array_dict[arr_key] = arr
            unique_arrays.append(arr)
    return unique_arrays


@numba.njit
def calculate_basis_error(potential_basis, points):
    """Calculate how well a basis represents points with integer coefficients."""
    try:
        # Calculate inverse of basis matrix
        basis_inv = np.linalg.inv(potential_basis)
        
        # Calculate coefficients for all points
        coeffs = basis_inv @ points.T
        coeffs_rounded = np.round(coeffs)
        
        # Calculate error and integer score
        reconstructed = (potential_basis @ coeffs_rounded).T
        errors = np.zeros(points.shape[0])
        for i in range(points.shape[0]):
            errors[i] = np.sqrt(np.sum((points[i] - reconstructed[i])**2))
        
        mean_error = np.mean(errors)
        
        # Count points that are represented exactly by integers
        integer_score = 0
        for i in range(points.shape[0]):
            is_integer = True
            for j in range(3):
                if abs(coeffs[j, i] - coeffs_rounded[j, i]) >= 0.05:
                    is_integer = False
                    break
            if is_integer:
                integer_score += 1
                
        return integer_score, mean_error, basis_inv
    except:
        # If matrix inversion fails
        return -1, float('inf'), np.zeros((3, 3))


def find_grid_basis(points, verbose=True):
    """
    Find the best basis vectors for a grid of points by trying combinations of points.
    
    Parameters:
    points : numpy.ndarray
        N x 3 array of grid point coordinates (already differences)
    
    Returns:
    A : numpy.ndarray
        3 x 3 matrix representing the best basis vectors
    """
    n_points = points.shape[0]
    
    # We need at least 3 points to form a basis
    if n_points < 3:
        raise ValueError("Need at least 3 points to form a basis")
    
    best_basis = None
    best_error = float('inf')
    best_integer_score = 0
    best_det = 0

    distance = np.linalg.norm(points, axis=1)
    points = points[np.argsort(distance)]

    # Try all combinations of 3 points as potential basis vectors
    for indices in combinations(range(int(n_points/5)), 3):
        # Form a potential basis from these three points
        potential_basis = points[list(indices)].T  # 3x3 matrix
        
        # Check if this basis is valid (linearly independent)
        if np.abs(np.linalg.det(potential_basis)) < 1e-6:
            continue  # Singular matrix, not a valid basis
        
        # Calculate how well this basis represents all points with integer coefficients
        try:
            integer_score, error, refined_basis = calculate_basis_error_numba(potential_basis, points)
            # We prefer bases that represent more points with integers
            # If tied, we prefer the one with lower overall error
            if integer_score > best_integer_score:
                best_basis = potential_basis
                best_error = error
                best_integer_score = integer_score
                best_det = np.linalg.det(potential_basis)
            elif integer_score == best_integer_score:
                if error < best_error:
                    best_basis = potential_basis
                    best_error = error
                    best_integer_score = integer_score
                    best_det = np.linalg.det(potential_basis)

        except np.linalg.LinAlgError:
            # Skip if matrix inversion fails
            continue
    
    if best_basis is None:
        return None, None, None
    # Report how well the basis performs
    refined_integer_score, refined_error, refined_basis = refine_basis(best_basis, points)
    if verbose:
        print(f"Found basis representing {refined_integer_score}/{n_points} points exactly with integers")
        print(f"Total squared error for non-integer coefficients: {refined_error}")
        print(refined_basis)
    return refined_basis, refined_integer_score, refined_error


def refine_basis(basis, points):
    """Calculate how well a basis represents points with integer coefficients."""
    basis_inv = np.linalg.inv(basis)

    # Calculate coefficients for all points
    coeffs = basis_inv @ points.T
    coeffs_rounded = np.round(coeffs)
    indexed = np.all(np.abs(coeffs - coeffs_rounded) < 0.2, axis=0)

    # Extract the integer-indexed points and their rounded coefficients
    indexed_points = points[indexed]
    indexed_coeffs = coeffs_rounded[:, indexed]  # 3 x N matrix of integer coefficients

    """
    nx3        3x3     3xn        nx3       3x3
    a      =   X    @    b         b     @   X
    points = (basis @ coeff).T = coeff.T @ basis.T

    3xn       nx3       3x3
    a      =  b         X
    points = coeff.T @ basis.T
    """
    refined_basis, residuals, rank, s = np.linalg.lstsq(indexed_coeffs.T, indexed_points, rcond=None)
    refined_basis = refined_basis.T

    try:
        refined_inv = np.linalg.inv(refined_basis)
    except:
        # If there's an error with the refined basis, return the original results
        original_error = np.mean(np.linalg.norm(points - (potential_basis @ coeffs_rounded).T, axis=1))
        return integer_score, original_error, potential_basis

    refined_coeffs = refined_inv @ points.T
    refined_coeffs_rounded = np.round(refined_coeffs)
    
    # Recalculate error and integer score
    reconstructed = (refined_basis @ refined_coeffs_rounded).T
    errors = np.linalg.norm(points - reconstructed, axis=1)
    mean_error = np.mean(errors)
    
    # Count points that can be represented exactly with integers
    refined_indexed = np.all(np.abs(refined_coeffs - refined_coeffs_rounded) < 0.05, axis=0)
    refined_integer_score = np.sum(refined_indexed)
    return refined_integer_score, mean_error, refined_basis


@numba.njit
def calculate_basis_error_numba(potential_basis, points):
    """Calculate how well a basis represents points with integer coefficients."""
    try:
        # Calculate inverse of basis matrix
        basis_inv = np.linalg.inv(potential_basis)
        
        # Calculate coefficients for all points
        coeffs = basis_inv @ points.T
        coeffs_rounded = np.round(coeffs)
        
        # Calculate error and integer score
        reconstructed = (potential_basis @ coeffs_rounded).T
        errors = np.zeros(points.shape[0])
        for i in range(points.shape[0]):
            errors[i] = np.sqrt(np.sum((points[i] - reconstructed[i])**2))
        
        mean_error = np.mean(errors)
        
        # Count points that are represented exactly by integers
        integer_score = 0
        for i in range(points.shape[0]):
            is_integer = True
            for j in range(3):
                if abs(coeffs[j, i] - coeffs_rounded[j, i]) >= 0.05:
                    is_integer = False
                    break
            if is_integer:
                integer_score += 1
                
        return integer_score, mean_error, basis_inv
    except:
        # If matrix inversion fails
        return -1, float('inf'), np.zeros((3, 3))


class PeakListCreator:
    def __init__(
        self, 
        tag,
        save_to_directory=None,
        load_combined=False,
        overwrite_combined=False,
        runs=None,
        run_limits=None,
        run_limits_sacla=None,
        input_path_template=None,
        suffix='_strong.expt',
        min_reflections_per_experiment=3,
        max_reflections_per_experiment=100,
        known_unit_cell=None, 
        known_space_group=None,
    ):
        
        if type(input_path_template) == str or input_path_template is None:
            self.multiple_run_groups = False
            self.input_path_template = [input_path_template]
        elif type(input_path_template) == list:
            self.multiple_run_groups = True
            self.input_path_template = input_path_template
        if not run_limits_sacla is None:
            assert False
            self.runs = []
            for run_index in range(run_limits_sacla[0], run_limits_sacla[1] + 1):
                for sub_run_index in range(3):
                    self.runs.append(f'{run_index}-{sub_run_index}')
        elif not run_limits is None:
            if self.multiple_run_groups:
                self.runs = [np.arange(rl[0], rl[1] + 1) for rl in run_limits]
            else:
                self.runs = [np.arange(run_limits[0], run_limits[1] + 1)]
        else:
            if self.multiple_run_groups:
                self.runs = runs
            else:
                self.runs = [runs]

        self.max_reflections_per_experiment = max_reflections_per_experiment
        self.min_reflections_per_experiment = min_reflections_per_experiment
        
        self.suffix = suffix
        self.tag = tag
        self.load_combined = load_combined
        self.overwrite_combined = overwrite_combined
        if save_to_directory is None:
            self.save_to_directory = os.path.join(os.getcwd(), self.tag)
        else:
            self.save_to_directory = os.path.join(save_to_directory, self.tag)
        if not os.path.exists(self.save_to_directory):
            os.mkdir(self.save_to_directory)
        
        self.expt_file_name = os.path.join(self.save_to_directory, f'{self.tag}_combined_all.expt')
        self.refl_file_name = os.path.join(self.save_to_directory, f'{self.tag}_combined_all.refl')
        if self.load_combined == False:
            self._combine_expt_refl_files()
            self._parse_refl_file()
        else:
            self.q2_obs = np.load(
                os.path.join(self.save_to_directory, f'{self.tag}_q2_obs.npy'),
                )
            self.refl_counts = np.load(
                os.path.join(self.save_to_directory, f'{self.tag}_refl_counts.npy'),
                )
            self.expt_indices = np.load(
                os.path.join(self.save_to_directory, f'{self.tag}_expt_indices.npy'),
                )
            self.s0 = np.load(
                os.path.join(self.save_to_directory, f'{self.tag}_s0.npy'),
                )
            self.s1 = np.load(
                os.path.join(self.save_to_directory, f'{self.tag}_s1.npy'),
                )
        self.beam_delta = np.zeros(2)
        self.refl_mask = np.ones(self.q2_obs.size, dtype=bool)
        self.known_unit_cell = known_unit_cell
        self.known_space_group = known_space_group
        self.error = None
        self.triplets_obs = None

    def _run_combine_experiments(self, expt_file_names, refl_file_names, run_str, ref=True):
        command = ['dials.combine_experiments']
        command += expt_file_names
        command += refl_file_names
        if ref:
            command += ['reference_from_experiment.detector=0']
        command += [
            f'max_reflections_per_experiment={self.max_reflections_per_experiment}',
            f'min_reflections_per_experiment={self.min_reflections_per_experiment}',
            f'output.experiments_filename={self.tag}_combined_{run_str}.expt',
            f'output.reflections_filename={self.tag}_combined_{run_str}.refl',
            ]
        log_file_name = os.path.join(
            self.save_to_directory,
            f'{self.tag}_combine_experiments_{run_str}.log'
            )

        output_refl_file_name = os.path.join(
            self.save_to_directory,
            f'{self.tag}_combined_{run_str}.refl'
            )
        if self.overwrite_combined == False and os.path.exists(output_refl_file_name):
            print(f'Loading combined expt and refls for run {run_str}')
        else:
            print(f'Combining experiments in run {run_str}')
            with open(log_file_name, 'w') as log_file:
                subprocess.run(command, cwd=self.save_to_directory, stdout=log_file)
        with open(log_file_name, 'r') as log_file:
            expt_counts = 0
            refl_counts = 0
            count = False
            for line in log_file:
                if line == '+--------------+-------------------------+\n':
                    count = False
                if count:
                    expt_counts = int(line.split('|')[1])
                    refl_counts += int(line.split('|')[2])
                if line == '|--------------+-------------------------|\n':
                    count = True
        print(f'    Run {run_str} has {expt_counts} experiments and {refl_counts} reflections')
        return refl_counts

    def _combine_expt_refl_files(self):
        expt_file_names = []
        refl_file_names = []
        for rg_index in range(len(self.input_path_template)):
            expt_file_names_rg = []
            refl_file_names_rg = []
            for run in self.runs[rg_index]:
                expt_file_names_run = []
                refl_file_names_run = []
                if type(run) == str:
                    run_str = run
                else:
                    run_str = f'{run:04d}'
                input_path = self.input_path_template[rg_index].replace('!!!!', run_str)
                if os.path.exists(input_path):
                    for file_name in os.listdir(input_path):
                        if file_name.endswith(self.suffix):
                            expt_file_name = os.path.join(input_path, file_name)
                            refl_file_name = os.path.join(input_path, file_name.replace('.expt', '.refl'))
                            if os.path.exists(expt_file_name) and os.path.exists(refl_file_name):
                                expt_file_names_run.append(expt_file_name)
                                refl_file_names_run.append(refl_file_name)
                    if len(expt_file_names_run) > 0:
                        refl_counts = self._run_combine_experiments(
                            expt_file_names_run, refl_file_names_run, run_str, ref=True
                            )
                        if refl_counts > 0:
                            expt_file_names_rg.append(os.path.join(
                                self.save_to_directory, f'{self.tag}_combined_{run_str}.expt'
                                ))
                            refl_file_names_rg.append(os.path.join(
                                self.save_to_directory, f'{self.tag}_combined_{run_str}.refl'
                                ))
            self._run_combine_experiments(
                expt_file_names_rg, refl_file_names_rg, f'rg_index_{rg_index}', ref=True
                )
            expt_file_names.append(os.path.join(
                self.save_to_directory, f'{self.tag}_combined_rg_index_{rg_index}.expt'
                ))
            refl_file_names.append(os.path.join(
                self.save_to_directory, f'{self.tag}_combined_rg_index_{rg_index}.refl'
                ))
        self._run_combine_experiments(
            expt_file_names, refl_file_names, 'all', ref=False
            )
    
    def _get_s1_from_xyz(self, panel, xyz, wavelength):
        s1 = flumpy.to_numpy(
                panel.get_lab_coord(panel.pixel_to_millimeter(flex.vec2_double(
                    flex.double(xyz[:, 0].ravel()),
                    flex.double(xyz[:, 1].ravel())
                )))
            )
        # s1 is the vector going from the interation point to the peak with magnitude 1/wavelength
        s1_normed = s1 / (wavelength * np.linalg.norm(s1, axis=1)[:, np.newaxis])
        return s1_normed, s1

    def _get_q2_from_xyz(self, panel, xyz, s0):
        return np.array([1 / panel.get_resolution_at_pixel(s0, xyz[i][0:2])**2 for i in range(len(xyz))])

    def _get_q2_spacing(self, s1, s0):
        wavelength = 1 / np.linalg.norm(s0)
        dot_product = np.matmul(s1, s0)
        magnitudes = np.linalg.norm(s1, axis=1) * np.linalg.norm(s0)
        theta2 = np.arccos(dot_product / magnitudes)
        return ((2 * np.sin(theta2 / 2)) / wavelength)**2

    def _parse_refl_file(self):
        expts = ExperimentList.from_file(self.expt_file_name, check_format=False)
        refls = flex.reflection_table.from_file(self.refl_file_name)
        q2 = []
        s1 = []
        s0 = []
        expt_indices = []
        refl_counts = []
        print('Parsing Reflection File')
        for expt_index, expt in enumerate(expts):
            refls_expt = refls.select(refls['id'] == expt_index)
            if len(refls_expt) > 0:
                wavelength = expt.beam.get_wavelength()
                s0_lattice = expt.beam.get_s0() #|s0| = 1/wavelength
                # s1 is the vector going from the interaction point to the crystal
                # s1_normed has magnitude 1/wavelength
                s1_normed_lattice = []
                s1_lattice = []
                for panel_index, panel in enumerate(expt.detector):
                    refls_panel = refls_expt.select(refls_expt['panel'] == panel_index)
                    if len(refls_panel) > 0:
                        s1_normed_panel, s1_panel = self._get_s1_from_xyz(
                            panel, 
                            flumpy.to_numpy(refls_panel['xyzobs.px.value']), 
                            wavelength,
                            )
                        s1_normed_lattice.append(s1_normed_panel)
                        s1_lattice.append(s1_panel)
                s1_lattice = np.row_stack(s1_lattice)
                refl_counts.append(s1_lattice.shape[0])
                # s0 and s1 are retained for constructing secondary peaks and beam center optimization
                s1.append(np.row_stack(s1_lattice))
                s0.append(s0_lattice)
                expt_indices.append(expt_index*np.ones(s1_lattice.shape[0], dtype=int))
                # q2_lattice is the magnitude**2 of the scattering vector
                q2.append(self._get_q2_spacing(
                    np.row_stack(s1_normed_lattice), s0_lattice)
                    )
        self.q2_obs = np.concatenate(q2)
        self.refl_counts = np.array(refl_counts)
        self.expt_indices = np.concatenate(expt_indices)
        self.s0 = np.row_stack(s0)
        self.s1 = np.row_stack(s1)
        np.save(
            os.path.join(self.save_to_directory, f'{self.tag}_q2_obs.npy'),
            self.q2_obs
            )
        np.save(
            os.path.join(self.save_to_directory, f'{self.tag}_refl_counts.npy'),
            self.refl_counts
            )
        np.save(
            os.path.join(self.save_to_directory, f'{self.tag}_expt_indices.npy'),
            self.expt_indices
            )
        np.save(
            os.path.join(self.save_to_directory, f'{self.tag}_s0.npy'),
            self.s0
            )
        np.save(
            os.path.join(self.save_to_directory, f'{self.tag}_s1.npy'),
            self.s1
            )

    def quick_mask(self, n_bins=1000, threshold=20, pad=5, llur=None):
        """
        Masking algorithm:
            1: Calculate a 2D histogram of the reflection positions on the detector surface.
            2: Calculate an azimuthal average of the reflection positions.
            3: Project the azimuthal average onto the detector surface.
            4: Mask regions where the histogram is much larger than the azimuthal average.
        """
        # Array rows are coordinate y
        # Array cols are coordinate x
        bins_x = np.linspace(self.s1[:, 1].min(), self.s1[:, 1].max(), n_bins + 1)
        bins_y = np.linspace(self.s1[:, 0].min(), self.s1[:, 0].max(), n_bins + 1)
        centers_x = (bins_x[1:] + bins_x[:-1]) / 2
        centers_y = (bins_y[1:] + bins_y[:-1]) / 2

        # 2D histogram of the reflection positions
        hist, _, _ = np.histogram2d(x=self.s1[:, 1], y=self.s1[:, 0], bins=[bins_x, bins_y])

        # This maps the reflections onto the histogram coordinates
        refl_x = np.searchsorted(bins_x, self.s1[:, 1]) - 1
        refl_y = np.searchsorted(bins_y, self.s1[:, 0]) - 1
        refl_x[refl_x == -1] = 0
        refl_y[refl_y == -1] = 0

        # This should be the correct way to do this, the detector distance should be the average of
        # the detector distance of the reflections in the xy bin. This does not work though.
        # Using the same detector distance of 
        #centers_z, _, _, _ = scipy.stats.binned_statistic_2d(
        #    x=self.s1[:, 1],
        #    y=self.s1[:, 0], 
        #    values=self.s1[:, 2],
        #    bins=[bins_x, bins_y],
        #    statistic='mean'
        #    )
        #centers_z[np.isnan(centers_z)] = np.nanmean(centers_z)
        centers_z = self.s1[:, 2].mean()

        # This performs the azimuthal average and projection onto the detector surface.
        s1_lab_mag_centers = centers_x[np.newaxis, :]**2 + centers_y[:, np.newaxis]**2 + centers_z**2
        s1_lab_mag_bins = np.linspace(s1_lab_mag_centers.min(), s1_lab_mag_centers.max(), int(n_bins/2) + 1)
        azimuthal_mean, _, _ = scipy.stats.binned_statistic(
            x=s1_lab_mag_centers.ravel(), values=hist.ravel(), statistic='mean', bins=s1_lab_mag_bins
            )
        indices = np.searchsorted(s1_lab_mag_bins, s1_lab_mag_centers) - 1
        indices[indices == -1] = 0
        mean_projection = np.take(azimuthal_mean, indices)
        # This takes all the zero pixels and makes them nonzero to prevent large amounts of false positives
        mean_projection[mean_projection < mean_projection.mean()] = mean_projection.mean()

        # Create a detector surface mask and then pad it.
        mask = hist > threshold*mean_projection
        mask_indices_minimal = np.column_stack(np.nonzero(mask))
        mask_indices = []
        for index in range(mask_indices_minimal.shape[0]):
            mask_x = mask_indices_minimal[index, 1]
            mask_y = mask_indices_minimal[index, 0]
            for pad_x in range(-pad + mask_x, pad + mask_x + 1):
                for pad_y in range(-pad + mask_y, pad + mask_y + 1):
                    if 0 <= pad_x < n_bins:
                        if 0 <= pad_y < n_bins:
                            mask_indices.append([pad_y, pad_x])
        mask_indices = np.row_stack((mask_indices))
        mask[mask_indices[:, 0], mask_indices[:, 1]] = True



        # Mask for the reflections that fit within the detector mask
        # self.refl_mask is created in the __init__ method
        # Remaking it resets the mask
        self.refl_mask = np.ones(self.q2_obs.size, dtype=bool)
        for index in range(mask_indices.shape[0]):
            indices = np.logical_and(
                refl_x == mask_indices[index, 0],
                refl_y == mask_indices[index, 1]
                )
            self.refl_mask[indices] = False
        
        fig, axes = plt.subplots(1, 1, figsize=(8, 8))
        axes.scatter(
            self.s1[:, 0], self.s1[:, 1],
            s=0.01, color=[0, 0, 0], alpha=0.1
            )
        axes.imshow(
            mask, cmap='Reds', alpha=0.4,
            origin='lower', extent=(centers_x[0], centers_x[-1], centers_y[0], centers_y[-1])
            )
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_title('Scatter Plot Of Reflection Coordinates\nMask in red\nThere is a bug and the mask and reflections are offset')
        fig.tight_layout()
        plt.show()

        # Make sure the masked reflections actually line up with the mask
        fig, axes = plt.subplots(1, 1, figsize=(8, 8))
        axes.scatter(
            self.s1[self.refl_mask, 0], self.s1[self.refl_mask, 1],
            s=0.01, color=[0, 0, 0], alpha=0.1
            )
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_title('Scatter Plot Of Masked Reflection Coordinates')
        fig.tight_layout()
        plt.show()
        
        """
        # Diagnostic plots

        # Make sure the masked reflections actually line up with the mask
        fig, axes = plt.subplots(1, 1, figsize=(8, 8))
        axes.scatter(
            self.s1[self.refl_mask, 0], self.s1[self.refl_mask, 1],
            s=0.01, color=[0, 0, 0], alpha=0.1
            )
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_title('Scatter Plot Of Reflection Coordinates\nCoordindates Masked')
        fig.tight_layout()
        plt.show()

        # Azimuthal mean
        fig, axes = plt.subplots(1, 1, figsize=(7, 3))
        axes.plot(azimuthal_mean)
        plt.show()

        # 2D image of the s1_lab
        fig, axes = plt.subplots(1, 1, figsize=(10, 10))
        axes.imshow(s1_lab_mag_centers, origin='lower')
        fig.tight_layout()
        plt.show()

        # 2D Histogram of the reflections
        vmax = np.sort(hist.ravel())[int(0.999*hist.size)]
        fig, axes = plt.subplots(1, 1, figsize=(6, 6))
        axes.imshow(hist, cmap='gray_r', vmin=0, vmax=vmax, origin='lower')
        fig.tight_layout()
        plt.show()

        # Projection of the azimuthal mean onto the detector surface.
        fig, axes = plt.subplots(1, 1, figsize=(6, 6))
        axes.imshow(mean_projection, cmap='gray_r', vmin=0, vmax=vmax, origin='lower')
        fig.tight_layout()
        plt.show()
        """

    def make_histogram(self, n_bins=1000, d_min=60, d_max=3.5, q2_min=None, q2_max=None, mask=True):
        if q2_min is None:
            self.d_min = d_min
            self.q2_min = 1 / self.d_min**2
        else:
            self.q2_min = q2_min
            self.d_min = 1/np.sqrt(q2_min)
        if q2_max is None:
            self.d_max = d_max
            self.q2_max = 1 / self.d_max**2
        else:
            self.q2_max = q2_max
            self.d_max = 1/np.sqrt(q2_max)
        self.q2_bins = np.linspace(self.q2_min, self.q2_max, n_bins + 11)
        self.q2_centers = (self.q2_bins[1:] + self.q2_bins[:-1]) / 2
        if mask:
            self.q2_hist, _ = np.histogram(self.q2_obs[self.refl_mask], bins=self.q2_bins)
        else:
            self.q2_hist, _ = np.histogram(self.q2_obs, bins=self.q2_bins)

    def pick_peaks(self, exclude_list=[], exclude_max=20, add_peaks=[], shift={}, prominence=30, plot_kapton_peaks=False, yscale=None):
        found_peak_indices = scipy.signal.find_peaks(self.q2_hist, prominence=prominence)
        found_peaks = self.q2_centers[found_peak_indices[0]]
        found_peaks = np.delete(found_peaks[:exclude_max], exclude_list)
        peaks = np.sort(np.concatenate((found_peaks, add_peaks)))
    
        fig, axes = plt.subplots(1, 1, figsize=(30, 6))
        axes.plot(self.q2_centers, self.q2_hist, label='Histogram')
        for p_index, p in enumerate(peaks):
            if p_index in shift.keys():
                peaks[p_index] += shift[p_index]
            if p in add_peaks:
                color = [0.8, 0, 0]
            else:
                color = [0, 0, 0]
            axes.plot(
                [p, p], [0, self.q2_hist.max()],
                linestyle='dotted', linewidth=1, color=color
                )
            axes.annotate(p_index, xy=(p-0.001, (1-p_index/peaks.size) * self.q2_hist.max()))
        if plot_kapton_peaks:
            kapton_peaks = [15.25, 7.625, 5.083333333, 3.8125, 3.05]
            for p in kapton_peaks:
                if p > self.d_max:
                    axes.plot([1/p**2, 1/p**2], [0, self.q2_hist.max()], linestyle='dotted', linewidth=2, color=[0, 0.7, 0], label='Kapton Peaks')
        axes.set_xlabel('q2 (1/$\mathrm{\AA}^2$')
        if yscale == 'log':
            axes.set_yscale('log')
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_to_directory, f'{self.tag}_peaks.png'))
        plt.show()
        self.q2_peaks = peaks
        np.save(
            os.path.join(self.save_to_directory, f'{self.tag}_peaks.npy'),
            self.q2_peaks
            )
        print(repr(self.q2_peaks))
        print(repr(1/np.sqrt(self.q2_peaks)))

    def fit_peaks(self, n_max, ind_peak_indices, fit_shift=True, exclude_fit_shift=[]):
        def get_I_calc(amplitudes, q2_centers, broadening_params, q2, jac=False):
            breadths = (broadening_params[0] + broadening_params[1]*q2_centers)[:, np.newaxis]
            prefactor = 1 / np.sqrt(2*np.pi * breadths**2)
            exponential = np.exp(-1/2 * ((q2_centers[:, np.newaxis] - q2[np.newaxis]) / breadths)**2)
            I_calc = np.sum(amplitudes[:, np.newaxis] * prefactor * exponential, axis=0)
            if jac:
                dI_calc_damplitudes = prefactor * exponential

                dexponential_dq2_centers = -exponential * (q2_centers[:, np.newaxis] - q2[np.newaxis]) / breadths**2
                dI_calc_dq2_centers = amplitudes[:, np.newaxis] * prefactor * dexponential_dq2_centers
                return I_calc, dI_calc_damplitudes, dI_calc_dq2_centers
            else:
                return I_calc
        def fit_loss(x, amplitudes, q2_centers, mask, broadening_params, I_obs, q2, mode):
            amplitudes_all = np.zeros(mask.size)
            q2_centers_all = np.zeros(mask.size)
            if mode == 'amplitudes':
                amplitudes_all[mask] = x
                amplitudes_all[~mask] = amplitudes[~mask]
                q2_centers_all = q2_centers
            elif mode == 'amplitudes_centers':
                amplitudes_all[mask] = x[:mask.sum()]
                amplitudes_all[~mask] = amplitudes[~mask]
                q2_centers_all[mask] = x[mask.sum():]
                q2_centers_all[~mask] = q2_centers[~mask]
            I_calc = get_I_calc(amplitudes_all, q2_centers_all, broadening_params, q2, False)
            L = I_calc - I_obs
            return L
        def fit_jac(x, amplitudes, q2_centers, mask, broadening_params, I_obs, q2, mode):
            amplitudes_all = np.zeros(mask.size)
            q2_centers_all = np.zeros(mask.size)
            if mode == 'amplitudes':
                amplitudes_all[mask] = x
                amplitudes_all[~mask] = amplitudes[~mask]
                q2_centers_all = q2_centers
            elif mode == 'amplitudes_centers':
                amplitudes_all[mask] = x[:mask.sum()]
                amplitudes_all[~mask] = amplitudes[~mask]
                q2_centers_all[mask] = x[mask.sum():]
                q2_centers_all[~mask] = q2_centers[~mask]
            I_calc, dI_calc_damplitudes, dI_calc_dq2_centers = get_I_calc(amplitudes_all, q2_centers_all, broadening_params, q2, True)
            if mode == 'amplitudes':
                jac = dI_calc_damplitudes[mask].T
            elif mode == 'amplitudes_centers':
                jac = np.concatenate((dI_calc_damplitudes[mask], dI_calc_dq2_centers[mask]), axis=0).T
            return jac
        def basic_gaussian(p, x):
            return p[0] / np.sqrt(2*np.pi*p[1]**2) * np.exp(-1/2 * ((x - p[2]) / p[1])**2)
        def basic_gaussian_loss(p, x, y):
            return basic_gaussian(p, x) - y

        # Start by fitting individual peaks
        # Peaks fit individually will be fixed in the next stages when peaks during the profile fit.
        ind_amplitudes = np.zeros(len(ind_peak_indices))
        ind_breadths = np.zeros(len(ind_peak_indices))
        ind_q2_centers = np.zeros(len(ind_peak_indices))

        delta = int(0.003 / (self.q2_centers[1] - self.q2_centers[0]))
        for index, peak_index in enumerate(ind_peak_indices):
            loc = np.searchsorted(self.q2_centers, self.q2_peaks[peak_index])
            low = max(0, loc - delta)
            high = min(self.q2_centers.size, loc + delta)
            sigma = 0.0001
            amplitude = (self.q2_hist[low: high].max() - self.q2_hist[low: high].min()) * np.sqrt(2*np.pi)*sigma
            results = scipy.optimize.least_squares(
                basic_gaussian_loss,
                x0=(amplitude, sigma, self.q2_peaks[peak_index]),
                args=(self.q2_centers[low: high], self.q2_hist[low: high])
                )
            ind_amplitudes[index] = np.abs(results.x[0])
            ind_breadths[index] = np.abs(results.x[1])
            ind_q2_centers[index] = np.abs(results.x[2])

        broadening_params_polyfit = np.polyfit(x=ind_q2_centers, y=ind_breadths, deg=1)
        self.broadening_params = np.array([broadening_params_polyfit[1], broadening_params_polyfit[0]])
        self.q2_breadths = np.polyval(broadening_params_polyfit, self.q2_peaks)

        mask = np.ones(n_max, dtype=bool)
        amplitudes = np.zeros(n_max)
        q2_centers = self.q2_peaks[:n_max].copy()
        for index, peak_index in enumerate(ind_peak_indices):
            if peak_index < n_max:
                mask[peak_index] = False
                amplitudes[peak_index] = ind_amplitudes[index]
                q2_centers[peak_index] = ind_q2_centers[index]

        # Fit breadths and amplitudes
        max_index = np.searchsorted(self.q2_centers, self.q2_peaks[n_max]) + 20
        results = scipy.optimize.least_squares(
            fit_loss,
            x0=amplitudes[mask],
            jac=fit_jac,
            args=(
                amplitudes,
                q2_centers,
                mask,
                self.broadening_params,
                self.q2_hist[:max_index],
                self.q2_centers[:max_index],
                'amplitudes'
                ),
            method='lm',
            )
        amplitudes[mask] = results.x
        print(results)
        if fit_shift:
            # Fit breadths, amplitudes, and shift
            x0 = np.concatenate((amplitudes[mask], q2_centers[mask]))
            print(x0.shape, x0) 
            results = scipy.optimize.least_squares(
                fit_loss,
                x0=x0,
                jac=fit_jac,
                args=(
                    amplitudes,
                    q2_centers,
                    mask,
                    self.broadening_params,
                    self.q2_hist[:max_index],
                    self.q2_centers[:max_index],
                    'amplitudes_centers'
                    ),
                method='lm'
                )
            print(results)
            amplitudes[mask] = results.x[:mask.sum()]
            q2_centers[mask] = results.x[mask.sum():]
            q2_peaks_original = self.q2_peaks[:n_max].copy()
            for peak_index in range(self.q2_peaks[:n_max].size):
                if not peak_index in exclude_fit_shift:
                    self.q2_peaks[peak_index] = q2_centers[peak_index]

        I_calc = get_I_calc(amplitudes, q2_centers, self.broadening_params, self.q2_centers[:max_index])
        fig, axes = plt.subplots(1, 1, figsize=(30,  8), sharex=True)
        axes.plot(self.q2_centers[:max_index], self.q2_hist[:max_index])
        axes.plot(self.q2_centers[:max_index], I_calc)
        ylim = axes.get_ylim()
        for peak_index, p in enumerate(self.q2_peaks[:n_max]):
            if p in ind_peak_indices:
                color = [0.8, 0, 0]
            else:
                color = [0, 0, 0]
            axes.plot([p, p], ylim, color=color, linestyle='dotted')
        if fit_shift:
            for i in range(n_max):
                shift = self.q2_peaks[i] - q2_peaks_original[i]
                axes.annotate(
                    f'{shift:0.5f}',
                    xy=(self.q2_peaks[i], 0.9 * ylim[1]),
                    rotation=90
                    )
        fig, axes = plt.subplots(1, 1, figsize=(6, 3))
        axes.plot(self.q2_peaks[ind_peak_indices], ind_breadths, marker='.')
        axes.plot(self.q2_peaks[ind_peak_indices], np.polyval(broadening_params_polyfit, self.q2_peaks[ind_peak_indices]))
        plt.show()

    def optimize_beam_center(self, primary_peak_indices, mask=True):
        def get_q2_spacing(s1, s0):
            wavelength = 1 / np.linalg.norm(s0)
            dot_product = np.matmul(s1, s0)
            magnitudes = np.linalg.norm(s1) * np.linalg.norm(s0)
            theta2 = np.arccos(dot_product / magnitudes)
            return ((2 * np.sin(theta2 / 2)) / wavelength)**2
    
        def functional(delta, s1_list, s0_list):
            L = 0
            for peak_index in range(len(s1_list)):
                s1 = s1_list[peak_index]
                s0 = s0_list[peak_index]
                q2_calc = np.zeros(s1.shape[0])
                for i in range(s1.shape[0]):
                    s1_delta = s1[i].copy()
                    s1_delta[:2] += delta
                    q2_calc[i] = get_q2_spacing(s1_delta, s0[i])
                L += q2_calc.std()
            return L

        s1 = []
        s0 = []
        if mask:
            q2_obs_masked = self.q2_obs[self.refl_mask]
            s1_masked = self.s1[self.refl_mask]
            expt_indices_masked = self.expt_indices[self.refl_mask]
        else:
            q2_obs_masked = self.q2_obs
            s1_masked = self.s1
            expt_indices_masked = self.expt_indices

        for peak_index in primary_peak_indices:
            differences = np.abs(q2_obs_masked - self.q2_peaks[peak_index])
            indices = differences < 3*self.q2_breadths[peak_index]
            s1.append(s1_masked[indices])
            s0.append(self.s0[expt_indices_masked[indices]])
    
        initial_simplex = np.array([
            [0.05, 0.025],
            [0.001, -0.01],
            [-0.025, -0.05],
            ])
        print(functional(np.zeros(2), s1, s0))
        
        results = scipy.optimize.minimize(
            fun=functional,
            x0=[0, 0],
            args=(s1, s0),
            method='Nelder-Mead',
            options={'initial_simplex': initial_simplex}
            )
        print(results)
        self.beam_delta = results.x[:2]
        self.s1[:, :2] += self.beam_delta
        start = 0
        for expt_index, refl_counts in enumerate(self.refl_counts):
            self.q2_obs[start: start + refl_counts] = self._get_q2_spacing(
                self.s1[start: start + refl_counts], self.s0[expt_index]
                )
            start += refl_counts

    def bump_detector_distance(self, bump):
        self.s1[:, 2] += bump
        q2 = []
        start = 0
        for expt_index, refl_counts in enumerate(self.refl_counts):
            q2_obs = self.q2_obs[start: start + refl_counts]
            s1 = self.s1[start: start + refl_counts]
            s0 = self.s0[expt_index]
            wavelength = 1 / np.linalg.norm(s0)
            s1_normed = s1 / (wavelength * np.linalg.norm(s1, axis=1)[:, np.newaxis])
            # q2_lattice is the magnitude**2 of the scattering vector
            q2.append(self._get_q2_spacing(s1_normed, s0))
        self.q2_obs = np.concatenate(q2)

    def filter_peaks(self, n_peaks=20, max_difference=None, delta=None, max_refl_counts=None, threshold=0.50, mask=True):
        # assign peaks and get distances
        # The :n_peaks+1 appears unnecessary, but is important
        # If the peak gets assigned to the n_peaks index, it is probably out of the range of diffraction
        # This catches those cases for them to be ignored
        differences_all = np.abs(self.q2_obs[:, np.newaxis] - self.q2_peaks[np.newaxis, :n_peaks + 1])
        assignment = np.argmin(differences_all, axis=1)
        differences = np.take_along_axis(differences_all, assignment[:, np.newaxis], axis=1)[:, 0]

        joint_occurances = np.zeros((n_peaks, n_peaks))
        ind_occurances = np.zeros(n_peaks)
        start = 0
        n_experiments = 0
        for expt_index, refl_counts in enumerate(self.refl_counts):
            if mask:
                expt_refl_mask = self.refl_mask[start: start + refl_counts]
                assignment_expt = assignment[start: start + refl_counts][expt_refl_mask]
                differences_expt = differences[start: start + refl_counts][expt_refl_mask]
                masked_refl_counts = np.sum(expt_refl_mask)
            else:
                assignment_expt = assignment[start: start + refl_counts]
                differences_expt = differences[start: start + refl_counts]
                masked_refl_counts = refl_counts
            if masked_refl_counts > 0:
                if max_refl_counts is None or masked_refl_counts < max_refl_counts:
                    if not max_difference is None:
                        assignment_expt = assignment_expt[differences_expt < max_difference]
                    elif not delta is None:
                        peak_breadths = np.take(self.q2_breadths, assignment_expt)
                        tolerance = delta * peak_breadths
                        assignment_expt = assignment_expt[differences_expt < tolerance]
                        
                    unique_assignments = np.sort(np.unique(assignment_expt))
                    if unique_assignments.size > 0 and unique_assignments[-1] == n_peaks:
                        unique_assignments = unique_assignments[:-1]
                    #print(unique_assignments)
                    if unique_assignments.size > 0:
                        n_experiments += 1
                        for peak_index_0 in range(n_peaks):
                            if peak_index_0 in unique_assignments:
                                ind_occurances[peak_index_0] += 1
                                for peak_index_1 in range(n_peaks):
                                    if peak_index_1 in unique_assignments:
                                        joint_occurances[peak_index_0, peak_index_1] += 1
                    #print(ind_occurances)
                    #print(joint_occurances)
                    #print()
            start += refl_counts

        #joint_prob = joint_occurances / n_experiments
        #ind_prob = ind_occurances / n_experiments
        #separated_prob = 1/2*(ind_occurances[np.newaxis] + ind_occurances[:, np.newaxis]) / n_experiments
        #ratio = joint_prob/separated_prob

        ratio = joint_occurances / (ind_occurances[np.newaxis] * ind_occurances[:, np.newaxis] / n_experiments)

        ratio[np.arange(n_peaks), np.arange(n_peaks)] = np.nan
        paired = ratio > threshold

        print('Paired Peaks')
        for peak_index_0 in range(n_peaks):
            for peak_index_1 in range(peak_index_0, n_peaks):
                if paired[peak_index_0, peak_index_1]:
                    print(peak_index_0, peak_index_1)

        fig, axes = plt.subplots(1, 1, figsize=(10, 3))
        axes.bar(np.arange(n_peaks), ind_prob, width=1)
        axes.set_xlabel('Peak index')
        axes.set_ylabel('Occurance Probability')
        plt.show()

        cmap = 'binary'
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        
        separated_disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=separated_prob)
        separated_disp.plot(include_values=False, ax=axes[0, 0], cmap=cmap)
        
        joint_disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=joint_prob)
        joint_disp.plot(include_values=False, ax=axes[0, 1], cmap=cmap)

        ratio_disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=ratio)
        ratio_disp.plot(include_values=False, ax=axes[1, 0], cmap=cmap)
        
        paired_disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=paired)
        paired_disp.plot(include_values=False, ax=axes[1, 1], cmap=cmap)

        axes[0, 0].set_title('Separated Probability')
        axes[0, 1].set_title('Joint Probability')
        axes[1, 0].set_title('Joint/Separated Probability')
        axes[1, 1].set_title(f'Joint/Separated Probability > {threshold}')

        fig.tight_layout()
        plt.show()

    def output_json(self, note=None, extra_file_name=None):
        output = {
            'primary_peaks': self.q2_peaks,
            'secondary_peaks': self.q2_peaks_secondary,
            'primary_hist': np.column_stack((self.q2_centers, self.q2_hist)),
            'secondary_hist': np.column_stack((self.q2_diff_centers, self.q2_diff_hist)),
            'triplet_obs': self.triplets_obs,
            'broadening_params': self.broadening_params,
            'error': self.error,
            'note': note,
            }
        if extra_file_name is None:
            file_name = os.path.join(self.save_to_directory, f'{self.tag}_info.json')
        else:
            file_name = os.path.join(self.save_to_directory, f'{self.tag}_info_{extra_file_name}.json')
        pd.Series(output).to_json(file_name)

    def create_secondary_peaks(self, q2_max=None, max_difference=None, max_refl_counts=None, min_separation=None, n_bins=2000, mask=True):
        start = 0
        q2_diff = []
        min_separation_obs = []
        for expt_index, refl_counts in enumerate(self.refl_counts):
            if mask:
                expt_refl_mask = self.refl_mask[start: start + refl_counts]
                q2_obs = self.q2_obs[start: start + refl_counts][expt_refl_mask]
                s1 = self.s1[start: start + refl_counts][expt_refl_mask]
                masked_refl_counts = np.sum(expt_refl_mask)
            else:
                q2_obs = self.q2_obs[start: start + refl_counts]
                s1 = self.s1[start: start + refl_counts]
                masked_refl_counts = refl_counts
            if masked_refl_counts > 0:
                if max_refl_counts is None or masked_refl_counts < max_refl_counts:
                    s0 = self.s0[expt_index]
                    wavelength = 1 / np.linalg.norm(s0)
                    if not max_difference is None:
                        min_error = np.min(
                            np.abs(q2_obs[:, np.newaxis] - self.q2_peaks[np.newaxis]),
                            axis=1
                            )
                        indices = min_error < max_difference
                        q2_obs = q2_obs[indices]
                        s1 = s1[indices]
                    if not q2_max is None:
                        indices = q2_obs < q2_max
                        q2_obs = q2_obs[indices]
                        s1 = s1[indices]

                    if q2_obs.size > 1:
                        s1_normed = s1 / (wavelength * np.linalg.norm(s1, axis=1)[:, np.newaxis])
                        q2_diff_all = np.linalg.norm(
                            s1_normed[np.newaxis, :, :] - s1_normed[:, np.newaxis, :],
                            axis=2
                            )**2
                        indices = np.triu_indices(s1.shape[0], k=1)
                        q2_diff_lattice = q2_diff_all[indices[0], indices[1]]
                        min_separation_obs.append(np.min(q2_diff_lattice))
                        if min_separation is None or np.min(q2_diff_lattice) > min_separation:
                            q2_diff.append(q2_diff_lattice)
            start += refl_counts
        self.q2_diff = np.concatenate(q2_diff)
        min_separation_obs = np.array(min_separation_obs)

        self.q2_diff_bins = np.linspace(0.00000001, self.q2_max, n_bins + 1)
        self.q2_diff_centers = (self.q2_diff_bins[1:] + self.q2_diff_bins[:-1]) / 2
        self.q2_diff_hist, _ = np.histogram(self.q2_diff, bins=self.q2_diff_bins)

        fig, axes = plt.subplots(1, 1, figsize=(40, 5))
        axes.plot(self.q2_diff_centers, self.q2_diff_hist)
        np.save(
            os.path.join(self.save_to_directory, f'{self.tag}_q2_diff_hist.npy'),
            np.column_stack((self.q2_diff_centers, self.q2_diff_hist))
            )

        fig, axes = plt.subplots(1, 3, figsize=(7, 3))

        indices = self.q2_obs < self.q2_peaks.max()
        min_error = np.min(np.abs(self.q2_obs[indices, np.newaxis] - self.q2_peaks[np.newaxis]), axis=1)
        axes[0].hist(min_error, bins=100, log=True)
        if not max_difference is None:
            ylim = axes[0].get_ylim()
            axes[0].plot([max_difference, max_difference], ylim, color=[0, 0, 0])
            axes[0].set_ylim(ylim)
        axes[0].set_title('Primary Peaks Distance\nfrom a picked peak (STD)')

        axes[1].bar(
            np.arange(self.refl_counts.max() + 1), np.bincount(self.refl_counts),
            width=1
            )
        if not max_refl_counts is None:
            ylim = axes[1].get_ylim()
            axes[1].plot([max_refl_counts, max_refl_counts], ylim, color=[0, 0, 0])
            axes[1].set_ylim(ylim)
        axes[1].set_xscale('log')
        axes[1].set_title('Counts per experiment')

        bins = np.linspace(0, 0.005, 1001)
        centers = (bins[1:] + bins[:-1]) / 2
        hist, _ = np.histogram(min_separation_obs, bins=bins)
        axes[2].bar(centers, hist, width=(bins[1] - bins[0]))
        if not min_separation is None:
            ylim = axes[2].get_ylim()
            axes[2].plot([min_separation, min_separation], ylim, color=[0, 0, 0])
            axes[2].set_ylim(ylim)
        axes[2].set_xscale('log')
        axes[2].set_title('Closest peaks per experiment')
        fig.tight_layout()
        plt.show()
        
    def pick_secondary_peaks(self, include_list=[], prominence=30, yscale=None):
        indices = scipy.signal.find_peaks(self.q2_diff_hist, prominence=prominence)
        self.q2_peaks_secondary = []

        fig, axes = plt.subplots(2, 1, figsize=(45, 6), sharex=True)
        axes[0].plot(self.q2_centers, self.q2_hist)
        axes[1].plot(self.q2_diff_centers, self.q2_diff_hist)

        ylim0 = axes[0].get_ylim()
        ylim1 = axes[1].get_ylim()
        for p_index, p in enumerate(self.q2_peaks):
            if p_index == 0:
                label = 'Primary Picked'
            else:
                label = None
            axes[0].plot([p, p], ylim0, linestyle='dotted', linewidth=1.5, color=[0, 0, 0])
            axes[1].plot([p, p], ylim1, linestyle='dotted', linewidth=1.5, color=[0, 0, 0], label=label)

        for p_index, p in enumerate(self.q2_diff_centers[indices[0]]):
            if p_index == 0:
                label = 'Secondary Found'
            else:
                label = None
            if p_index in include_list:
                self.q2_peaks_secondary.append(p)
            axes[1].plot([p, p], ylim1, linestyle='dashed', linewidth=1.5, color=[0.8, 0, 0], label=label)
            axes[1].annotate(p_index, xy=(p, 0.9*ylim1[1]))
        axes[0].set_ylim(ylim0)
        axes[1].set_ylim(ylim1)
        axes[0].set_ylabel('Primary Positions')
        axes[1].set_ylabel('Secondary Positions')
        axes[1].set_xlabel('1 / d_spacing ($\mathrm{\AA}$)')
        if yscale == 'log':
            axes[0].set_yscale('log')
            axes[1].set_yscale('log')
        axes[1].legend(loc='upper left', frameon=False)

        fig.tight_layout()
        fig.savefig(os.path.join(self.save_to_directory, f'{self.tag}_secondary_peaks.png'))
        plt.show()
        np.save(
            os.path.join(self.save_to_directory, f'{self.tag}_secondary_peaks.npy'),
            1/np.sqrt(np.array(self.q2_peaks_secondary))
            )
        print(repr(1/np.sqrt(self.q2_peaks_secondary)))

    def plot_known_unit_cell(self, q2_max=0.5, unit_cell=None, space_group=None):
        if unit_cell is None:
            unit_cell = uctbx.unit_cell(parameters=self.known_unit_cell)
        else:
            unit_cell = uctbx.unit_cell(parameters=unit_cell)
        if space_group is None:
            sym = symmetry(unit_cell=unit_cell, space_group=self.known_space_group)
        else:
            sym = symmetry(unit_cell=unit_cell, space_group=space_group)

        hkl_list = cctbx.miller.build_set(sym, False, d_min=1/np.sqrt(q2_max))
        dspacings = unit_cell.d(hkl_list.indices()).as_numpy_array()
        q2_known = 1 / dspacings**2
   
        fig, axes = plt.subplots(1, 1, figsize=(40, 4), sharex=True)
        axes.plot(self.q2_centers, self.q2_hist)
        ylim0 = axes.get_ylim()
        for p in q2_known:
            axes.plot([p, p], ylim0, color=[0.8, 0, 0], linestyle='dotted', linewidth=2)
        for p_index, p in enumerate(self.q2_peaks):
            axes.plot([p, p], [ylim0[0], 0.75*ylim0[1]], color=[0, 0, 0], linestyle='dotted', linewidth=2)
        axes.set_ylim(ylim0)
        fig.tight_layout()
        plt.show()

    def make_triplets(self, triplet_peak_indices, delta=1, max_difference=False, min_separation=None, max_refl_counts=None, mask=True):
        if max_refl_counts is None:
            max_refl_counts = np.inf
        start = 0
        triplet_keys = []
        triplet_peaks = self.q2_peaks[triplet_peak_indices]
        triplet_breadths = np.abs(self.q2_breadths[triplet_peak_indices])
        for p0 in range(triplet_peaks.size):
            for p1 in range(p0, triplet_peaks.size):
                triplet_keys.append((triplet_peak_indices[p0], triplet_peak_indices[p1]))
        print(triplet_keys)
        self.triplets = dict.fromkeys(triplet_keys)
        for key in triplet_keys:
            self.triplets[key] = []
        for expt_index, refl_counts in enumerate(self.refl_counts):
            q2_obs, s1, start = slice_refls(self.q2_obs, self.s1, start, refl_counts, self.refl_mask, mask)
            # If there are too many refls on a frame, it might have multiple lattices.
            if q2_obs.size >= 2 and q2_obs.size < max_refl_counts:
                q = get_scattering_vectors(self.s0[expt_index], s1)
                
                # This removes peaks that are larger than the 1D peak list
                indices = q2_obs < (self.q2_peaks[-1] + delta*self.q2_breadths[-1])
                q2_obs = q2_obs[indices]
                q = q[indices]

                # Only consider peaks close to a peak in the picked peak list.
                if max_difference:
                    min_error = np.min(
                        np.abs(q2_obs[:, np.newaxis] - self.q2_peaks[np.newaxis]) / self.q2_breadths,
                        axis=1
                        )
                    indices = min_error < delta
                    q2_obs = q2_obs[indices]
                    q = q[indices]

                if q2_obs.size > 1:
                    q2_diff_lattice = calc_pairwise_diff(q, metric='euclidean')**2
                    angular_diff_lattice = calc_pairwise_diff(q, q_ref=self.s0[expt_index], metric='angular')
                    indices = np.triu_indices(q.shape[0], k=1)
                    q20_obs = q2_obs[indices[0]]
                    q21_obs = q2_obs[indices[1]]
                    # If there are peaks that are very close, it might be a multiple lattice.
                    if min_separation_check(q, min_separation):
                        q20_triplet_index = np.argmin(
                            np.abs(q20_obs[:, np.newaxis] - triplet_peaks[np.newaxis]) / triplet_breadths[np.newaxis],
                            axis=1
                            )
                        q21_triplet_index = np.argmin(
                            np.abs(q21_obs[:, np.newaxis] - triplet_peaks[np.newaxis]) / triplet_breadths[np.newaxis],
                            axis=1
                            )
                        for pair_index in range(q2_diff_lattice.size):
                            p0 = q20_triplet_index[pair_index]
                            p1 = q21_triplet_index[pair_index]
                            key = (
                                triplet_peak_indices[p0],
                                triplet_peak_indices[p1]
                                ) 
                            check0 = np.logical_and(
                                q20_obs[pair_index] > triplet_peaks[p0] - delta*triplet_breadths[p0],
                                q20_obs[pair_index] < triplet_peaks[p0] + delta*triplet_breadths[p0],
                                )
                            check1 = np.logical_and(
                                q21_obs[pair_index] > triplet_peaks[p1] - delta*triplet_breadths[p1],
                                q21_obs[pair_index] < triplet_peaks[p1] + delta*triplet_breadths[p1],
                                )
                            check2 = np.invert(np.isnan(angular_diff_lattice[pair_index]))
                            check3 = np.abs(angular_diff_lattice[pair_index]) < 2.8
                            if check0 and check1 and check2 and check3:
                                if key[0] < key[1]:
                                    q20_triplet_peak = triplet_peaks[p0]
                                    q21_triplet_peak = triplet_peaks[p1]
                                else:
                                    key = (key[1], key[0])
                                    q20_triplet_peak = triplet_peaks[p1]
                                    q21_triplet_peak = triplet_peaks[p0]
                                self.triplets[key].append([
                                    q20_triplet_peak,
                                    q21_triplet_peak,
                                    q2_diff_lattice[pair_index],
                                    angular_diff_lattice[pair_index],
                                    ])
        for key in triplet_keys:
            if len(self.triplets[key]) > 0:
                self.triplets[key] = np.row_stack(self.triplets[key])

    def pick_triplets(self, prominence_factor=5, hkl=None, xnn=None, lattice_system=None, auto=False):
        triplet_keys = list(self.triplets.keys())
        triplets_obs = []
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for key_index, key in enumerate(triplet_keys):
            if len(self.triplets[key]) > 0:
                triplets_obs_pair = []
                fig, axes = plt.subplots(1, 1, figsize=(10, 3), sharex=False)
                bins_full = np.linspace(-np.pi, np.pi, 401)
                centers_full = (bins_full[1:] + bins_full[:-1]) / 2
                bins_half = np.linspace(0, np.pi/2, 101)
                centers_half = (bins_half[1:] + bins_half[:-1]) / 2
                
                q2_0 = self.triplets[key][0, 0]
                q2_1 = self.triplets[key][0, 1]
                q2_diff = self.triplets[key][:, 2]
                angles = self.triplets[key][:, 3]

                # Angles are between two points, A & B, on a common plane
                # Angles are between -180 to +180 where the sign comes from the dot product with
                #     the negative beam.
                # "Rotation about A" refer to a 180 degree rotation about the vector between 0 and A.
                # We can always find an angle between A & B such that 0 < angle < 90. This is
                #   because two points cannot fully define the orientation of the crystal.
                #   Inversions of B are valid - the rest of reciprocal space remains the same.
                #   Rotations about A are only valid if there is a mirror symmetry. Otherwise the
                #   rest of reciprocal space is changed.
                # If 0 < angle < 90: keep.
                #   angle_sym = angle
                # If 90 < angle < 180: Invert B, then rotate 180 about A.
                #   angle_sym = np.pi - angle
                # If -90 < angle < 0: Rotate 180 about A
                #   angle_sym = -angle
                # If -180 < angle < -90: Invert B
                #   angle_sym = np.pi + angle
                angles_sym = apply_simple_symmetry(angles)

                # Get the expected angle for peaks in the 1D primary peak list
                angles_peaks = law_of_cosines(q2_0, q2_1, self.q2_peaks)

                # Get the expected angle for peaks in the 1D secondary peak list
                angles_secondary_peaks = []
                if len(self.q2_peaks_secondary) > 0:
                    angles_secondary_peaks = law_of_cosines(q2_0, q2_1, self.q2_peaks_secondary)

                hist_half, _ = np.histogram(angles_sym, bins=bins_half)
                hist_full, _ = np.histogram(angles, bins=bins_full)
                axes.bar(centers_full, hist_full, width=bins_full[1] - bins_full[0], color=colors[0], label='Obs Diff.')
                axes.bar(centers_half, hist_half, width=bins_half[1] - bins_half[0], color=colors[3], alpha=0.5, label='Obs Diff. Sym.')
                ylim = axes.get_ylim()
                ylim = [ylim[0], ylim[1]*1.1]

                # scipy.signal.find_peaks does not find peaks at the first and last index.
                # Padding zeros at the start and end are attempts to pick them up.
                n_pad = 5

                diff_peak_indices, _ = scipy.signal.find_peaks(
                    np.concatenate((np.zeros(n_pad), hist_half, np.zeros(n_pad))),
                    prominence=prominence_factor*(np.std(hist_half) + 1)
                    )
                diff_peak_indices -= n_pad
                diff_peak_indices = diff_peak_indices[diff_peak_indices >= 0]
                diff_peak_indices = diff_peak_indices[diff_peak_indices <= hist_half.size]
                if diff_peak_indices.size > 0:
                    # This gets the peak positions in units of angle
                    diff_peaks = centers_half[diff_peak_indices]
                    weights = hist_half[diff_peak_indices]
                    for p_index, p in enumerate(diff_peaks):
                        selection = np.logical_and(
                            angles_sym > p - 0.02,
                            angles_sym < p + 0.02,
                            )
                        median_angle = np.nanmedian(angles_sym[selection])                        
                        median_difference = q2_0 + q2_1 - 2*np.sqrt(q2_0*q2_1)*np.cos(median_angle)
                        triplets_obs_pair.append([
                            key[0], key[1],
                            median_difference, median_angle,
                            weights[p_index]
                            ])
                        if p_index == 0:
                            label = 'Found Differences'
                        else:
                            label = None
                        axes.plot(
                            [median_angle, median_angle], ylim,
                            color=colors[1], alpha=0.75, label=label, linestyle='dotted'
                            )
                        axes.annotate(
                            f'{np.round(median_angle, decimals=3)}',
                            xy=[median_angle, ylim[1]*0.85],
                            )
                    axes.set_ylim(ylim)
                if hkl is None:
                    axes.set_ylabel(
                        str(key)
                        + f'\n{np.round(q2_0, decimals=5)} {np.round(q2_1, decimals=5)}'
                        )
                else:
                    from Utilities import get_hkl_matrix
                    axes.set_ylabel(
                        str(key)
                        + f'\n{np.round(q2_0, decimals=5)} {np.round(q2_1, decimals=5)}'
                        + f'\n{hkl[key[0]]}, {hkl[key[1]]}'
                        )
                    
                    mi_sym = [
                        np.array([1, 1, 1]),
                        np.array([1, 1, -1]),
                        np.array([1, -1, 1]),
                        np.array([-1, 1, 1]),
                        ]
                    permutations = [np.array([
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        ])]
                    if lattice_system == 'cubic':
                        permutations.append(np.array([
                            [0, 1, 0],
                            [1, 0, 0],
                            [0, 0, 1],
                            ]))
                        permutations.append(np.array([
                            [0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0],
                            ]))
                        permutations.append(np.array([
                            [1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0],
                            ]))
                    elif lattice_system in ['tetragonal', 'hexagonal']:
                        permutations.append(np.array([
                            [0, 1, 0],
                            [1, 0, 0],
                            [0, 0, 1],
                            ]))
                        permutations.append(np.array([
                            [0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0],
                            ]))
                        permutations.append(np.array([
                            [1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0],
                            ]))
                    make_label = True
                    for i in range(len(mi_sym)):
                        for j in range(len(mi_sym)):
                            for k0 in range(len(permutations)):
                                for k1 in range(len(permutations)):
                                    hkl0 = np.matmul(permutations[k0], mi_sym[i]*hkl[key[0]])
                                    hkl1 = np.matmul(permutations[k1], mi_sym[j]*hkl[key[1]])
                                    hkl_diff = hkl0 - hkl1
                                    hkl2_diff = get_hkl_matrix(hkl_diff[np.newaxis], lattice_system)
                                    q2_diff_calc = np.sum(xnn * hkl2_diff, axis=1)[0]
                                    angle_diff_calc = law_of_cosines(q2_0, q2_1, q2_diff_calc, clip=False)
                                    if make_label:
                                        label = 'Pred Diff'
                                    else:
                                        label = None
                                    axes.plot(
                                        [angle_diff_calc, angle_diff_calc], [ylim[0], 0.15*ylim[1]],
                                        color=[0, 0.8, 0], label=label
                                        )
                                    axes.annotate(
                                        np.round(angle_diff_calc, decimals=3),
                                        xy=[angle_diff_calc, ylim[1]*0.2],
                                        )
                                    make_label = False
                for p_index, peak_angle in enumerate(angles_peaks):
                    if not np.isnan(peak_angle):
                        if p_index == 0:
                            label = 'Obs Peaks'
                        else:
                            label = None
                        axes.plot(
                            [peak_angle, peak_angle], [ylim[0], 0.1*ylim[1]],
                            color=[0, 0, 0], label=label
                            )
                for p_index, peak_angle in enumerate(angles_secondary_peaks):
                    if not np.isnan(peak_angle):
                        if p_index == 0:
                            label = 'Secondary Peaks'
                        else:
                            label = None
                        axes.plot(
                            [peak_angle, peak_angle], [ylim[0], 0.1*ylim[1]],
                            color=[0, 0, 0], linestyle='dashed', label=label
                            )
                axes.set_ylim()
                axes.legend(frameon=False)
                fig.tight_layout()
                plt.show(block=False)
                for index in range(len(triplets_obs_pair)):
                    if auto:
                        print(f'Found triplet at angle {triplets_obs_pair[index][3]:0.4f}')
                        if triplets_obs_pair[index][3] < 0.05:
                            triplets_obs_pair[index][3] = 0
                        elif triplets_obs_pair[index][3] > np.pi/2 - 0.05:
                            triplets_obs_pair[index][3] = np.pi/2
                        triplets_obs.append(triplets_obs_pair[index])
                    else:
                        print(f'Found triplet at angle {triplets_obs_pair[index][3]:0.4f}')
                        accept = input(f'   Accept with y, specify angle with 0 and 90$^o$ with 90')
                        if accept == 'y':
                            triplets_obs.append(triplets_obs_pair[index])
                        elif accept in [0, 0.0, '0']:
                            triplets_obs_pair[index][3] = 0
                            triplets_obs.append(triplets_obs_pair[index])
                        elif accept in [90, 90.0, '90']:
                            triplets_obs_pair[index][3] = np.pi/2
                            triplets_obs.append(triplets_obs_pair[index])
                plt.close()
        self.triplets_obs = np.stack(triplets_obs)

    def make_settings(self, setting_peak_indices, delta=1, angle_tolerance=0.05, max_refl_counts=None, min_separation=None, mask=True):
        if max_refl_counts is None:
            max_refl_counts = np.inf
        setting_keys = []
        setting_q2 = self.q2_peaks[setting_peak_indices]
        setting_breadths = np.abs(self.q2_breadths[setting_peak_indices])
        for p0 in range(setting_q2.size - 2):
            for p1 in range(p0 + 1, setting_q2.size - 1):
                for p2 in range(p1 + 1, setting_q2.size):
                    check0 = np.any(np.all(self.triplets_obs[:, :2] == [p0, p1], axis=1))
                    check1 = np.any(np.all(self.triplets_obs[:, :2] == [p0, p2], axis=1))
                    check2 = np.any(np.all(self.triplets_obs[:, :2] == [p1, p2], axis=1))
                    if check0 and check1 and check2:
                        setting_keys.append((setting_peak_indices[p0], setting_peak_indices[p1], setting_peak_indices[p2]))
        
        angle_references = {}
        for index in range(self.triplets_obs.shape[0]):
            key = (int(self.triplets_obs[index, 0]), int(self.triplets_obs[index, 1]))
            if not key in angle_references.keys():
                angle_references[key] = [self.triplets_obs[index, 3]]
            else:
                angle_references[key].append(self.triplets_obs[index, 3])

        self.settings = dict.fromkeys(setting_keys)
        for key in setting_keys:
            self.settings[key] = [[] for _ in range(len(angle_references[key[:2]]))]

        start = 0
        occurances = 0
        for expt_index, refl_counts in enumerate(self.refl_counts):
            q2_obs, s1, start = slice_refls(self.q2_obs, self.s1, start, refl_counts, self.refl_mask, mask)
            # If there are too many refls on a frame, it might have multiple lattices.
            if q2_obs.size >= 3 and q2_obs.size < max_refl_counts:
                q = get_scattering_vectors(self.s0[expt_index], s1)
                if min_separation_check(q, min_separation):
                    # This assigns the closest peak to each reflection
                    #    setting_indices - the index along the input setting_peak_indices
                    #    peak_indices - the index along self.q2_obs
                    normalized_differences_from_1d = np.abs(q2_obs[:, np.newaxis] - setting_q2[np.newaxis]) / setting_breadths[np.newaxis]
                    setting_indices = np.argmin(normalized_differences_from_1d, axis=1)
                    peak_indices = np.take(setting_peak_indices, setting_indices)
        
                    # This removes peaks that are further from the reference peak than we care about
                    indices = np.min(normalized_differences_from_1d, axis=1) < delta
                    q = q[indices]
                    q2_obs = q2_obs[indices]
                    peak_indices = peak_indices[indices]

                    if q2_obs.size >= 3:
                        for key in setting_keys:
                            # This gets the indices of the reflections that match each peak
                            #     column i: reflections assigned to peak key_i
                            combined = np.column_stack((
                                peak_indices == key[0],
                                peak_indices == key[1],
                                peak_indices == key[2],
                                ))
                            # Each peak must have each reflection that belongs to it.
                            if np.all(np.any(combined, axis=0)):
                                # Select only the reflections that belong to the key.
                                indices = np.any(combined, axis=1)
                                q_key = q[indices]
                                q2_obs_key = q2_obs[indices]
                                peak_indices_key = peak_indices[indices]
        
                                p0_indices = np.argwhere(peak_indices_key == key[0])[0]
                                p1_indices = np.argwhere(peak_indices_key == key[1])[0]
                                p2_indices = np.argwhere(peak_indices_key == key[2])[0]
                                settings_expt = [[] for _ in range(len(self.settings[key]))]
                                for p0_index in p0_indices:
                                    for p1_index in p1_indices:
                                        difference = calc_pairwise_diff(
                                            np.row_stack((q_key[p0_index], q_key[p1_index])),
                                            q_ref=self.s0[expt_index],
                                            metric='angular'
                                            )
                                        angle_p0p1 = apply_simple_symmetry(calc_pairwise_diff(
                                            np.row_stack((q_key[p0_index], q_key[p1_index])),
                                            q_ref=self.s0[expt_index],
                                            metric='angular'
                                            ))[0]
                                        check0 = False
                                        for angle_ref_index, angle_ref in enumerate(angle_references[(key[0], key[1])]):
                                            if np.abs(angle_p0p1 - angle_ref) < angle_tolerance:
                                                check0 = True
                                                break
                                        for p2_index in p2_indices:
                                            angle_p0p2 = apply_simple_symmetry(calc_pairwise_diff(
                                                np.row_stack((q_key[p0_index], q_key[p2_index])),
                                                q_ref=self.s0[expt_index],
                                                metric='angular'
                                                ))[0]
                                            check1 = False
                                            for angle_ref in angle_references[(key[0], key[2])]:
                                                if np.abs(angle_p0p2 - angle_ref) < angle_tolerance:
                                                    check1 = True

                                            angle_p1p2 = apply_simple_symmetry(calc_pairwise_diff(
                                                np.row_stack((q_key[p1_index], q_key[p2_index])),
                                                q_ref=self.s0[expt_index],
                                                metric='angular'
                                                ))[0]
                                            check2 = False
                                            for angle_ref in angle_references[(key[1], key[2])]:
                                                if np.abs(angle_p1p2 - angle_ref) < angle_tolerance:
                                                    check2 = True
                                            if check0 and check1 and check2:
                                                occurances += 1
                                                settings_expt[angle_ref_index].append(np.row_stack((
                                                    q_key[p0_index],
                                                    q_key[p1_index],
                                                    q_key[p2_index],
                                                    )))
                                for settings_index in range(len(self.settings[key])):   
                                    if len(settings_expt[settings_index]) > 0:
                                        self.settings[key][settings_index].append(np.stack(settings_expt[settings_index], axis=0))
        print(f'Number of occurances: {occurances}')
        for key in setting_keys:
            for settings_index in range(len(self.settings[key])):   
                if len(self.settings[key][settings_index]) > 0:
                    self.settings[key][settings_index] = np.concatenate(self.settings[key][settings_index], axis=0)
        self.align_settings()

    def align_settings(self):
        """
        Align and standardize orientation of the settings
        
        Parameters:
        self.settings: dictionary
            Keys are (peak #, peak #, peak #)
            Values are lists of arrays, each array of shape (N, 3, 3) and represents a different
              angular pairing between peaks 
                axis 0: # of observations
                axis 1: Peak # in key
                axis 2: xyz
        
        Returns:
        aligned_settings: dictionary
            Same structure as input, with aligned vectors
        """
        # This gets the reference angles between pairs
        angle_references = {}
        for index in range(self.triplets_obs.shape[0]):
            key = (int(self.triplets_obs[index, 0]), int(self.triplets_obs[index, 1]))
            if not key in angle_references.keys():
                angle_references[key] = [self.triplets_obs[index, 3]]
            else:
                angle_references[key].append(self.triplets_obs[index, 3])

        self.aligned_settings = {}

        # Loop over three peak sets
        for key, settings in self.settings.items():
            aligned_settings_list = []

            # loop over different angular relations between three peaks
            for settings_common_ang_index, settings_common_ang in enumerate(settings):
                if len(settings_common_ang) > 0:
                    N = settings_common_ang.shape[0]
                    aligned = np.zeros_like(settings_common_ang)
                    angle_ref = angle_references[(key[0], key[1])][settings_common_ang_index]
                    for i in range(N):
                        q0, q1, q2 = settings_common_ang[i]
                        
                        # 1. Rotate q0 to x-axis and q1 to xz-plane
                        R1 = get_R_point_to_ref(q0, np.array([1, 0, 0]))

                        # Apply first rotation
                        q0_rot = R1 @ q0
                        q1_rot = R1 @ q1
                        q2_rot = R1 @ q2
                        
                        # Rotate around x-axis to get q1 into xz-plane
                        angle2 = np.arctan2(q1_rot[1], q1_rot[2])
                        R2 = np.array([[1, 0, 0],
                                      [0, np.cos(angle2), -np.sin(angle2)],
                                      [0, np.sin(angle2), np.cos(angle2)]])
                        
                        q0_aligned = R2 @ q0_rot
                        q1_aligned = R2 @ q1_rot
                        q2_aligned = R2 @ q2_rot
        
                        # 2. Handle angle between q0 and q1.
                        #    This should be between 0 and 90 degrees.
                        angle_01 = np.arctan2(q1_aligned[2], q1_aligned[0])
                        
                        # Rotation matrix about x-axis by pi
                        Rx_pi = np.array([[1, 0, 0],
                                         [0, -1, 0],
                                         [0, 0, -1]])
                        
                        # Apply rules based on angle
                        if angle_01 > np.pi/2:
                            q1_aligned = -q1_aligned
                            q0_aligned = Rx_pi @ q0_aligned
                            q1_aligned = Rx_pi @ q1_aligned
                            q2_aligned = Rx_pi @ q2_aligned
                        elif angle_01 < -np.pi/2:
                            q1_aligned = -q1_aligned
                        elif -np.pi/2 <= angle_01 < 0:
                            q0_aligned = Rx_pi @ q0_aligned
                            q1_aligned = Rx_pi @ q1_aligned
                            q2_aligned = Rx_pi @ q2_aligned
        
                        # 3. Handle q2 orientation
                        #   We can only invert q2 at this point. Invert it so it is in a consistent locations.
                        #   The inverted point will be recovered with a -q operation applied to the graph.
                        #   If not in the xy plane - then make the y-component of q2 in the positive y-direction
                        #   If in the xy plane, then make the x-component positive.
                        q2_aligned_y_frac = q2_aligned[1] / np.linalg.norm(q2_aligned)
                        if q2_aligned_y_frac < -0.05:
                            q2_aligned = -q2_aligned
                        elif np.abs(q2_aligned_y_frac) < 0.05:
                            if q2_aligned[0] < 0:
                                q2_aligned = -q2_aligned
                        # Store aligned vectors
                        aligned[i] = np.vstack([q0_aligned, q1_aligned, q2_aligned])
                    
                    aligned_settings_list.append(aligned)
                
            self.aligned_settings[key] = aligned_settings_list

    def plot_settings(self, eps=0.02, min_samples=3):
        """
        Plot the aligned settings from dictionary, separate figure for each key
        All arrays from same key plotted together with different markers and shades
        
        Parameters:
        aligned_settings: dictionary
            Keys are tuples (peak #, peak #, peak #)
            Values are lists of arrays, each array of shape (N, 3, 3).
                Each array is a different angular combination between peaks
        """
        # Different marker styles for different arrays
        markers = ['o', 's', '^', 'D']  # circle, square, triangle, diamond
        
        # Base colors for q0, q1, q2
        base_colors = {
            0: [1, 0, 0],  # red
            1: [0, 1, 0],  # green
            2: [0, 0, 1]   # blue
        }
        
        # Shade factors for different arrays (from 1.0 to 0.25)
        def get_shade(idx, num_arrays):
            return 1.0 - (idx * 0.4 / (num_arrays - 1)) if num_arrays > 1 else 1.0

        self.base_graphs = dict.fromkeys(self.aligned_settings.keys())
        # Each key value pair in self.aligned_settings is a different set of three peaks
        for key, aligned_setting in self.aligned_settings.items():
            self.base_graphs[key] = []
            num_arrays = len(aligned_setting)
            # For each set of three peaks there are multiple found angular relations
            # aligned_setting_common_ang has shape N x 3 x 3
            #   axis 0: number of observations
            #   axis 1: peak # key[0], peak # key[1], peak # key[2]
            #   axis 2: x, y, z
            for arr_idx, aligned_setting_common_ang in enumerate(aligned_setting):
                shade = get_shade(arr_idx, num_arrays)
                # Plot each set of vectors

                fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={'projection': '3d'})
                ax.scatter(0, 0, 0, color=[0, 0, 0], marker='X',  s=50)
                # vectors_set is an iteration over observations
                for vectors_set in aligned_setting_common_ang:
                    # i, v is an iteration over peaks in the key
                    for peak_index, q_peak in enumerate(vectors_set):
                        # Create shaded color
                        color = [c * shade for c in base_colors[peak_index]]
                        ax.scatter(q_peak[0], q_peak[1], q_peak[2], 
                                 color=color,
                                 marker=markers[arr_idx], 
                                 s=50,
                                 )

                # q_cluster should have shape:
                #    axis 0: N found combinations
                #    axis 1: 3 peaks
                #    axis 2: 3 for xyz
                q_cluster = [[] for _ in range(3)]
                N_found_combinations = np.inf
                for peak_index in range(3):
                    aligned_setting_common_ang_peak = aligned_setting_common_ang[:, peak_index, :]

                    cluster = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples)
                    cluster = cluster.fit(aligned_setting_common_ang_peak)
                    labels = np.unique(cluster.labels_)
                    labels = labels[labels != -1]
                    N_found_combinations = min(N_found_combinations, labels.size)
                    for label_index in labels:
                        q_cluster[peak_index].append(
                            aligned_setting_common_ang_peak[cluster.labels_ == label_index].mean(axis=0)
                            )
                        ax.scatter(
                            q_cluster[peak_index][-1][0], q_cluster[peak_index][-1][1], q_cluster[peak_index][-1][2],
                            color=[1, 0, 0], s=200, alpha=0.5
                            )
                
                if N_found_combinations > 0:
                    q_cluster_ = np.zeros((N_found_combinations, 3, 3))
                    q_cluster_[:, 0, :] = np.repeat(q_cluster[0][0][np.newaxis], axis=0, repeats=N_found_combinations)
                    q_cluster_[:, 1, :] = np.repeat(q_cluster[1][0][np.newaxis], axis=0, repeats=N_found_combinations)
                    for combination_index in range(N_found_combinations):
                        q_cluster_[combination_index, 2, :] = q_cluster[2][combination_index]
                    self.base_graphs[key].append(q_cluster_)

                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title(key)
                ax.set_box_aspect([1,1,1])
                
                # Set equal axis limits centered on origin
                max_range = np.max([np.max(np.abs(vectors)) for vectors in aligned_setting_common_ang])
                ax.set_xlim([-max_range, max_range])
                ax.set_ylim([-max_range, max_range])
                ax.set_zlim([-max_range, max_range])
                plt.show()

    def make_graphs(self, iterations=5, red_tolerance=0.05, comb_tolerance=0.01):
        """
        Parameters:
        tetrahedron: dictionary
            Keys: tuples of (peak #0, peak #1, peak #2)
            Values: list of combinations between peak #0 and peak #1,
                    each containing combinations with peak #2
        Tetrahedron is a dictionary with the following heirarchy
            1: Key value pair: (peak #0, peak #1, peak #2)
            2: List with different found combinations between peak #0 and peak #1
            3: For a given combination in heirarchy 2, the different found combinations between peak #2

        Returns:
        graphs
        scores: list of scores for each graph

        Algorithm:
            Goal is to build up reciprocal space graphs by aligning peaks.
            Graphs will have shape N_nodes x 4:
                axis 0: # of reciprocal space points
                axis 1: x, y, z, peak #

            0: Convert tetrahedron at heirarchy 2 into graphs and store separately
            1: For each graph, attempt to align any two peaks with common peak #.
               If both peaks are closer than tolerance, create a new graph.
            2: 
        """
        initial_graphs = []
        # Loop through each base tetrahedron configuration and create the initial graphs
        for key, combinations in self.base_graphs.items():
            if not combinations is None:
                peak0, peak1, peak2 = key
                for comb in combinations:
                    for comb_index in range(comb.shape[0]):
                        initial_graphs.append(np.row_stack([
                            np.array([*comb[comb_index][0], peak0]),
                            np.array([*comb[comb_index][1], peak1]),
                            np.array([*comb[comb_index][2], peak2]),
                            ]))
        initial_graphs = remove_redundant_arrays(initial_graphs, red_tolerance=red_tolerance)
        initial_graphs = remove_redundant_arrays_slow(initial_graphs, red_tolerance=red_tolerance)
        self.graphs = initial_graphs.copy()
        last_graphs = initial_graphs.copy()

        # Do a fixed number of graph building iterations
        # Three list of graphs:
        #   1: graphs - all graphs
        #   2: last_graphs - graphs built in the last iterations
        #   3: building_graphs - graphs being built in the current iteration
        print(f'Initial Graphs: {len(self.graphs)}')
        for iteration in range(iterations):
            building_graphs = []
            # Try to add each graph to the last iterations graphs
            for last_graph0 in tqdm.tqdm(last_graphs):
                for last_graph1 in last_graphs:
                    building_graphs += combine_graphs(
                        last_graph0,
                        last_graph1,
                        comb_tolerance=comb_tolerance,
                        red_tolerance=red_tolerance
                        )
            last_graphs = remove_redundant_arrays(building_graphs, red_tolerance=red_tolerance)
            #last_graphs = remove_redundant_arrays_slow(last_graphs, red_tolerance=red_tolerance)
            self.graphs += last_graphs
            self.graphs = remove_redundant_arrays(self.graphs, red_tolerance=red_tolerance)
            print(f'Iteration {iteration} built {len(building_graphs)} total and {len(last_graphs)} unique graphs')
            print(f'    Currently {len(self.graphs)} graphs')

        biggest_graph = 0
        biggest_graph_counts = 1
        graph_sizes = []
        for g in self.graphs:
            graph_sizes.append(g.shape[0])
            if g.shape[0] > biggest_graph:
                biggest_graph = g.shape[0]
                biggest_graph_counts = 1
            elif g.shape[0] == biggest_graph:
                biggest_graph_counts += 1
        print(f'Biggest graph is {biggest_graph}. There are {biggest_graph_counts}')
        fig, axes = plt.subplots(1, 1, figsize=(6, 3))
        axes.bar(np.arange(biggest_graph + 1), np.bincount(graph_sizes), width=1)
        plt.show()

    def plot_graphs(self, size_lim, plot=True):
        basis_all = []
        score_all = []
        for graph in self.graphs:
            if graph.shape[0] >= size_lim:
                points = np.row_stack([graph[:, :3], -graph[:, :3]])
                differences = points[np.newaxis, :, :] - points[:, np.newaxis, :]
                tril_indices = np.tril_indices(points.shape[0], k=1)
                differences = differences[tril_indices[0], tril_indices[1]]

                all_peaks = np.row_stack([points, differences])
                cluster = sklearn.cluster.DBSCAN(eps=0.01, min_samples=1)
                cluster = cluster.fit(all_peaks)
                labels = np.unique(cluster.labels_)
                labels = labels[labels != -1]
                all_peaks_unique = []
                for label_index in labels:
                    all_peaks_unique.append(
                        all_peaks[cluster.labels_ == label_index].mean(axis=0)
                        )
                all_peaks_unique = np.row_stack(all_peaks_unique)
                all_peaks_unique = all_peaks_unique[np.linalg.norm(all_peaks_unique, axis=1) > 0.005]
                basis, score, error = find_grid_basis(all_peaks_unique, verbose=plot)
                if not basis is None:
                    basis_all.append(basis)
                    score_all.append([score, error])
                    if plot:
                        fig, axis = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={'projection': '3d'})
                        axis.scatter(0, 0, 0, color=[0, 0, 0], s=50, marker='x')
                        for index_0 in range(graph.shape[0]):
                            axis.scatter(
                                graph[index_0, 0], graph[index_0, 1], graph[index_0, 2],
                                color=[0, 0, 0.8]
                                )
                            axis.scatter(
                                -graph[index_0, 0], -graph[index_0, 1], -graph[index_0, 2],
                                color=[0.8, 0, 0]
                                )
                        axis.scatter(
                            differences[:, 0], differences[:, 1], differences[:, 2],
                            color=[0, 0.8, 0]
                            )
                        axis.set_box_aspect([1,1,1])
                        for i in range(3):
                            axis.quiver(
                                0, 0, 0,
                                basis[0, i], basis[1, i], basis[2, i],
                                )
                        # Set equal axis limits centered on origin
                        max_range = max(np.max(np.abs(points)), np.max(np.abs(differences)))
                        axis.set_xlim([-max_range, max_range])
                        axis.set_ylim([-max_range, max_range])
                        axis.set_zlim([-max_range, max_range])
                        fig.tight_layout()
                        plt.show()
        self.basis = np.stack(basis_all, axis=0)
        self.basis_score = np.stack(score_all, axis=0)
        """
        d = -np.sum(self.basis, axis=2)
        s6 = np.column_stack((
            np.sum(self.basis[:, :, 1] * self.basis[:, :, 2], axis=1),
            np.sum(self.basis[:, :, 0] * self.basis[:, :, 2], axis=1),
            np.sum(self.basis[:, :, 0] * self.basis[:, :, 1], axis=1),
            np.sum(self.basis[:, :, 0] * d, axis=1),
            np.sum(self.basis[:, :, 1] * d, axis=1),
            np.sum(self.basis[:, :, 2] * d, axis=1),
            ))
        unit_cell, _, s6 = selling_reduction(reciprocal_uc_conversion(get_unit_cell_from_s6(s6)))
        good_indices = np.all(unit_cell < 100, axis=1)
        unit_cell = unit_cell[good_indices]
        s6 = s6[good_indices]
        cluster = sklearn.cluster.DBSCAN(eps=2, min_samples=1)
        cluster = cluster.fit(s6)
        labels = np.unique(cluster.labels_)
        labels = labels[labels != -1]
        s6_unique = []
        for label_index in labels:
            s6_unique.append(
                s6[cluster.labels_ == label_index].mean(axis=0)
                )
        s6_unique = np.row_stack(s6_unique)
        unit_cell_unique = get_unit_cell_from_s6(s6_unique)
        print('Unit cells')
        for i in range(unit_cell_unique.shape[0]):
            print(unit_cell_unique[i])
        """

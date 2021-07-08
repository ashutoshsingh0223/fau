import cv2
from typing import Tuple, Dict, List
import numpy as np
import scipy.spatial
from itertools import product

import random

# These are typehints, they mostly make the code readable and testable
t_points = np.array
t_descriptors = np.array
t_homography = np.array
t_img = np.array
t_images = Dict[str, t_img]
t_homographies = Dict[Tuple[str, str], t_homography]  # The keys are the keys of src and destination images

np.set_printoptions(edgeitems=30, linewidth=180,
                    formatter=dict(float=lambda x: "%8.05f" % x))


def extract_features(img: t_img, num_features: int = 500) -> Tuple[t_points, t_descriptors]:
    """Extract keypoints and their descriptors.
    The OpenCV implementation of ORB is used as a backend.
    https://en.wikipedia.org/wiki/Oriented_FAST_and_rotated_BRIEF

    Args:
        img: a numpy array of [H x W x 3] size with byte values.
        num_features: an integer signifying how many points we desire.

    Returns:
        A tuple containing a numpy array of [N x 2] and numpy array of [N x 32]
    """

    algo = cv2.ORB_create(nfeatures=num_features)
    kp, des = algo.detectAndCompute(img, None)
    kp = np.array([k.pt for k in kp])
    return kp, des


def filter_and_align_descriptors(f1: Tuple[t_points, t_descriptors], f2: Tuple[t_points, t_descriptors],
                                 similarity_threshold=.7, similarity_metric='hamming') -> Tuple[t_points, t_points]:
    """Aligns pairs of keypoints from two images.
    Aligns keypoints from two images based on descriptor similarity.
    If K points have been detected in image1 and J points have been detected in image2, the result will be to sets of N
    points representing points with similar descriptors; where N <= J and K <= points.

    Args:
        f1: A tuple of two numpy arrays with the first array having dimensions [N x 2] and the second one [N x M]. M
            representing the dimensionality of the point features. In the case of ORB features, M is 32.
        f2: A tuple of two numpy arrays with the first array having dimensions [J x 2] and the second one [J x M]. M
            representing the dimensionality of the point features. In the case of ORB features, M is 32.
        similarity_threshold: The ratio the distance of most similar descriptor in image2 to the distance of the second
            most similar ratio.
        similarity_metric: A string with the name of the metric by witch distances are calculated. It must be compatible
            with the ones that are defined for scipy.spatial.distance.cdist.

    Returns:
        A tuple of numpy arrays both sized [N x 2] representing the similar point locations.

    """
    assert f1[0].dtype == f2[0].dtype == np.double
    assert f1[0].shape[1] == f2[0].shape[1] == 2  # descriptor size
    assert f1[1].shape[1] == f2[1].shape[1] == 32  # points size

    # step 1: compute distance matrix (1 to 8 lines)
    distance_matrix = scipy.spatial.distance.cdist(f1[1], f2[1], metric=similarity_metric)

    # step 2: computing the indexes of src dst so that src[src_idx,:] and dst[dst,:] refer to matching points.
    matching_indices = np.where(distance_matrix >= 0)

    # step 3: find a boolean index of the matched pairs that is true only if a match was significant.
    # A match is considered significant if the ratio of it's distance to the second best is lower than a given
    # threshold.

    # For source
    distance_matrix_temp = distance_matrix.copy()
    # Sorting all rows in ascending order
    distance_matrix_temp.sort(axis=1)
    second_best = distance_matrix_temp[:, 1]
    ratios = distance_matrix / second_best.reshape(len(second_best), 1)
    indices = np.where(ratios < similarity_threshold)
    src_points = f1[0][indices[0], :]
    dst_points = f2[0][indices[1], :]


    # Hint: use the previously computed distance matrix to find the second best match.

    # step 4: removing non significant matches and return the aligned points (their location only!)

    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # matches = bf.match(f1[1], f2[1])
    # matches = sorted(matches, key=lambda x: x.distance)
    # points1 = np.zeros((len(matches), 2), dtype=np.float32)
    # points2 = np.zeros((len(matches), 2), dtype=np.float32)
    # thresholdMatches = []
    # for i, match in enumerate(matches):
    #     points1[i, :] = f1[match.queryIdx].pt
    #     points2[i, :] = f2[match.trainIdx].pt

    return src_points, dst_points


def compute_homography(f1: np.array, f2: np.array) -> np.array:
    """Computes the homography matrix given matching points.

    In order to define a homography a minimum of 4 points are needed but the homography can also be overdefined with 5
    or more points.

    Args:
        f1: A numpy array of size [N x 2] containing x and y coordinates of the source points.
        f2: A numpy array of size [N x 2] containing x and y coordinates of the destination points.

    Returns:
        A [3 x 3] numpy array containing normalised homography matrix.
    """
    # Homogeneous coordinates
    homography_matrix = np.zeros((3, 3))
    assert f1.shape[0] == f1.shape[0] >= 4

    A = np.zeros((2*f1.shape[0], 9), dtype=np.float64)
    for i in range(f1.shape[0]):
        p_x, p_y = f1[i]
        q_x, q_y = f2[i]
        A[2 * i] = np.array([-1 * p_x, -1 * p_y, -1, 0, 0, 0, p_x * q_x, p_y * q_x, q_x], dtype=np.float64)
        A[2 * i + 1] = np.array([0, 0, 0, -1 * p_x, -1 * p_y, -1, p_x * q_y, p_y * q_y, q_y], dtype=np.float64)

    U, S, V = np.linalg.svd(A)
    # Taking last row ov V as we want last column of V^T
    homography_matrix = V[-1, :]
    homography_matrix = homography_matrix.reshape(3, 3)
    homography_matrix_normalized = homography_matrix / homography_matrix[-1, -1]

    # TODO 3
    # - Construct the (>=8) x 9 matrix A.
    # - Use the formula from the exercise sheet.
    # - Note that every match contributes to exactly two rows of the matrix.
    # - Extract the homogeneous solution of Ah=0 as the rightmost column vector of V.
    # - Store the result in H.
    # - Normalize H
    # Hint: No loops are needed but up to to 2 nested loops might make the solution easier.
    return homography_matrix_normalized


def _get_inlier_count(src_points: np.array, dst_points: np.array, homography: np.array,
                      distance_threshold: float) -> int:
    """Computes the number of inliers for a homography given aligned points.
    ## - Project the image points from image 1 to image 2
    ## - A point is an inlier if the distance between the projected point and
    ##      the point in image 2 is smaller than threshold.
    Args:
        src_points: a numpy array of [N x 2] containing source points.
        dst_points: a numpy array of [N x 2] containing source points.
        homography: a [3 x 3] numpy array.
        distance_threshold: a float representing the norm of the difference between to points so that they will be
            considered the same (near enough).

    Returns:
        An integer counting how many transformed source points matched destination.
    """
    assert src_points.shape[1] == dst_points.shape[1] == 2
    assert src_points.shape[0] == dst_points.shape[0]

    # step 1: create normalized coordinates for points (maybe [x, y] --> [x, y, 1]) (4 lines)
    src_homo = np.ones((src_points.shape[0], src_points.shape[1] + 1))
    src_homo[:, :-1] = src_points

    # step 2: project the image points from image 1 to image 2 using the homography (1 line)
    proj = np.matmul(homography, src_homo.T).T

    # step 3: re-normalize the projected points ([x, y, l] --> [x/l, y/l]) (1 line)
    norm_proj = (proj / proj[:, -1].reshape(len(proj), 1))[:, :-1]

    # step 4: compute and return number of inliers (3 lines)
    # Hint: You might use np.linalg.norm
    distances = np.linalg.norm(norm_proj - dst_points, axis=1)
    num = len(np.where(distances > distance_threshold)[0])

    return num


def ransac(src_features: Tuple[t_points, t_descriptors], dst_features: Tuple[t_points, t_descriptors], steps,
           distance_threshold, n_points=4, similarity_threshold=.7) -> np.array:
    """Computes the best homography given noisy point descriptors.

    https://en.wikipedia.org/wiki/Random_sample_consensus
    
    Args:
        src_features: A tuple with points and their descriptors detected in the source image.
        dst_features: A tuple with points and their descriptors detected in the destination image.
        steps: An integer defining how many iterations to define.
        distance_threshold: A float defining how far should to points be to be considered the same.
        n_points: The number of point pairs used to compute the homography, it must be grater than 3.
        similarity_threshold: The ratio of the most similar descriptor to the second most similar in order to consider
            that descriptors from the two images match.

    Returns:
        A numpy array containing the homography.
    """

    # step 1: filter and align descriptors (1 line)
    src_key_points, dst_key_points = filter_and_align_descriptors(f1=src_features, f2=dst_features,
                                                                  similarity_threshold=similarity_threshold)

    # step 2: initialize the optimization loop
    best_count = 0
    best_homography = np.eye(3)
    # step 3: optimization loop
    for n in range(steps):
        if n == steps - 1:
            print(f"Step: {n:4}  {best_count} RANSAC points match!")
        # step a: select random subset of points (atleast 4 points) (2 lines)
        indices = random.sample(range(0, len(src_key_points)), n_points)

        # step b: compute homography for the random points (1 line)
        homography = compute_homography(src_key_points[indices, :], dst_key_points[indices, :])

        # step c: compare the current homography to the current best homography and update the best homography using
        # inlier count (4 lines)
        num = _get_inlier_count(src_key_points, dst_key_points, homography, distance_threshold=distance_threshold)
        if num > best_count:
            best_count = num
            best_homography = homography

    # step 4: return the best homography
    return best_homography



def probagate_homographies(homographies: t_homographies, reference_name: str) -> t_homographies:
    """Computes homographies from every image to the reference image given a homographies between all pairs of
    consecutive images.

    This method could be loosely described as applying Dijkstra's algorithm applied to exploit the commutative
    relationship of matrix multiplication and compute homography matrices between all images and any image.

    Args:
        homographies: A dictionary where the keys are tuples with the names of each image pair and the values are
            [3 x 3] arrays containing the homographies between those images.
        reference_name: The of the image which will be the destination for all homographies.

    Returns:
        A dictionary of the same form as the input mappning all images to the reference.
    """
    initial = {k: v for k, v in homographies.items()}  # deep copy
    for k, h in list(initial.items()):
        initial[(k[1], k[0])] = np.linalg.inv(h)
    initial[(reference_name, reference_name)] = np.eye(3)  # Added the identity homography for the reference
    desired = set([(k[0], reference_name) for k in homographies.keys()])
    solved = {k: v for k, v in initial.items() if k[1] == reference_name}
    while not (set(solved.keys()) >= desired):

        new_steps = set([(i, s) for i, s in product(initial.keys(), solved.keys()) if
                     s[1] != i[0] and s[0] == i[1] and s[0] != s[1] and (i[0], s[1]) not in solved.keys()])
        # s[1] != i[0] no pair who's product leads to identity
        # s[0] == i[1] only connected pairs
        # s[0]!=s[1] no identity in the solution
        # set removes duplicates

        assert len(new_steps) > 0  # not all desired can be linked to reference
        for initial_k, solved_k in new_steps:
            new_key = initial_k[0], solved_k[1]
            solved[solved_k]
            initial[initial_k]
            solved[new_key] = np.matmul(solved[solved_k], initial[initial_k])
    return solved


def compute_panorama_borders(images: t_images, homographies: t_homographies) -> Tuple[float, float, float, float]:
    """Computes the bounding box of the panorama defined the images and the homographies mapping them to the reference.

    This bounding box can have non integer and even negative coordinates.

    Args:
        images: A dictionary mapping image names to numpy arrays containing images.
        homographies:  A dictionary mapping Tuples with pairs image names to numpy arrays representing homographies
            mapping from the first image to the second.

    Returns:
        A tuple containing the bounding box [left, top, right, bottom] of the whole panorama if stiched.

    """
    homographies = {k[0]: v for k, v in homographies.items()}  # assining homographies to their source image
    assert homographies.keys() == images.keys()  # map homographies to source image only
    all_corners = []
    for name in sorted(images.keys()):
        img, homography = images[name], homographies[name]
        width, height = img.shape[0], img.shape[1]
        corners = ((0, 0), (0, width), (height, width), (height, 0))
        corners = np.array(corners, dtype='float32')
        all_corners.append(cv2.perspectiveTransform(corners[None, :, :], homography)[0, :, :])
    all_corners = np.concatenate(all_corners, axis=0)
    left, right = np.floor(all_corners[:, 0].min()), np.ceil(all_corners[:, 0].max())
    top, bottom = np.floor(all_corners[:, 1].min()), np.ceil(all_corners[:, 1].max())
    return left, top, right, bottom


def translate_homographies(homographies: t_homographies, dx: float, dy: float):
    """Applies a uniform translation to a dictionary with homographies.

    Args:
        homographies: A dictionary mapping Tuples with pairs image names to numpy arrays representing homographies
            mapping from the first image to the second.
        dx: a float representing the horizontal displacement of the translation.
        dy: a float representing the vertical displacement of the translation.

    Returns:
        a copy of the homographies dict which maps the same keys to the translated matrices.
    """
    # step 1: create a translation matrix (3 lines)
    print(homographies)
    modified_homographies = {}
    for k in homographies:
        H_ = np.eye(len(homographies[k]))
        H_[:, -1] = np.array([dx, dy, 1])
        modified_homographies[k] = np.dot(H_, homographies[k])

    return modified_homographies


def stitch_panorama(images: t_images, homographies: t_homographies, output_size: Tuple[int, int],
                   rendering_order: List[str] = []) -> t_images:
    """Stiches images after it reprojects them with a homography.

    Args:
        images: A dictionary mapping image names to numpy arrays containing images.
        homographies: A dictionary mapping Tuples with pairs image names to numpy arrays representing homographies
            mapping from the first image to the reference image.
        output_size: A tuple with integers representing the witdh and height of the resulting panorama.
        rendering_order: A list containing the names of the images representing the order in witch the images will be
            overlaid. The list must contain either all images names in some permutation or be empty in which case, the
            images will be rendered in the alphanumeric order of their names.
    Returns:
        A numpy array with the panorama image.
    """
    homographies = {k[0]: v for k, v in homographies.items()}  # assining homographies to their source image
    assert homographies.keys() == images.keys()
    if rendering_order == []:
        rendering_order = sorted(images.keys())
    panorama = np.zeros([output_size[1], output_size[0], 3], dtype=np.uint8)
    for name in rendering_order:
        rgba_img = cv2.cvtColor(images[name], cv2.COLOR_RGB2RGBA)
        rgba_img[:, :, 3] = 255
        tmp = cv2.warpPerspective(rgba_img, homographies[name], output_size, cv2.INTER_LINEAR_EXACT)
        new_pixels = ((tmp[:, :, 3] == 255)[:, :, None] & (panorama == np.zeros([1, 1, 3])))
        old_pixels = 1 - new_pixels
        panorama[:, :, :] = panorama * old_pixels + tmp[:, :, :3] * new_pixels
    return panorama


def create_stitched_image(images: t_images, homographies: t_homographies, reference_name: str,
                          rendering_order: List[str] = []):
    """Will create a panorama by stitching the input images after reprojecting them.

    Args:
        images: A dictionary mapping image names to numpy arrays containing images.
        homographies: A dictionary mapping Tuples with pairs image names to numpy arrays representing homographies
            that can reproject the first image to be aligned with the reference image.
        reference_name: A string with the name of the image to which all other images will be aligned.
        rendering_order: A list containing the names of the images representing the order in witch the images will be
            overlaid. The list must contain either all images names in some permutation or be empty in which case, the
            images will be rendered in the alphanumeric order of their names.
    Returns:
        A numpy array with the panorama image.
    """
    #  from homographies between consecutive images we compute all homographies from any image to the reference.
    homographies = probagate_homographies(homographies, reference_name=reference_name)
    #  lets calculate the panorama size
    left, top, right, bottom = compute_panorama_borders(images, homographies)
    width = int(1 + np.ceil(right) - np.floor(left))
    height = int(1 + np.ceil(bottom) - np.floor(top))
    #  lets make the homographies translate all images inside the panorama.
    homographies = translate_homographies(homographies, -left, -top)
    return stitch_panorama(images, homographies, (width, height), rendering_order=rendering_order)


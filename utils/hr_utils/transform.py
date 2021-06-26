import cv2
import math
import torch
import numpy as np


def get_transform(center, scale, output_size, rot=0):
    """
    General image processing functions
    """
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(output_size[1]) / h
    t[1, 1] = float(output_size[0]) / h
    t[0, 2] = output_size[1] * (-float(center[0]) / h + .5)
    t[1, 2] = output_size[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -output_size[1] / 2
        t_mat[1, 2] = -output_size[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform_keypoint(kps, meta, invert=False):
    keypoints = kps.copy()
    if invert:
        meta = np.linalg.inv(meta)
    keypoints[:, :2] = np.dot(keypoints[:, :2], meta[:2, :2].T) + meta[:2, 2]
    return keypoints


def _get_max_preds(heatmaps):
    """Get keypoint predictions from score maps.
    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W
    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
    Returns:
        np.ndarray[N, K, 2]: Predicted keypoint location.
        np.ndarray[N, K, 1]: Scores (confidence) of the keypoints.
    """
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    pred_mask = np.tile(maxvals > 0.0, (1, 1, 2))
    preds *= pred_mask
    return preds, maxvals


def _taylor(heatmap, coord):
    """Distribution aware coordinate decoding method.
    Note:
        heatmap height: H
        heatmap width: W
    Args:
        heatmap (np.ndarray[H, W]): Heatmap of a particular joint type.
        coord (np.ndarray[2,]): Coordinates of the predicted keypoints.
    Returns:
        Updated coordinates.
    """
    H, W = heatmap.shape[:2]
    px, py = int(coord[0]), int(coord[1])
    if 1 < px < W - 2 and 1 < py < H - 2:
        dx = 0.5 * (heatmap[py][px + 1] - heatmap[py][px - 1])
        dy = 0.5 * (heatmap[py + 1][px] - heatmap[py - 1][px])
        dxx = 0.25 * (heatmap[py][px + 2] - 2 * heatmap[py][px] +
                      heatmap[py][px - 2])
        dxy = 0.25 * (heatmap[py + 1][px + 1] - heatmap[py - 1][px + 1] -
                      heatmap[py + 1][px - 1] + heatmap[py - 1][px - 1])
        dyy = 0.25 * (heatmap[py + 2 * 1][px] - 2 * heatmap[py][px] +
                      heatmap[py - 2 * 1][px])
        derivative = np.array([[dx], [dy]])
        hessian = np.array([[dxx, dxy], [dxy, dyy]])
        if dxx * dyy - dxy**2 != 0:
            hessianinv = np.linalg.inv(hessian)
            offset = -hessianinv @ derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord += offset
    return coord


def _gaussian_blur(heatmaps, kernel=11):
    """Modulate heatmap distribution with Gaussian.
     sigma = 0.3*((kernel_size-1)*0.5-1)+0.8
     sigma~=3 if k=17
     sigma=2 if k=11;
     sigma~=1.5 if k=7;
     sigma~=1 if k=3;
    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W
    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.
    Returns:
        Modulated heatmap distribution.
    """
    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    batch_size = heatmaps.shape[0]
    num_joints = heatmaps.shape[1]
    height = heatmaps.shape[2]
    width = heatmaps.shape[3]
    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = np.max(heatmaps[i, j])
            dr = np.zeros((height + 2 * border, width + 2 * border))
            dr[border:-border, border:-border] = heatmaps[i, j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            heatmaps[i, j] = dr[border:-border, border:-border].copy()
            heatmaps[i, j] *= origin_max / np.max(heatmaps[i, j])
    return heatmaps


def keypoints_from_heatmaps(heatmaps,
                            metas=None,
                            post_process=True,
                            unbiased=True,
                            kernel=7):
    """Get final keypoint predictions from heatmaps and transform them back to
    the image.
    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W
    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        center (np.ndarray[N, 2]): Center of the bounding box (x, y).
        scale (np.ndarray[N, 2]): Scale of the bounding box
            wrt height/width.
        post_process (bool): Option to use post processing or not.
        unbiased (bool): Option to use unbiased decoding.
            Paper ref: Zhang et al. Distribution-Aware Coordinate
            Representation for Human Pose Estimation (CVPR 2020).
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.
    Returns:
        preds (np.ndarray[N, K, 2]): Predicted keypoint location in images.
        maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """

    preds, maxvals = _get_max_preds(heatmaps)
    N, K, H, W = heatmaps.shape

    if post_process:
        if unbiased:  # alleviate biased coordinate
            assert kernel > 0
            # apply Gaussian distribution modulation.
            heatmaps = _gaussian_blur(heatmaps, kernel)
            heatmaps = np.maximum(heatmaps, 1e-10)
            heatmaps = np.log(heatmaps)
            for n in range(N):
                for k in range(K):
                    preds[n][k] = _taylor(heatmaps[n][k], preds[n][k])
        else:
            # add +/-0.25 shift to the predicted locations for higher acc.
            for n in range(N):
                for k in range(K):
                    heatmap = heatmaps[n][k]
                    px = int(preds[n][k][0])
                    py = int(preds[n][k][1])
                    if 1 < px < W - 1 and 1 < py < H - 1:
                        diff = np.array([
                            heatmap[py][px + 1] - heatmap[py][px - 1],
                            heatmap[py + 1][px] - heatmap[py - 1][px]
                        ])
                        preds[n][k] += np.sign(diff) * .25
    preds *= 4
    # # Transform back to the image
    if metas is not None:
        for i in range(N):
            preds[i] = transform_keypoint(preds[i], metas[i], invert=True)

    return preds, maxvals


def generate_target_unbiased_encoding(joints,
                                      sigma=1.5,
                                      image_size=(256, 256),
                                      heatmap_size=(64, 64)):
    num_joints = joints.shape[0]
    target_weight = np.ones((num_joints, 1), dtype=np.float32)
    target = np.zeros((num_joints, heatmap_size[1], heatmap_size[0]),
                      dtype=np.float32)
    tmp_size = sigma * 3
    image_size = np.array(image_size, dtype=np.int32)
    heatmap_size = np.array(heatmap_size, dtype=np.int32)
    feat_stride = image_size / heatmap_size
    for joint_id in range(num_joints):
        mu_x = joints[joint_id, 0] / feat_stride[0]
        mu_y = joints[joint_id, 1] / feat_stride[1]
        # Check that any part of the gaussian is in-bounds
        ul = [mu_x - tmp_size, mu_y - tmp_size]
        br = [mu_x + tmp_size + 1, mu_y + tmp_size + 1]
        if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] or br[
                0] < 0 or br[1] < 0 or joints[joint_id, 2] <= 0:
            target_weight[joint_id] = 0
            continue
        x = np.arange(0, heatmap_size[0], 1, np.float32)
        y = np.arange(0, heatmap_size[1], 1, np.float32)
        y = y[:, None]
        target[joint_id] = np.exp(-((x - mu_x)**2 + (y - mu_y)**2) /
                                  (2 * sigma**2))
    return target, target_weight


import cv2
import shutil
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
         [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15],
         [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]


def fig2data(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    import PIL.Image as Image
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    return image


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copy(filename, 'bect_checkpoint.pth')


def show_image(image, pts, filename, box=None):
    figure = plt.figure()
    # plt.imshow(image)
    plt.imshow(image)
    for p in range(pts.shape[0]):
        if pts[p, 2] != 0:
            plt.plot(pts[p, 0], pts[p, 1], 'w.')
            # plt.text(pts[p, 0], pts[p, 1], '{0}'.format(p))
    for ie, e in enumerate(edges):
        if np.all(pts[e, 2] != 0):
            rgb = matplotlib.colors.hsv_to_rgb(
                [ie / float(len(edges)), 1.0, 1.0])
            plt.plot(pts[e, 0], pts[e, 1], color=rgb)
    if box is not None:
        x, y, w, h = box
        head_box = np.array([[x, y], [x + w, y + h]])
        plt.plot(head_box[0, 0], head_box[0, 1], 'b*', ms=10)
        plt.plot(head_box[1, 0], head_box[1, 1], 'b*', ms=10)
        plt.plot(head_box[0:2, 0], head_box[[0, 0], 1], 'b')
        plt.plot(head_box[0:2, 0], head_box[[1, 1], 1], 'b')
        plt.plot(head_box[[0, 0], 0], head_box[0:2, 1], 'b')
        plt.plot(head_box[[1, 1], 0], head_box[0:2, 1], 'b')

    # img = fig2data(figure)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # cv2.resize(img, (960, 544))
    # cv2.imwrite(filename, img)
    # cv2.imshow('figure2data', img)
    # cv2.waitKey(0)

    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def _calc_distances(preds, targets, normalize):
    """Calculate the normalized distances between preds and target.
    Note:
        batch_size: N
        num_keypoints: K
    Args:
        preds (np.ndarray[N, K, 2]): Predicted keypoint location.
        targets (np.ndarray[N, K, 3]): Groundtruth keypoint location.
        normalize (np.ndarray[N]): Typical value is heatmap_size/10
    Returns:
        np.ndarray[K, N]: The normalized distances.
        If target keypoints are missing, the distance is -1.
    """
    N, K, _ = preds.shape
    distances = np.full((K, N), -1, dtype=np.float32)
    eps = np.finfo(np.float32).eps
    mask = (targets[..., 0] > eps) & (targets[..., 1] > eps) & (targets[..., 2]
                                                                > eps)
    distances[mask.T] = (np.linalg.norm(
        (preds - targets[..., :2]), axis=-1) / normalize[:, None])[mask]
    return distances


def _distance_acc(distances, thr=0.5):
    """Return the percentage below the distance threshold, while ignoring
    distances values with -1.
    Note:
        batch_size: N
    Args:
        distances (np.ndarray[N, ]): The normalized distances.
        thr (float): Threshold of the distances.
    Returns:
        float: Percentage of distances below the threshold.
        If all target keypoints are missing, return -1.
    """
    distance_valid = distances != -1
    num_distance_valid = distance_valid.sum()
    if num_distance_valid > 0:
        return (distances[distance_valid] < thr).sum() / num_distance_valid
    return -1


def pose_pck_accuracy(pred, gt, normalize, thr=0.5):
    """Calculate the pose accuracy of PCK for each individual keypoint and the
    averaged accuracy across all keypoints.
    Note:
        The PCK performance metric is the percentage of joints with
        predicted locations that are no further than a normalized
        distance of the ground truth.
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W
    Args:
        pred (np.ndarray[N, K, 2]): Model output keypoints.
        gt (np.ndarray[N, K, 3]): Groundtruth keypoints.
        thr (float): Threshold of PCK calculation.
        normalize (np.ndarray[N]): Normalization factor.
    Returns:
        np.ndarray[K]: Accuracy of each keypoint.
        float: Averaged accuracy across all keypoints.
        int: Number of valid keypoints.
    """
    N, K, _ = pred.shape
    if K == 0:
        return None, 0, 0
    distances = _calc_distances(pred, gt, normalize)
    acc = np.array([_distance_acc(d, thr) for d in distances])
    valid_acc = acc[acc >= 0]
    cnt = len(valid_acc)
    avg_acc = valid_acc.mean() if cnt > 0 else 0
    return acc, avg_acc, cnt


def pose_auc(pred, gt, normalize, num_step=20):
    """Calculate the pose accuracy of PCK for each individual keypoint and the
    averaged accuracy across all keypoints for coordinates.
    Note:
        batch_size: N
        num_keypoints: K
    Args:
        pred (np.ndarray[N, K, 2]): Predicted keypoint location.
        gt (np.ndarray[N, K, 3]): Groundtruth keypoint location.
        normalize (float): Normalization factor.
    Returns:
        float: Area under curve.
    """
    x = [1.0 * i / num_step for i in range(num_step)]
    y = []
    for thr in x:
        _, avg_acc, _ = pose_pck_accuracy(pred, gt, normalize, thr)
        y.append(avg_acc)
    auc = 0
    for i in range(num_step):
        auc += 1.0 / num_step * y[i]
    return auc
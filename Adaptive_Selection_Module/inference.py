from utils import *
import joblib
import numpy as np
def adaptiveSelection(rgb_imgs, point_pairs):
    layered_points1, layered_points2, _ = featureTriangle(point_pairs[0], point_pairs[1], 20)
    H, _, _ = calcHomography([layered_points1[0], layered_points2[0]], ransac=False)
    _, points1, points2 = calcHomography(point_pairs, ransac=True)
    mean_error2, median_error2, inlier_ratio2 = evaluate_homography(points1, points2, H)
    mean_corner_shift2, max_corner_shift2 = compute_corner_displacement(rgb_imgs[0].shape, H)
    point_pairs2 = [point_pairs[1], point_pairs[0]]
    
    layered_points1_vers, layered_points2_vers, _ = featureTriangle(point_pairs2[0], point_pairs2[1], 20)
    H, _, _ = calcHomography([layered_points1_vers[0], layered_points2_vers[0]], ransac=False)
    _, points1_vers, points2_vers = calcHomography(point_pairs2, ransac=True)
    mean_error1, median_error1, inlier_ratio1 = evaluate_homography(points1_vers, points2_vers, H)
    mean_corner_shift1, max_corner_shift1 = compute_corner_displacement(rgb_imgs[1].shape, H)
    features = [
    mean_corner_shift1, max_corner_shift1,
    mean_error1, median_error1, inlier_ratio1,
    mean_corner_shift2, max_corner_shift2,
    mean_error2, median_error2, inlier_ratio2
    ]
    clf = joblib.load("stitching_rf_model.pkl")
    scaler = joblib.load("stitching_rf_scaler.pkl")
    features = np.array(features, dtype=np.float32).reshape(1, -1)
    features_scaled = scaler.transform(features)
    score = clf.predict(features_scaled)[0]
    return "1->2" if score == 1 else "2->1"

def main(rgb_imgs, point_pairs):
    label = adaptiveSelection(rgb_imgs, point_pairs)

import numpy as np

from flowsight.detect import foot_point
from flowsight.geometry import fit_homography, floor_to_image, image_to_floor, reprojection_error
from flowsight.render import blur_heads
from flowsight.track import FloorKalman


def test_foot_point_is_bottom_centre():
    fp = foot_point(np.array([[100, 50, 140, 210], [0, 0, 10, 20]]))
    assert np.allclose(fp, [[120, 210], [5, 20]])


def test_foot_point_empty():
    assert foot_point(np.zeros((0, 4))).shape == (0, 2)


IMG = [[20, 535], [940, 535], [710, 15], [250, 15]]
FLR = [[2, 21], [18, 21], [18, 2], [2, 2]]


def test_homography_maps_control_points_exactly():
    H = fit_homography(IMG, FLR)
    assert np.allclose(image_to_floor(H, IMG), FLR, atol=1e-6)
    assert reprojection_error(H, IMG, FLR)["floor_rms_m"] < 1e-6


def test_homography_roundtrip():
    H = fit_homography(IMG, FLR)
    pts = np.array([[480, 300], [100, 100], [800, 400.0]])
    assert np.allclose(floor_to_image(H, image_to_floor(H, pts)), pts, atol=1e-6)


def test_homography_known_scale():
    # image = 100 px per metre, no perspective
    img = [[0, 0], [100, 0], [100, 100], [0, 100]]
    H = fit_homography(img, [[0, 0], [1, 0], [1, 1], [0, 1]])
    assert np.allclose(image_to_floor(H, [[250, 50]]), [[2.5, 0.5]])


def test_homography_needs_four_points():
    import pytest
    with pytest.raises(ValueError):
        fit_homography([[0, 0], [1, 1], [2, 0]], [[0, 0], [1, 1], [2, 0]])


def test_blur_heads_only_touches_top_fifth():
    rng = np.random.default_rng(0)
    f = rng.integers(0, 255, (200, 100, 3), dtype=np.uint8)
    orig = f.copy()
    out = blur_heads(f, np.array([[10, 10, 90, 190]]), frac=0.2)
    assert not np.array_equal(out[10:46, 10:90], orig[10:46, 10:90])    # head blurred
    assert np.array_equal(out[60:190, 10:90], orig[60:190, 10:90])      # body untouched


def test_kalman_speed_for_constant_walk():
    k = FloorKalman(0, 0)
    sp = 0.0
    for i in range(1, 60):
        x, y, vx, vy = k.update(i / 15, 1.2 * i / 15, 0)
        sp = np.hypot(vx, vy)
    assert abs(sp - 1.2) < 0.15


def test_kalman_stationary_is_slow():
    k = FloorKalman(5, 5)
    rng = np.random.default_rng(1)
    for i in range(1, 90):
        _, _, vx, vy = k.update(i / 15, 5 + rng.normal(0, .1), 5 + rng.normal(0, .1))
    assert np.hypot(vx, vy) < 0.2

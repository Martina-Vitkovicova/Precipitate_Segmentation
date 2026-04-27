import numpy as np


def normalise_light_direction(light_direction: tuple[int, int, int]) -> np.ndarray:
    """
    Convert semantic light direction [x, y, z] to a normalised image-space vector.

    Semantic convention:
        +x -> light from right
        +y -> light from top
        +z -> more frontal

    Image-space convention:
        +x -> right
        +y -> down

    Therefore y must be flipped once here.
    """
    lx, ly, lz = light_direction
    light_vec = np.array([lx, -ly, lz], dtype=np.float32)

    norm = np.linalg.norm(light_vec)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    return light_vec / norm

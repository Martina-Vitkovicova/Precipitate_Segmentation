from typing import List, Tuple
import cv2
import numpy as np

from src.config_parsing import GeneralParams, EtchPitsParams, TextureParams
from src.generators.background_gen import add_porous_texture
from src.type_definitions import RandomState, Image


def add_etch_pits(image: Image, etch_pits_rng: RandomState, general_params: GeneralParams,
                  params: EtchPitsParams) -> Image:
    num = etch_pits_rng.randint(*params.num_range)
    dark_shadow, light_shadow, highlight = (params.shading_dark_shadow, params.shading_light_shadow,
                                            params.shading_highlight)
    h, w = image.shape

    # padding so pits near the border don't leave artifact borders
    pad = int(max(params.size_range))
    pad_image = cv2.copyMakeBorder(image, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT_101)
    pad_h, pad_w = pad_image.shape

    texture_params = TextureParams(params.texture_density_range, params.texture_blur_range,
                                   params.texture_intensity_range, params.texture_size_range)

    for _ in range(num):
        angle = etch_pits_rng.randint(0, 180)
        size = generate_pit_size(etch_pits_rng, params.size_range)
        base_color = etch_pits_rng.randint(*params.intensity_range)
        centers = generate_centers(etch_pits_rng, pad_w, pad_h, general_params.light_direction, size)
        colors = [highlight, dark_shadow, light_shadow, base_color]

        pad_image = add_colored_shapes(pad_image, etch_pits_rng, centers, size, angle, colors,
                                       general_params.light_direction, texture_params)

    # crop back to original size
    image = pad_image[pad:pad + h, pad:pad + w]
    return image


def generate_centers(rng, w, h, light_dir, size):
    center_x, center_y = rng.randint(0, w), rng.randint(0, h)

    # normalise light direction
    light_x, light_y, _ = light_dir
    light_vec = np.array([light_x, light_y], dtype=np.float32)
    norm = np.linalg.norm(light_vec) + 1e-6
    light_vec /= norm

    step = 0.25 * min(size)
    step_vec = step * light_vec

    # offsets along the light direction (<0 = towards light)
    offsets = [0.0, -1.1, 0.0, 0.75]

    centers = []
    for k in offsets:
        offset = k * step_vec
        x = int(round(center_x + offset[0]))
        y = int(round(center_y + offset[1]))
        centers.append((x, y))

    return centers


def generate_pit_size(rng, size_range):
    size_x = rng.randint(*size_range)
    size_y = max(3, size_x - rng.randint(size_x // 3, size_x // 2))
    return size_x, size_y


def add_colored_shapes(image, rng, centers, size, angle, colors, light_dir, texture_params):
    temp_image = np.zeros_like(image, dtype=np.uint8)

    scales = [1.0, 0.85, 0.76, 0.7]
    base_center = centers[0]
    base_contour = create_pit_shape(image, base_center, angle, size, scales[0], rng)

    for center, color, scale in zip(centers, colors, scales):
        scaled_contour = scale_contour(base_contour, scale, center)
        dx = center[0] - base_center[0]
        dy = center[1] - base_center[1]
        moved_contour = translate_contour(scaled_contour, (dx, dy))

        color_w_deviation = max(0, min(255, int(color + rng.randint(-5, 5))))
        fin_color = (color_w_deviation, color_w_deviation, color_w_deviation)
        cv2.drawContours(temp_image, [moved_contour], -1, fin_color, -1)

    blur_size = max(3, int(min(size) * 0.4) | 1)
    blurred = cv2.GaussianBlur(temp_image, (blur_size, blur_size), sigmaX=0)

    mask = (temp_image > 0).astype(np.uint8)
    blurred_w_texture = add_texture(blurred, mask, rng, light_dir, texture_params)
    result = merge_pit_into_image(image, blurred_w_texture, mask, size)

    return result


def create_pit_shape(image, center, angle, size, scale, rng):
    temp_shape_mask = np.zeros_like(image, dtype=np.uint8)
    ellipse_axes = (int(size[0] * scale), int(size[1] * scale))
    color = (255, 255, 255)

    cv2.ellipse(temp_shape_mask, center, ellipse_axes, angle, 0, 360, color, -1)
    contours, _ = cv2.findContours(temp_shape_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    base_contour = contours[0]

    avg_size = (size[0] + size[1]) // 2
    max_noise = avg_size * 0.5
    smoothing = max_noise * 0.15
    noise = generate_smooth_noise(base_contour, center, rng, max_noise, smoothing)
    base_contour = (base_contour.astype(np.int32) + noise).astype(np.int32)

    return base_contour.reshape(-1, 1, 2)


def generate_smooth_noise(contour, center, rng, max_noise, smoothing):
    pts = contour[:, 0, :].astype(np.float32)

    # unit radial direction at each point
    center = np.array(center, np.float32)
    vec_to_center = pts - center
    radii = np.linalg.norm(vec_to_center, axis=1, keepdims=True) + 1e-6
    radial_dir = vec_to_center / radii

    n = len(contour)
    blend_len = max(2, int(n * 0.2))
    core_noise = rng.normal(0.0, max_noise * 2, size=n - blend_len).astype(np.float32)
    t = np.linspace(0, 1, blend_len + 2, dtype=np.float32)[1:-1]
    transition = (1 - t) * core_noise[-1] + t * core_noise[0]
    noise = np.concatenate([core_noise, transition]).astype(np.float32)

    noise_smooth = cv2.GaussianBlur(noise, (0, 0), sigmaX=smoothing)
    radial_displacement = radial_dir * noise_smooth.reshape(n, 1)

    return radial_displacement.astype(np.int32)


def scale_contour(contour, scale_factor, center):
    center_x, center_y = center
    centered = contour - [center_x, center_y]
    scaled = centered * scale_factor
    scaled = (scaled + [center_x, center_y]).astype(np.int32)

    return scaled


def translate_contour(contour, delta):
    dx, dy = delta
    translated = contour + np.array([[dx, dy]], dtype=np.int32)
    return translated.astype(np.int32)


def add_texture(blurred, mask, rng, light_dir, texture_params):
    result = blurred.astype(np.float32)
    temp_texture = np.zeros(mask.shape, dtype=np.float32)

    temp_texture = add_porous_texture(temp_texture, rng, light_dir, texture_params)

    mask = mask.astype(bool)
    result[mask] = np.clip(blurred[mask] + 0.4 * temp_texture[mask], 0, 255)

    return result


def merge_pit_into_image(image, pit, mask, size):
    avg_size = (size[0] + size[1]) / 2
    er_size = max(3, int(avg_size * 0.4) | 1)
    temp_mask = cv2.erode(mask, np.ones((er_size, er_size)), iterations=1)

    blur_size = max(3, int(avg_size * 0.3) | 1)
    alpha = cv2.GaussianBlur(temp_mask.astype(np.float32), (blur_size, blur_size), sigmaX=0)
    alpha /= alpha.max()
    alpha = np.clip(alpha, 0.0, 1.0)

    img_f = image.astype(np.float32)
    pit_f = pit.astype(np.float32)
    hf_strength = 0.35

    low = cv2.GaussianBlur(img_f, (0, 0), 2)
    hf = img_f - low

    # bleed that HF texture into the pit interior (stronger where alpha is high)
    pit_f = np.clip(pit_f + (hf_strength * alpha) * hf, 0.0, 255.0)

    merged = img_f * (1.0 - alpha) + pit_f * alpha
    merged = np.clip(merged, 0, 255).astype(np.uint8)

    return merged

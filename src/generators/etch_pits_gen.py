from dataclasses import dataclass
from typing import List
import cv2
import numpy as np

from src.utils.config_parsing import GeneralParams, EtchPitsParams, TextureParams
from src.generators.background_gen import add_porous_texture
from src.utils.light_dir_normalisation import normalise_light_direction
from src.utils.type_definitions import RandomState, Image, Vec3i, IntPair, Point, Contour, Mask, IntRange


@dataclass(frozen=True)
class PitLayerStructure:
    name: str
    # Relative shift along the normalized light direction (< 0 means toward the light).
    offset: float
    # Relative contour size multiplier for this layer.
    scale: float


@dataclass(frozen=True)
class PitLayer(PitLayerStructure):
    intensity: int
    center: Point


def add_etch_pits(image: Image, etch_pits_rng: RandomState, general_params: GeneralParams,
                  params: EtchPitsParams) -> Image:
    """
    Add etched pits - oval recesses with strong directional shading - into an image.

    Etch pits are generated as shallow, directionally shaded depressions whose
    number, size, intensity, and texturing behaviour are defined by
    `EtchPitsParams`. Global illumination information from `GeneralParams` is used
    to align pit shading and internal texture opposite to the common light direction.

    The function modifies the input image in-place and returns the updated image.
    All stochastic choices are driven exclusively by `etch_pits_rng`.
    """
    num = etch_pits_rng.randint(*params.num_range)
    h, w = image.shape

    # padding so pits near the border don't create artifact borders
    pad = int(max(params.size_range))
    pad_image = cv2.copyMakeBorder(image, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT_101)
    pad_h, pad_w = pad_image.shape

    texture_params = TextureParams(params.texture_density_range, params.texture_blur_range,
                                   params.texture_intensity_range, params.texture_size_range)

    pit_layer_structure = _define_pit_layer_structure(params.layer_offsets, params.layer_scales)

    for _ in range(num):
        angle = etch_pits_rng.randint(0, 180)
        size = _generate_pit_size(etch_pits_rng, params.size_range)
        base_intensity = etch_pits_rng.randint(*params.intensity_range)
        centers = _generate_centers(etch_pits_rng, pad_w, pad_h, general_params.light_direction, size,
                                    pit_layer_structure, params.layer_offset_ratio)

        pit_layers = _get_pit_layers(centers, base_intensity, pit_layer_structure, params)
        pad_image = _draw_pit(pad_image, etch_pits_rng, size, angle, general_params.light_direction, texture_params,
                              pit_layers, params.border_noise)

    # crop back to original size
    image = pad_image[pad:pad + h, pad:pad + w]
    return image


def _define_pit_layer_structure(offsets: dict[str, float] | None = None, scales: dict[str, float] | None = None,
                                ) -> list[PitLayerStructure]:
    layer_order = ["highlight", "dark_shadow", "light_shadow", "base_fill"]

    return [PitLayerStructure(name=name, offset=offsets[name], scale=scales[name]) for name in layer_order]


def _get_pit_layers(centers: List[Point], base_intensity: int, pit_layer_templates: List[PitLayerStructure],
                    params: EtchPitsParams) -> List[PitLayer]:
    layer_intensities = {
        "highlight": params.shading_highlight,
        "dark_shadow": params.shading_dark_shadow,
        "light_shadow": params.shading_light_shadow,
        "base_fill": base_intensity,
    }

    return [
        PitLayer(
            name=layer_template.name,
            offset=layer_template.offset,
            scale=layer_template.scale,
            intensity=layer_intensities[layer_template.name],
            center=centers[i],
        )
        for i, layer_template in enumerate(pit_layer_templates)
    ]


def _generate_centers(rng: RandomState, width: int, height: int, light_dir: Vec3i, size: IntPair,
                      layer_structure: List[PitLayerStructure], layer_offset_ratio: float) -> List[Point]:
    """
    Generate a set of pit layer centers with offsets along the light direction.

    The first center is the base pit position; subsequent centers are shifted
    to create layered shading consistent with the illumination direction.
    """
    center_x, center_y = rng.randint(0, width), rng.randint(0, height)

    light_vec_3d = normalise_light_direction(light_dir)
    light_vec_2d = light_vec_3d[:2]

    # Base displacement magnitude for layer offsets, proportional to the smaller pit axis.
    step = layer_offset_ratio * min(size)
    step_vec = step * light_vec_2d

    centers = []
    for layer in layer_structure:
        offset = -layer.offset * step_vec
        x = int(round(center_x + float(offset[0])))
        y = int(round(center_y + float(offset[1])))
        centers.append((x, y))

    return centers


def _generate_pit_size(rng: RandomState, size_range: IntRange) -> IntPair:
    """ Sample an elliptical pit size (major/minor axes) from a size range."""
    major_axis = rng.randint(*size_range)
    reduction = rng.randint(max(1, major_axis // 3), max(2, major_axis // 2))
    minor_axis = max(3, major_axis - reduction)
    return major_axis, minor_axis


def _draw_pit(image: Image, rng: RandomState, size: IntPair, angle: int, light_dir: Vec3i,
              texture_params: TextureParams, layers: List[PitLayer], border_noise: float) -> Image:
    """Draw a single pit from resolved pit layers, then blend it into the image."""
    pit_layers_image = np.zeros_like(image, dtype=np.uint8)

    # Use the first layer as the reference contour. Other layers are scaled and shifted
    # variants of the same base shape.
    base_layer = layers[0]
    base_center = base_layer.center
    base_contour = _create_pit_shape(image, base_center, angle, size, base_layer.scale, rng, border_noise)

    for layer in layers:
        scaled_contour = _scale_contour(base_contour, layer.scale, base_center)

        dx = layer.center[0] - base_center[0]
        dy = layer.center[1] - base_center[1]
        moved_contour = _translate_contour(scaled_contour, (dx, dy))

        intensity_w_deviation = int(np.clip(layer.intensity + rng.randint(-5, 5), 0, 255))
        fin_intensity = (intensity_w_deviation, intensity_w_deviation, intensity_w_deviation)
        cv2.drawContours(pit_layers_image, [moved_contour], -1, fin_intensity, -1)

    blur_size = max(3, int(min(size) * 0.4) | 1)
    blurred = cv2.GaussianBlur(pit_layers_image, (blur_size, blur_size), sigmaX=0)

    mask = (pit_layers_image > 0).astype(np.uint8)
    textured_pit = _add_texture(blurred, mask, rng, light_dir, texture_params)

    return _merge_pit_into_image(image, textured_pit, mask, size)


def _create_pit_shape(image: Image, center: Point, angle: int, size: IntPair, scale: float, rng: RandomState,
                      border_noise: float) \
        -> Contour:
    """ Create a base pit contour: an ellipse with smooth radial perturbations. """
    temp_shape_mask = np.zeros_like(image, dtype=np.uint8)
    ellipse_axes = (int(size[0] * scale), int(size[1] * scale))
    intensity = (255, 255, 255)

    cv2.ellipse(temp_shape_mask, center, ellipse_axes, angle, 0, 360, intensity, -1)
    contours, _ = cv2.findContours(temp_shape_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    base_contour = contours[0]

    avg_size = (size[0] + size[1]) // 2
    max_noise = avg_size * 0.5 * border_noise
    smoothing = max_noise * 0.15
    noise = _generate_smooth_noise(base_contour, center, rng, max_noise, smoothing)
    base_contour = (base_contour.astype(np.int32) + noise).astype(np.int32)

    return base_contour.reshape(-1, 1, 2)


def _generate_smooth_noise(contour: Contour, center: Point, rng: RandomState, max_noise: float, smoothing: float) \
        -> np.ndarray:
    """ Generate smooth, wrap-around radial displacement noise for a contour. """
    pts = contour[:, 0, :].astype(np.float32)

    # unit radial direction at each point
    center = np.array(center, np.float32)
    vec_to_center = pts - center
    radii = np.linalg.norm(vec_to_center, axis=1, keepdims=True) + 1e-6
    radial_dir = vec_to_center / radii

    n = len(contour)
    blend_len = max(2, int(n * 0.2))
    core_noise = rng.normal(0.0, max_noise * 2, size=n - blend_len).astype(np.float32)

    # wrap-around blend to avoid seam: interpolate last->first over blend_len.
    t = np.linspace(0, 1, blend_len + 2, dtype=np.float32)[1:-1]
    transition = (1 - t) * core_noise[-1] + t * core_noise[0]
    noise = np.concatenate([core_noise, transition]).astype(np.float32)

    noise_smooth = cv2.GaussianBlur(noise, (0, 0), sigmaX=smoothing)
    radial_displacement = radial_dir * noise_smooth.reshape(n, 1)

    return radial_displacement.astype(np.int32)


def _scale_contour(contour: Contour, scale_factor: float, center: Point) -> Contour:
    center_x, center_y = center
    centered = contour - [center_x, center_y]
    scaled = centered * scale_factor
    scaled = (scaled + [center_x, center_y]).astype(np.int32)

    return scaled


def _translate_contour(contour: Contour, delta: IntPair) -> Contour:
    dx, dy = delta
    translated = contour + np.array([[dx, dy]], dtype=np.int32)
    return translated.astype(np.int32)


def _add_texture(blurred_img: Image, mask: Mask, rng: RandomState, light_dir: Vec3i, texture_params: TextureParams) \
        -> Image:
    """ Add porous texture inside the pit, marked in `mask`. """
    result = blurred_img.astype(np.float32)
    temp_texture = np.zeros(mask.shape, dtype=np.float32)

    temp_texture = add_porous_texture(temp_texture, rng, light_dir, texture_params)

    mask = mask.astype(bool)
    result[mask] = np.clip(blurred_img[mask] + 0.4 * temp_texture[mask], 0, 255)

    return result.astype(np.uint8)


def _merge_pit_into_image(image: Image, pit: Image, mask: Mask, size: IntPair) -> Image:
    """
    Blend the pit into the base image with a soft alpha mask and HF (high-frequency) bleed.
    The HF bleed transfers some of the base image texture into the pit interior,
    helping avoid overly smooth synthetic regions.
    """
    avg_size = (float(size[0]) + float(size[1])) / 2.0
    er_size = max(3, (int(avg_size * 0.4) | 1))
    temp_mask = cv2.erode(mask, np.ones((er_size, er_size), dtype=np.uint8), iterations=1)

    blur_size = max(3, (int(avg_size * 0.3) | 1))
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

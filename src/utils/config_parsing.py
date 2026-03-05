from dataclasses import dataclass
from typing import Sequence, List, Dict, Any

from src.utils.type_definitions import IntRange, FloatRange, IntPair, Vec3i


def _as_int_range(v: Sequence[Any], *, name: str) -> IntRange:
    if not (isinstance(v, (list, tuple)) and len(v) == 2):
        raise ValueError(f"{name} must be [min, max]")
    a, b = int(v[0]), int(v[1])
    return (a, b)


def _as_float_range(v: Sequence[Any], *, name: str) -> FloatRange:
    if not (isinstance(v, (list, tuple)) and len(v) == 2):
        raise ValueError(f"{name} must be [min, max]")
    a, b = float(v[0]), float(v[1])
    return (a, b)


def _as_int_pair(v: Sequence[Any], *, name: str) -> IntPair:
    if not (isinstance(v, (list, tuple)) and len(v) == 2):
        raise ValueError(f"{name} must be [w, h]")
    return (int(v[0]), int(v[1]))


def _as_vec3i(v: Sequence[Any], *, name: str) -> Vec3i:
    if not (isinstance(v, (list, tuple)) and len(v) == 3):
        raise ValueError(f"{name} must be [x, y, z]")
    return (int(v[0]), int(v[1]), int(v[2]))


def _as_int_list(v: Sequence[Any], *, name: str) -> List[int]:
    if not isinstance(v, (list, tuple)):
        raise ValueError(f"{name} must be a list")
    return [int(x) for x in v]


@dataclass(frozen=True)
class GeneralParams:
    image_size: IntPair
    light_direction: Vec3i

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GeneralParams":
        return GeneralParams(
            image_size=_as_int_pair(d["image_size"], name="general.image_size"),
            light_direction=_as_vec3i(d["light_direction"], name="general.light_direction"),
        )


@dataclass(frozen=True)
class MatrixParams:
    grains_num_range: IntRange
    grains_intensity_range: IntRange

    border_width: int
    border_variable_width_enabled: bool
    border_jaggedness: float
    border_perturbation_noise: float

    border_shading_enabled: bool
    border_shading_intensity_range: IntRange

    perlin_noise_enabled: bool
    perlin_noise_alpha_range: IntRange
    perlin_noise_scales: List[int]

    gaussian_noise_enabled: bool
    gaussian_noise_alpha_range: IntRange

    porous_texture_enabled: bool
    porous_texture_density_range: FloatRange
    porous_texture_blur_range: FloatRange
    porous_texture_intensity_range: FloatRange
    porous_texture_size_range: IntRange

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MatrixParams":
        return MatrixParams(
            grains_num_range=_as_int_range(d["grains_num_range"], name="matrix.grains_num_range"),
            grains_intensity_range=_as_int_range(d["grains_intensity_range"], name="matrix.grains_intensity_range"),

            border_width=int(d["border_width"]),
            border_variable_width_enabled=bool(d["border_variable_width_enabled"]),
            border_jaggedness=float(d["border_jaggedness"]),
            border_perturbation_noise=float(d["border_perturbation_noise"]),

            border_shading_enabled=bool(d["border_shading_enabled"]),
            border_shading_intensity_range=_as_int_range(
                d["border_shading_intensity_range"], name="matrix.border_shading_intensity_range"
            ),

            perlin_noise_enabled=bool(d["perlin_noise_enabled"]),
            perlin_noise_alpha_range=_as_int_range(d["perlin_noise_alpha_range"], name="matrix.perlin_noise_alpha_range"),
            perlin_noise_scales=_as_int_list(d["perlin_noise_scales"], name="matrix.perlin_noise_scales"),

            gaussian_noise_enabled=bool(d["gaussian_noise_enabled"]),
            gaussian_noise_alpha_range=_as_int_range(
                d["gaussian_noise_alpha_range"], name="matrix.gaussian_noise_alpha_range"
            ),

            porous_texture_enabled=bool(d["porous_texture_enabled"]),
            porous_texture_density_range=_as_float_range(
                d["porous_texture_density_range"], name="matrix.porous_texture_density_range"
            ),
            porous_texture_blur_range=_as_float_range(
                d["porous_texture_blur_range"], name="matrix.porous_texture_blur_range"
            ),
            porous_texture_intensity_range=_as_float_range(
                d["porous_texture_intensity_range"], name="matrix.porous_texture_intensity_range"
            ),
            porous_texture_size_range=_as_int_range(d["porous_texture_size_range"],
                                                    name="matrix.porous_texture_size_range"),
        )


@dataclass(frozen=True)
class PrecipitatesParams:
    num_range: IntRange
    intensity_range: IntRange
    size_range: IntRange
    size_distribution: List[int]

    emboss_enabled: bool
    emboss_intensity_range: IntRange

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PrecipitatesParams":
        return PrecipitatesParams(
            num_range=_as_int_range(d["num_range"], name="precipitates.num_range"),
            intensity_range=_as_int_range(d["intensity_range"], name="precipitates.intensity_range"),
            size_range=_as_int_range(d["size_range"], name="precipitates.size_range"),
            size_distribution=_as_int_list(d["size_distribution"], name="precipitates.size_distribution"),
            emboss_enabled=bool(d["emboss_enabled"]),
            emboss_intensity_range=_as_int_range(d["emboss_intensity_range"], name="precipitates.emboss_intensity_range"),
        )


@dataclass(frozen=True)
class EtchPitsParams:
    num_range: IntRange
    intensity_range: IntRange
    size_range: IntRange

    shading_dark_shadow: int
    shading_light_shadow: int
    shading_highlight: int

    texture_density_range: FloatRange
    texture_blur_range: FloatRange
    texture_intensity_range: FloatRange
    texture_size_range: IntRange

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "EtchPitsParams":
        return EtchPitsParams(
            num_range=_as_int_range(d["num_range"], name="etch_pits.num_range"),
            intensity_range=_as_int_range(d["intensity_range"], name="etch_pits.intensity_range"),
            size_range=_as_int_range(d["size_range"], name="etch_pits.size_range"),
            shading_dark_shadow=int(d["shading_dark_shadow"]),
            shading_light_shadow=int(d["shading_light_shadow"]),
            shading_highlight=int(d["shading_highlight"]),
            texture_density_range=_as_float_range(d["texture_density_range"], name="etch_pits.texture_density_range"),
            texture_blur_range=_as_float_range(d["texture_blur_range"], name="etch_pits.texture_blur_range"),
            texture_intensity_range=_as_float_range(d["texture_intensity_range"], name="etch_pits.texture_intensity_range"),
            texture_size_range=_as_int_range(d["texture_size_range"], name="etch_pits.texture_size_range"),
        )


@dataclass(frozen=True)
class EffectsParams:
    corner_blur_enabled: bool
    corner_blur_radius_range: IntRange
    corner_blur_kernel_size_range: IntRange

    bottom_blur_enabled: bool
    bottom_blur_height_range: IntRange
    bottom_blur_kernel_size_range: IntRange

    illumination_gradient_enabled: bool
    illumination_gradient_strength_range: FloatRange
    illumination_gradient_angle_range: IntRange

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "EffectsParams":
        return EffectsParams(
            corner_blur_enabled=bool(d["corner_blur_enabled"]),
            corner_blur_radius_range=_as_int_range(d["corner_blur_radius_range"], name="effects.corner_blur_radius_range"),
            corner_blur_kernel_size_range=_as_int_range(
                d["corner_blur_kernel_size_range"], name="effects.corner_blur_kernel_size_range"
            ),
            bottom_blur_enabled=bool(d["bottom_blur_enabled"]),
            bottom_blur_height_range=_as_int_range(d["bottom_blur_height_range"], name="effects.bottom_blur_height_range"),
            bottom_blur_kernel_size_range=_as_int_range(
                d["bottom_blur_kernel_size_range"], name="effects.bottom_blur_kernel_size_range"
            ),
            illumination_gradient_enabled=bool(d["illumination_gradient_enabled"]),
            illumination_gradient_strength_range=_as_float_range(
                d["illumination_gradient_strength_range"], name="effects.illumination_gradient_strength_range"
            ),
            illumination_gradient_angle_range=_as_int_range(
                d["illumination_gradient_angle_range"], name="effects.illumination_gradient_angle_range"
            ),
        )


@dataclass(frozen=True)
class TextureParams:
    density_range: FloatRange
    blur_range: FloatRange
    intensity_range: FloatRange
    size_range: IntRange


@dataclass(frozen=True)
class GeneratorConfig:
    general: GeneralParams
    matrix: MatrixParams
    precipitates: PrecipitatesParams
    etch_pits: EtchPitsParams
    effects: EffectsParams

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GeneratorConfig":
        return GeneratorConfig(
            general=GeneralParams.from_dict(d["general"]),
            matrix=MatrixParams.from_dict(d["matrix"]),
            precipitates=PrecipitatesParams.from_dict(d["precipitates"]),
            etch_pits=EtchPitsParams.from_dict(d["etch_pits"]),
            effects=EffectsParams.from_dict(d["effects"]),
        )

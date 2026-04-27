# Image Generator Parameters

This directory contains configuration files for synthetic SEM image generation. The file `base_params.json` defines a general configuration based on averaged feature values from the real SEM dataset. The files `params_cl0.json` to `params_cl4.json` define cluster-specific configurations, where each file corresponds to one image cluster with distinct visual characteristics.

The parameters control various image and feature parameters like the generated image size, matrix structure, etch pits, precipitates, texture, shading, blur and more.

All intensity parameters use 8-bit grayscale values from `0` to `255`, where `0` is black and `255` is white. Parameters defined as ranges are sampled randomly during image generation, so each generated image can vary within the configured limits.
## general

The `general` section defines global image-generation settings, including the output image size and the direction of the virtual light source used for shading.

| Parameter | Type                           | Typical values                               | Description                                                                                                                                                                                   |
|---|--------------------------------|----------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `image_size` | <nobr>`[int, int]`</nobr>      | `[512, 512]` to  <nobr>`[2048, 2048]`</nobr> | Output image size in pixels. Usually square, for example `[1024, 1024]`, but rectangular images are also supported.                                                                           |
| `light_direction` | <nobr>`[int, int, int]`</nobr> | `[1, 2, 4] `                                 | Semantic light direction as `[x, y, z]`. `+x` means light from the right, `+y` means light from the top, and larger `z` means more frontal illumination. The vector is normalised before use. |

## matrix

The `matrix` section controls the generated metal matrix background consisting of grains.

| Parameter | Type | Typical values                    | Description                                                                                                                                |
|---|---|-----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| `grains_num_range` | <nobr>`[int, int]`</nobr> | <nobr>`20`–`80` </nobr>           | Controls how many matrix grains appear in the image.                                                                                       |
| `grains_intensity_range` | <nobr>`[int, int]`</nobr> | <nobr>`110`–`160` </nobr>         | Controls the grayscale brightness range of grain interiors.                                                                                |
| `border_width` | `int` | <nobr>`2`–`7` </nobr>             | Controls the thickness of grain boundaries. When variable width is enabled, it controls the maximum local boundary width.                  |
| `border_variable_width_enabled` | `bool` | `true`                            | If enabled, grain-boundary width varies locally along each boundary. Otherwise, the width is constant.                                     |
| `border_jaggedness` | `float` | `0.2`–`0.6`                       | Controls how irregular the grain boundaries are.                                                                                           |
| `border_perturbation_noise` | `float` | `0.2`–`0.8`                       | Controls the strength of boundary distortion.                                                                                              |
| `border_shading_enabled` | `bool` | `true`                            | Adds directional shading to grain boundaries.                                                                                              |
| `border_shading_intensity_range` | <nobr>`[int, int]`</nobr> | <nobr>`75`–`185`</nobr>           | Controls the brightness range used for shaded boundaries.                                                                                  |
| `border_flat_intensity` | `int` | <nobr>`100`–`150`</nobr>          | Boundary brightness used when boundary shading is disabled.                                                                                |
| `perlin_noise_enabled` | `bool` | `true`                            | Adds smooth brightness variation inside grains.                                                                                            |
| `perlin_noise_alpha_range` | <nobr>`[int, int]`</nobr> | <nobr>`2`–`6`</nobr>              | Controls the strength of smooth grain texture.                                                                                             |
| `perlin_noise_scales` | <nobr>`list[int]`</nobr> | <nobr>`[32, 64, 128, 256]`</nobr> | Controls the scale of smooth texture patterns. Smaller values create finer texture; larger values create broader, more cloud-like texture. |
| `gaussian_noise_enabled` | `bool` | `true`                            | Adds fine random noise inside grains.                                                                                                      |
| `gaussian_noise_alpha_range` | <nobr>`[int, int]`</nobr> | <nobr>`0`–`3`</nobr>              | Controls the strength of fine random noise.                                                                                                |
| `porous_texture_enabled` | `bool` | `true`                            | Adds small spot-like texture to the matrix.                                                                                                |
| `porous_texture_density_range` | <nobr>`[float, float]`</nobr> | <nobr>`0.05`–`0.18`</nobr>        | Controls how densly spot-like features are added.                                                                                   |
| `porous_texture_blur_range` | <nobr>`[float, float]`</nobr> | <nobr>`0.2`–`0.6`</nobr>          | Controls how soft the spot-like texture appears.                                                                                           |
| `porous_texture_intensity_range` | <nobr>`[float, float]`</nobr> | <nobr>`3`–`10`</nobr>             | Controls how visible the pore-like texture is.                                                                                             |
| `porous_texture_size_range` | <nobr>`[int, int]`</nobr> | <nobr>`0`–`2`</nobr>              | Controls the size of pore-like texture features.                                                                                           |


## etch_pits

The `etch_pits` section controls recessed surface feature artifacts.

| Parameter | Type | Typical values             | Description                                                                                                                                                                  |
|---|---|----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `num_range` | <nobr>`[int, int]`</nobr> | <nobr>`0`–`7`</nobr>       | Controls how many etch pits are added to the image.                                                                                                                          |
| `intensity_range` | <nobr>`[int, int]`</nobr> | <nobr>`100`–`135`</nobr>   | Controls the interior brightness of etch pits. Since etch pits are usually darker than the surrounding matrix, this range is typically lower than the grain intensity range. |
| `size_range` | <nobr>`[int, int]`</nobr> | <nobr>`5`–`25`</nobr>      | Controls the size of etch pits.                                                                                                                                              |
| `layer_offset_ratio` | `float` | <nobr>`0.15`–`0.4`</nobr>  | Controls how far the highlight and shadow layers are shifted from the pit center. Larger values create stronger shading.                                                     |
| `border_noise` | `float` | <nobr>`0.1`–`0.4`</nobr>   | Controls how irregular the etch pit shape is. Lower values create smoother pit borders.                                                                                      |
| `layer_offsets` | `object` | see below                  | Controls the relative position of each shading layer.                                                                                                                        |
| `layer_scales` | `object` | see below                  | Controls the relative size of each shading layer.                                                                                                                            |
| `shading_dark_shadow` | `int` | <nobr>`10`–`40`</nobr>     | Brightness of the darkest shadow layer.                                                                                                                                      |
| `shading_light_shadow` | `int` | <nobr>`50`–`80`</nobr>     | Brightness of the lighter shadow layer.                                                                                                                                      |
| `shading_highlight` | `int` | <nobr>`190`–`245`</nobr>   | Brightness of the highlight layer.                                                                                                                                           |
| `texture_density_range` | <nobr>`[float, float]`</nobr> | <nobr>`0.09`–`0.15`</nobr> | Controls how much fine texture appears inside etch pits.                                                                                                                     |
| `texture_blur_range` | <nobr>`[float, float]`</nobr> | <nobr>`0.1`–`0.4`</nobr>   | Controls the softness of the internal etch-pit texture.                                                                                                                      |
| `texture_intensity_range` | <nobr>`[float, float]`</nobr> | <nobr>`6`–`17`</nobr>      | Controls the contrast of the internal etch-pit texture.                                                                                                                      |
| `texture_size_range` | <nobr>`[int, int]`</nobr> | <nobr>`0`–`3`</nobr>       | Controls the size of internal texture spots.                                                                                                                                 |

| Parameter | Type | Typical values                                                                                                                                                 | Description                                                                                                                                                                             |
|---|---|----------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `layer_offsets` | <nobr>`dict[str, float]`</nobr> | <nobr>`highlight: 0.0`</nobr><br><nobr>`dark_shadow: -2.2 to -1.5`</nobr><br><nobr>`light_shadow: -0.8 to -0.4`</nobr><br><nobr>`base_fill: 0.1 to 0.2`</nobr> | Controls the relative position of each etch-pit layer. Negative values shift towards the light source; positive values shift them away. Larger absolute values create stronger shading. |
| `layer_scales` | <nobr>`dict[str, float]`</nobr> | <nobr>`highlight: 1.0`</nobr><br><nobr>`dark_shadow: 0.7 to 0.9`</nobr><br><nobr>`light_shadow: 0.6 to 0.8`</nobr><br><nobr>`base_fill: 0.6 to 0.7`</nobr>     | Controls the relative size of each etch-pit layer. Smaller value makes the layer more compact.                                                                                          |

Note: The `highlight` layer is the reference layer, so its offset is usually `0.0` and its scale is `1.0`.


## precipitates

The `precipitates` section controls dark particle-like features added on top of the matrix.

| Parameter | Type | Typical values                               | Description                                                                                                           |
|---|---|----------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| `num_range` | <nobr>`[int, int]`</nobr> | <nobr>`80`–`300`</nobr>                      | Controls how many precipitates are added.                                                                             |
| `intensity_range` | <nobr>`[int, int]`</nobr> | <nobr>`5`–`40`</nobr>                        | Controls precipitate base brightness.                                                                                 |
| `size_range` | <nobr>`[int, int]`</nobr> | <nobr>`1`–`35`</nobr>                        | Controls the precipitate size range.                                                                                  |
| `size_distribution` | <nobr>`list[int]`</nobr> | <nobr>`[4000, 55, 40, 30, 15, 10, 5]`</nobr> | Controls how often different precipitate sizes occur. Larger weights make the corresponding size group more frequent. |
| `emboss_enabled` | `bool` | `true`                                       | Adds shading to precipitates, giving them a raised appearance.                                                        |
| `emboss_intensity_range` | <nobr>`[int, int]`</nobr> | <nobr>`10`–`235`</nobr>                      | Controls the brightness range used for the shading.                             |



## effects

The `effects` section controls global image effects applied after the main structures are generated.

| Parameter | Type | Typical values              | Description                                                                                             |
|---|---|-----------------------------|---------------------------------------------------------------------------------------------------------|
| `corner_blur_enabled` | `bool` | `true`                      | Enables circular blur regions near the image corners.                                                   |
| `corner_blur_radius_range` | <nobr>`[int, int]`</nobr> | <nobr>`50`–`400`</nobr>     | Controls how far the corner blur extends into the image in pixels.                                      |
| `corner_blur_kernel_size_range` | <nobr>`[int, int]`</nobr> | <nobr>`0`–`10`</nobr>       | Controls the strength of the corner blur. Larger values produce stronger blur.                          |
| `bottom_blur_enabled` | `bool` | `true`                      | Enables rectangular blur region near the bottom of the image.                                           |
| `bottom_blur_height_range` | <nobr>`[int, int]`</nobr> | <nobr>`50`–`250`</nobr>     | Controls the height of the blurred bottom region in pixels.                                             |
| `bottom_blur_kernel_size_range` | <nobr>`[int, int]`</nobr> | <nobr>`1`–`8`</nobr>        | Controls the strength of the bottom blur. Larger values produce stronger blur.                          |
| `illumination_gradient_enabled` | `bool` | `true`                      | Adds a gradual brightness change across the image.                                                      |
| `illumination_gradient_strength_range` | <nobr>`[float, float]`</nobr> | <nobr>`0.06`–`0.17`</nobr>  | Controls the strength of the illumination gradient. Higher values create stronger brightness variation. |
| `illumination_gradient_angle_range` | <nobr>`[int, int]`</nobr> | <nobr>`-55` to `-25`</nobr> | Controls the direction of the illumination gradient in degrees.                                         |
| `noise_enabled` | `bool` | `true`                      | Enables full image noise.                                                                               |
| `noise_amplitude` | `float` | <nobr>`0.3`–`0.7`</nobr>    | Controls the strength of the image noise.                       |
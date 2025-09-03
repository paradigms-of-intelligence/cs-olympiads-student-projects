"""sRGB <-> XYB conversion routines using JAX for 0-1 float range."""

import jax.numpy as jnp
from jax.typing import ArrayLike
import jax
import numpy as np
from PIL import Image


# --- padding for entropy model computation

def shifted_border_pad(img):
    """
    Pad an NHWC image so that:

    - Top row: shifted right by 1 from original top row
    - Left col: shifted down by 1 from original left col
    - Bottom/right: zero padded
    """
    N, H, W, C = img.shape

    # Start with normal zero-padding
    padded = jnp.pad(img, ((0, 0), (1, 0), (1, 0), (0, 0)), mode="constant")

    # Top row: shifted right
    top_row_shifted = jnp.concatenate(
        [jnp.zeros((N, 1, 1, C)), img[:, 0:1, :-1, :]], axis=2
    )
    padded = padded.at[:, 0:1, :-1, :].set(top_row_shifted)

    # Left column: shifted down
    left_col_shifted = jnp.concatenate(
        [jnp.zeros((N, 1, 1, C)), img[:, :-1, 0:1, :]], axis=1
    )
    padded = padded.at[:, :-1, 0:1, :].set(left_col_shifted)

    return padded


# --- upscaling ---

UPSCALING_PREKERNELS = {  # used to generate final upscaling kernels
    2: [-0.01716200, -0.03452303, -0.04022174, -0.02921014, -0.00624645, 0.14111091, 0.28896755, 0.00278718,
        -0.01610267, 0.56661550, 0.03777607, -0.01986694, -0.03144731, -0.01185068, -0.00213539],
    4: [-0.02419067, -0.03491987, -0.03693351, -0.03094285, -0.00529785, -0.01663432, -0.03556863, -0.03888905,
        -0.03516850, -0.00989469, 0.23651958, 0.33392945, -0.01073543, -0.01313181, -0.03556694, 0.13048175,
        0.40103025, 0.03951150, -0.02077584, 0.46914198, -0.00209270, -0.01484589, -0.04064806, 0.18942530,
        0.56279892, 0.06674400, -0.02335494, -0.03551682, -0.00754830, -0.02267919, -0.02363578, 0.00315804,
        -0.03399098, -0.01359519, -0.00091653, -0.00335467, -0.01163294, -0.01610294, -0.00974088, -0.00191622,
        -0.01095446, -0.03198464, -0.04455121, -0.02799790, -0.00645912, 0.06390599, 0.22963888, 0.00630981,
        -0.01897349, 0.67537268, 0.08483369, -0.02534994, -0.02205197, -0.01667999, -0.00384443],
    8: [-0.02928613, -0.03706353, -0.03783812, -0.03324558, -0.00447632, -0.02519406, -0.03752601, -0.03901508,
        -0.03663285, -0.00646649, -0.02066407, -0.03838633, -0.04002101, -0.03900035, -0.00901973, -0.01626393,
        -0.03954148, -0.04046620, -0.03979621, -0.01224485, 0.29895328, 0.35757708, -0.02447552, -0.01081748,
        -0.04314594, 0.23903219, 0.41119301, -0.00573046, -0.01450239, -0.04246845, 0.17567618, 0.45220643,
        0.02287757, -0.01936783, -0.03583255, 0.11572472, 0.47416733, 0.06284440, -0.02685066, 0.42720050,
        -0.02248939, -0.01155273, -0.04562755, 0.28689496, 0.49093869, -0.00007891, -0.01545926, -0.04562659,
        0.21238920, 0.53980934, 0.03369474, -0.02070211, -0.03866988, 0.14229550, 0.56593398, 0.08045181,
        -0.02888298, -0.03680918, -0.00542229, -0.02920477, -0.02788574, -0.02118180, -0.03942402, -0.00775547,
        -0.02433614, -0.03193943, -0.02030828, -0.04044014, -0.01074016, -0.01930822, -0.03620399, -0.01974125,
        -0.03919545, -0.01456093, -0.00045072, -0.00360110, -0.01020207, -0.01231907, -0.00638988, -0.00071592,
        -0.00279122, -0.00957115, -0.01288327, -0.00730937, -0.00107783, -0.00210156, -0.00890705, -0.01317668,
        -0.00813895, -0.00153491, -0.02128481, -0.04173044, -0.04831487, -0.03293190, -0.00525260, -0.01720322,
        -0.04052736, -0.05045706, -0.03607317, -0.00738030, -0.01341764, -0.03965629, -0.05151616, -0.03814886,
        -0.01005819, 0.18968273, 0.33063684, -0.01300105, -0.01372950, -0.04017465, 0.13727832, 0.36402234,
        0.01027890, -0.01832107, -0.03365072, 0.08734506, 0.38194295, 0.04338228, -0.02525993, 0.56408126,
        0.00458352, -0.01648227, -0.04887868, 0.24585519, 0.62026135, 0.04314807, -0.02213737, -0.04158014,
        0.16637289, 0.65027023, 0.09621636, -0.03101388, -0.04082742, -0.00904519, -0.02790922, -0.02117818,
        0.00798662, -0.03995711, -0.01243427, -0.02231705, -0.02946266, 0.00992055, -0.03600283, -0.01684920,
        -0.00111684, -0.00411204, -0.01297130, -0.01723725, -0.01022545, -0.00165306, -0.00313110, -0.01218016,
        -0.01763266, -0.01125620, -0.00231663, -0.01374149, -0.03797620, -0.05142937, -0.03117307, -0.00581914,
        -0.01064003, -0.03608089, -0.05272168, -0.03375670, -0.00795586, 0.09628104, 0.27129991, -0.00353779,
        -0.01734151, -0.03153981, 0.05686230, 0.28500998, 0.02230594, -0.02374955, 0.68214326, 0.05018048,
        -0.02320852, -0.04383616, 0.18459474, 0.71517975, 0.10805613, -0.03263677, -0.03637639, -0.01394373,
        -0.02511203, -0.01728636, 0.05407331, -0.02867568, -0.01893131, -0.00240854, -0.00446511, -0.01636187,
        -0.02377053, -0.01522848, -0.00333334, -0.00819975, -0.02964169, -0.04499287, -0.02745350, -0.00612408,
        0.02727416, 0.19446600, 0.00159832, -0.02232473, 0.74982506, 0.11452620, -0.03348048, -0.01605681,
        -0.02070339, -0.00458223]}


def upscale(img, k):
    """
    Upscale image using sub-pixel convolution with k*k kernels.

    Args:
        img: input image with shape (H, W, C)
        k: the scale, must be one of 1, 2, 4 or 8

    Returns:
        upscaled image with shape (H*k, W*k, C)
    """

    if k == 1:
        return img
    kernel = generate_kernel(UPSCALING_PREKERNELS[k], k)
    H, W, C = img.shape

    # Add batch dimension
    img = img[None, ...]

    conv_results = []
    for i in range(k):
        for j in range(k):
            k_ij = kernel[i, j]  # (5,5)
            rhs = k_ij[:, :, None, None] * jnp.eye(C, dtype=img.dtype)[None, None, :, :]  # (5,5,C,C)

            conv_result = jax.lax.conv_general_dilated(
                img, rhs,
                window_strides=(1, 1),
                padding='SAME',
                dimension_numbers=('NHWC', 'HWIO', 'NHWC')
            )[0]  # Remove batch dimension
            conv_results.append(conv_result)

    # Stack conv results and reshape
    conv_stack = jnp.stack(conv_results, axis=0)  # (k*k, H, W, C)
    conv_reshaped = conv_stack.reshape(k, k, H, W, C)  # (k, k, H, W, C)

    # Rearrange and reshape to get final upscaled image
    upscaled = jnp.transpose(conv_reshaped, (2, 0, 3, 1, 4))  # (H, k, W, k, C)
    upscaled = upscaled.reshape(H * k, W * k, C)
    return upscaled


def generate_kernel(weights, N):
    """
    Generate a symmetric convolution kernel

    This function generates an N×N×5×5 convolution kernel from given weight array.
    It works through the following steps:
    1. First fill the upper-left region (getting values from weight array)
    2. Then fill the remaining parts through mirror symmetry, creating a fully symmetric kernel

    Args:
    - weights: 1D array containing kernel weights
    - N: kernel dimension parameter

    Returns:
    - kernel: 4D array with shape (N, N, 5, 5)
    """
    # Ensure JAX-friendly types
    weights = jnp.asarray(weights, dtype=jnp.float32)
    # Initialize kernel as all zeros
    kernel = jnp.zeros((N, N, 5, 5), dtype=jnp.float32)

    half_n = N // 2

    # Create index grids
    row_indices, col_indices = jnp.meshgrid(jnp.arange(5 * half_n), jnp.arange(5 * half_n), indexing='ij')

    # Calculate upper triangular indices
    min_coord = jnp.minimum(row_indices, col_indices)
    max_coord = jnp.maximum(row_indices, col_indices)

    # Calculate index in compressed upper triangular array
    triangular_index = 5 * half_n * min_coord - min_coord * (min_coord - 1) // 2 + max_coord - min_coord
    triangular_index = triangular_index.astype(jnp.int32)

    # Get weight values
    weight_values = weights[triangular_index]

    # Calculate quadrant indices
    block_row = row_indices // 5
    block_col = col_indices // 5
    inner_row = row_indices % 5
    inner_col = col_indices % 5

    # Fill upper-left region
    kernel = kernel.at[block_row, block_col, inner_row, inner_col].set(weight_values)

    # Upper-right (horizontal mirror)
    kernel = kernel.at[(2 * half_n - 1) - block_col, block_row, 4 - inner_col, inner_row].set(weight_values)

    # Lower-left (vertical mirror)
    kernel = kernel.at[block_col, (2 * half_n - 1) - block_row, inner_col, 4 - inner_row].set(weight_values)

    # Lower-right (horizontal and vertical mirror)
    kernel = kernel.at[(2 * half_n - 1) - block_col, (2 * half_n - 1) - block_row, 4 - inner_col, 4 - inner_row].set(
        weight_values)

    return kernel


# --- XYB <-> sRGB converting ---

OPSIN_MATRIX = jnp.asarray(
    [
        [0.300000011920929, 0.621999979019165, 0.078000001609325],
        [0.230000004172325, 0.691999971866608, 0.078000001609325],
        [0.243422687053680, 0.204767450690269, 0.551809847354889],
    ]
)

INV_OPSIN_MATRIX_T = jnp.linalg.inv(OPSIN_MATRIX).T

OPSIN_BIAS = jnp.asarray([0.003793073119596, 0.003793073119596, 0.003793073119596])
OPSIN_BIAS_CBRT = OPSIN_BIAS ** (1 / 3.0)


@jax.jit
def srgb_to_jxl_xyb(srgb01: ArrayLike) -> ArrayLike:
    """Converts an sRGB 0-1 image of shape [..., 3] to the XYB colorspace."""
    # Linearize sRGB using the standard formula.
    lrgb = jnp.where(
        srgb01 < 0.04045,
        srgb01 / 12.92,
        ((jnp.where(srgb01 < 0.04045, 0.0, srgb01) + 0.055) / 1.055) ** 2.4,
    )

    # Convert to opsin space.
    opsin = jnp.matmul(lrgb, OPSIN_MATRIX.T) + OPSIN_BIAS

    # Apply cube root and subtract bias.
    opsin_cbrt = jnp.cbrt(opsin) - OPSIN_BIAS_CBRT

    # Construct the final XYB image.
    x = 0.5 * (opsin_cbrt[..., 0] - opsin_cbrt[..., 1])
    y = 0.5 * (opsin_cbrt[..., 0] + opsin_cbrt[..., 1])
    b = opsin_cbrt[..., 2]

    return jnp.stack((x, y, b), axis=-1)


def jxl_xyb_to_srgb(xyb: ArrayLike) -> ArrayLike:
    """Converts an XYB image of shape [..., 3] to a sRGB 0-1 image."""
    # Deconstruct XYB to get the cube-rooted opsin values.
    opsin_cbrt = (
        jnp.stack(
            (
                xyb[..., 0] + xyb[..., 1],
                xyb[..., 1] - xyb[..., 0],
                xyb[..., 2],
            ),
            axis=-1,
        )
    )

    # Reverse the cube root and add bias.
    opsin = (opsin_cbrt + OPSIN_BIAS_CBRT) ** 3

    # Convert from opsin space back to linear RGB.
    lrgb = jnp.matmul(opsin - OPSIN_BIAS, INV_OPSIN_MATRIX_T)
    # Apply sRGB gamma correction.
    srgb01 = jnp.where(
        lrgb < 0.0031308,
        12.92 * lrgb,
        1.055 * jnp.where(lrgb < 0.0031308, 0.0031308, lrgb) ** (1 / 2.4) - 0.055,
    )

    # Return the 0-1 range sRGB image.
    return srgb01

def add_y_to_b(img):
    x = img[..., 0, :, :]
    y = img[..., 1, :, :]
    b = img[..., 2, :, :]
    return jnp.stack([x, y, b + y], axis=-3)

def subtract_y_from_b(img):
    x = img[..., 0, :, :]
    y = img[..., 1, :, :]
    b = img[..., 2, :, :]
    return jnp.stack([x, y, b - y], axis=-3)


# --- DCT <-> XYB converting ---

def generate_dct_matrix(N=8):
    k = jnp.arange(N)[:, None]  # (8,1)
    n = jnp.arange(N)[None, :]  # (1,8)
    dct_mat = jnp.cos(jnp.pi * (2 * n + 1) * k / (2 * N))
    dct_mat = dct_mat * jnp.sqrt(2 / N)
    dct_mat = dct_mat.at[0].set(dct_mat[0] / jnp.sqrt(2))  # first row special case
    return dct_mat


DCT_MATRIX_8X8 = generate_dct_matrix(8)


def block_dct_2d(blocks):
    """
    Perform 2d DCT on (..., 8, 8) blocks.
    Works for any leading dims, including batch & channel.
    """
    return jnp.matmul(DCT_MATRIX_8X8, jnp.matmul(blocks, DCT_MATRIX_8X8.T))


def block_idct_2d(blocks: jnp.ndarray) -> jnp.ndarray:
    """
    Apply 2D inverse DCT (III) on (..., 8, 8) blocks.
    Works for any leading dims, including batch & channel.
    """
    # Since DCT is orthogonal, inverse is just DCT.T
    return jnp.matmul(DCT_MATRIX_8X8.T, jnp.matmul(blocks, DCT_MATRIX_8X8))


@jax.jit
def xyb_to_dct(img: ArrayLike) -> ArrayLike:
    """
    Converts an XYB image of shape [H, W, 3] to its DCT representation with shape [H/8, W/8, C, 8, 8].
    H and W must be divisible by 8.
    """
    assert img.shape[0] % 8 == 0 and img.shape[1] % 8 == 0
    # Reshape into (H//8, 8, W//8, 8, C)
    H, W, C = img.shape
    blocks = img.reshape(H // 8, 8, W // 8, 8, C)

    # Move channel to front for easier vmapping: (H//8, W//8, C, 8, 8)
    blocks = blocks.transpose(0, 2, 4, 1, 3)

    # Apply DCT to each channel-block
    dct_blocks = block_dct_2d(blocks)
    return dct_blocks


@jax.jit
def dct_to_xyb(dct_blocks: ArrayLike) -> ArrayLike:
    """
    Converts an XYB image in DCT representation with shape [H/8, W/8, C, 8, 8] to a XYB image of shape [H, W, 3].
    """
    H_blocks, W_blocks, C, _, _ = dct_blocks.shape
    # Apply IDCT to each channel-block
    img_blocks = block_idct_2d(dct_blocks)  # shape (H//8, W//8, C, 8, 8)

    # Rearrange to (H//8, 8, W//8, 8, C)
    img_blocks = img_blocks.transpose(0, 3, 1, 4, 2)

    # Merge block dimensions back to (H, W, C)
    H, W = H_blocks * 8, W_blocks * 8
    img = img_blocks.reshape(H, W, C)

    return img

# --- saving ---

def save_xyb(
        xyb: ArrayLike, save_path: str = "flattened_xyb.txt"
):
    header = f"{xyb.shape[1]}\n{xyb.shape[0]}\n"
    np.savetxt(save_path, np.array(xyb).flatten(), fmt="%.6f", header=header, comments="")


def save_rgb_image(img, path):
    img = np.array((img * 255).astype(jnp.uint8))
    img = Image.fromarray(img)
    img.save(path)
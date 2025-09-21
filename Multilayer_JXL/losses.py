import jax.numpy as jnp
import jax
from jax import lax
from image_converter_utils import shifted_border_pad, subtract_y_from_b, upscale
import functools
import faster_wasserstein_vgg16 as fastW

from optimizer_values import OptimizerValues

# precompute kernel for context loss
base_kernel = jnp.array(
    [[0.0, 0.5, 0.0], [0.5, -1.0, 0.0], [0.0, 0.0, 0.0]], dtype=jnp.float32
)

# Reshape to HWIO with singleton I/O channels
kernel = base_kernel[:, :, jnp.newaxis, jnp.newaxis]  # shape (Hk, Wk, 1, 1)
# Broadcast kernel across all channels
kernel = jnp.broadcast_to(kernel, (base_kernel.shape[0], base_kernel.shape[1], 1, 3))
weight_kernel = jnp.zeros(shape=(8, 8), )
weight_kernel.at[0, 0].set(0)


@jax.jit
def compression_loss_context(img, xyb_multiplier_context):
    r"""
    Computes the compression loss as the average over all values v

    .. math::
       \ln(1 + xyb\_multiplier\cdot |v - \\frac{1}{2}(u+l) |)

    where u is the value of the cell above (on the same channel) and l is the one on the left.

    Parameters
    ----------
    img
        the image for which the los should be computed in [H, W, C] format
    xyb_multiplier_context
        a weighting of shape [C] for the different channels
    """
    img_batched = jnp.expand_dims(img, 0)
    img_convolved = lax.conv_general_dilated(
        shifted_border_pad(img_batched),
        kernel,
        (1, 1),
        "VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        feature_group_count=3,
    )
    scales = jnp.array(xyb_multiplier_context, dtype=jnp.float32).reshape(1, 1, 1, 3)
    img_convoluted = jnp.log(1 + jnp.abs(img_convolved) * scales)
    return jnp.mean(img_convoluted)


@jax.jit
def compression_loss_context_multi(img, xyb_multiplier_context):
    r"""
    Computes the compression loss as the average over all values v

    .. math::
       \ln(1 + xyb\_multiplier\cdot |v - \\frac{1}{2}(u+l) |)

    where u is the value of the cell above (on the same channel) and l is the one on the left.
    Does this over all layers (which should be smaller by a factor of 2) and weights the result according to their size

    Parameters
    ----------
    img
        the layers of images for which the loss should be computed in [H, W, C] format
    xyb_multiplier_context
        a weighting of shape [C] for the different channels
    """
    return sum(
        [
            compression_loss_context(img[i], xyb_multiplier_context) * (0.25 ** i)
            for i in range(len(img))
        ]
    )


def compression_loss_coefficients(dct_blocks, xyb_multiplier_context):
    dct_blocks_non_average = jnp.abs(dct_blocks) * weight_kernel
    scales = jnp.array(xyb_multiplier_context, dtype=jnp.float32).reshape(1, 1, 3, 1, 1)
    dct_blocks_recomputed = jnp.log(1 + dct_blocks_non_average * scales)
    return jnp.mean(dct_blocks_recomputed)


@jax.jit
def compression_loss_DCT(dct_blocks, xyb_multiplier_dct, xyb_multiplier_context, gamma=63):
    """
    Computes the compression loss as the average over all non-left-upper-corner dct coefficients + context loss of upper left corners

    Parameters
    ----------
    dct_blocks
        the image for which the loss should be computed in [H / 8, W / 8, C, 8, 8] format
    xyb_multiplier_dct
        a weighting of shape [C] for the different channels which is used for the coefficients
    xyb_multiplier_context
         a weighting of shape [C] for the different channels which is used for the context loss
    gamma
        weighting of context loss against coefficient loss
    """
    dct_blocks = subtract_y_from_b(dct_blocks)
    compact = dct_blocks[..., 0, 0]
    compact_loss = compression_loss_context(
        compact, xyb_multiplier_context
    )
    return (compact_loss + gamma * compression_loss_coefficients(dct_blocks, xyb_multiplier_dct)) / (gamma + 1)


@jax.jit
def compression_loss_DCT_multi(dct_blocks, xyb_multiplier_dct, xyb_multiplier_context, gamma):
    """
    Computes the compression loss as the average over all non-left-upper-corner dct coefficients + context loss of upper left corners
    Does this over all layers (which should be smaller by a factor of 2) and weights the result according to their size

    Parameters
    ----------
    dct_blocks
        the image for which the loss should be computed in [H / 8, W / 8, C, 8, 8] format
    xyb_multiplier_dct
        a weighting of shape [C] for the different channels which is used for the coefficients
    xyb_multiplier_context
         a weighting of shape [C] for the different channels which is used for the context loss
    gamma
        weighting of context loss against coefficient loss
    """
    return [
        compression_loss_DCT(dct_blocks[i], xyb_multiplier_dct, xyb_multiplier_context, gamma)
        * (0.25 ** i)
        for i in range(len(dct_blocks))
    ]


@jax.jit
def l2_loss(img, target, l2_rgb_multiplier):
    return jnp.mean((l2_rgb_multiplier * (img - target)) ** 2)


@jax.jit
def wasserstein_loss(img, target_features, log2_sigma):
    return fastW.vgg16_wasserstein_distortion_precompute(
        img, target_features, log2_sigma
    )


@functools.partial(jax.jit, static_argnames=['use_l2'])
def loss_fn(
        optimizer_values: OptimizerValues,
        target,
        target_features,
        log2_sigma,
        lambd,
        gamma,
        xyb_multiplier_dct,
        xyb_multiplier_context,
        l2_rgb_multiplier,
        use_l2,
):
    """
    calculate the loss as a combination of compression loss and wasserstein loss / l2 loss

    :param optimizer_values: an instance of OptimizerValues class which represents the layers of images
    :param target: the target image
    :param target_features: the precomputed features of the target image
    :param log2_sigma: the log_sigma filter in the wasserstein loss
    :param lambd: weight of compression loss in the compression loss (the weighted average is taken, with wasserstein/l2 loss having weight 1). Somewhere between 0 and 10 is recommended.
    :param gamma: weight of coefficients loss in the compression loss (the weighted average is taken, with context loss having weight 1). Somewhere between 16 and 255 is recommended.
    :param xyb_multiplier_dct: a list of 3 values for x, y, and b respectively which are multiplied to the layers when computing the coefficient loss
    :param xyb_multiplier_context: a list of 3 values for x, y, and b respectively which are multiplied to the layers when computing the context loss
    :param l2_rgb_multiplier: a number which should be multiplied to the difference for the MSE term in training
    :param use_l2: whether to use l2 instead of wasserstein loss
    :return: the total loss and a tuple with individual losses (wasserstein/l2, list of compression losses)
    """
    compression_losses = compression_loss_DCT_multi(
        optimizer_values.convert_to_xyb_dct(), xyb_multiplier_dct, xyb_multiplier_context, gamma
    )
    compression_loss_combined = sum(compression_losses)

    if use_l2:
        ws_loss = l2_loss(optimizer_values.combine_to_rgb(), target, l2_rgb_multiplier)
    else:
        # Compute the Wasserstein loss between the target and the generated image
        ws_loss = wasserstein_loss(optimizer_values.combine_to_rgb(), target_features, log2_sigma)
    return (ws_loss + lambd * compression_loss_combined) / (1 + lambd), (
        ws_loss,
        compression_losses,
    )

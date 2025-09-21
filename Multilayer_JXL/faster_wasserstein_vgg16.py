import jax
import jax.numpy as jnp
from codex.loss import pretrained_features, multi_wasserstein_distortion, load_vgg16_model

Array = jax.Array
ArrayLike = jax.typing.ArrayLike
load_vgg16_model()


def get_features(image, num_scales=3):
    """
    Precompute features of an image in [H, W, C] format
    Parameters
    ----------
    image
        input image in [H, W, C] format
    num_scales
        The number of scales of the image the features should be computed on. The image
        will be downsampled ``num_scales - 1`` times and VGG features computed on the
        original image plus the downsampled versions. The concatenated list of all
        features will be used to compute the distortion.

    Returns
    -------
    A list of feature arrays of format (channel, height, width).
    """
    image = jnp.transpose(image, (2, 0, 1))
    return pretrained_features.compute_vgg16_features(
        image, num_scales=num_scales
    )


@jax.jit
def vgg16_wasserstein_distortion_precompute(
        image_a: ArrayLike,
        features_b: ArrayLike,
        log2_sigma: Array,
        *,
        num_scales: int = 3,
        num_levels: int = 5,
        sqrt_grad_limit: float = 1e6,
) -> Array:
    """VGG-16 Wasserstein Distortion between two images.

    Parameters
    ----------
    image_a
        First image to be compared in format ``(height, width, 3)``. Note that this is
        not the same as in the codex module code!
    features_b
        precomputed features for the second image.
    log2_sigma
        Array, shape ``(height, width)``. The base two logarithm of the sigma map, which
        indicates the amount of summarization in each location. Doesn't have to have the
        same shape as the image arrays.
    num_scales
        The number of scales of the image the features should be computed on. The image
        will be downsampled ``num_scales - 1`` times and VGG features computed on the
        original image plus the downsampled versions. The concatenated list of all
        features will be used to compute the distortion. Should be the same as in
    num_levels
        The number of multi-scale levels of the feature statistics to compute. Must be
        greater or equal to the maximum of `log2_sigma`.
    sqrt_grad_limit
        Upper limit for the gradient of the square root applied to the empirical feature
        variance estimates, for numerical stability.

    Returns
    -------
    Array
        Scalar distortion value.

    Notes
    -----
    This is the distortion loss function used in [1]_. Please cite the paper if you use
    this code for scientific or research work.

    .. [1] J. Ballé, L. Versari, E. Dupont, H. Kim, M. Bauer: "Good, Cheap,
       and Fast: Overfitted Image Compression with Wasserstein Distortion," 2025 IEEE/CVF
       Conf. on Computer Vision and Pattern Recognition (CVPR), 2025.
       https://arxiv.org/abs/2412.00505
    """

    image_a = jnp.transpose(image_a, (2, 0, 1))
    features_a = pretrained_features.compute_vgg16_features(
        image_a, num_scales=num_scales
    )

    return multi_wasserstein_distortion(
        features_a,
        features_b,
        log2_sigma,
        num_levels=num_levels,
        sqrt_grad_limit=sqrt_grad_limit,
        return_intermediates=False,
    )

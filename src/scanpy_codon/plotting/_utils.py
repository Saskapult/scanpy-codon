from python import warnings


def _deprecated_scale(
    density_norm,
    scale,
    # default: DensityNorm | Empty = _empty,
):
    # if scale is _empty:
    if scale is None:
        return density_norm
    # if density_norm != default:
    #     msg = "can’t specify both `scale` and `density_norm`"
    #     raise ValueError(msg)
    msg = "`scale` is deprecated, use `density_norm` instead"
    warnings.warn(msg)
    return scale

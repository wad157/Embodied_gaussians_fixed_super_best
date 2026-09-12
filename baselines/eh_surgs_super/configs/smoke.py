"""Short interface smoke test; never use for reported metrics."""

ModelParams = dict(extra_mark="super-smoke", camera_extent=10, white_background=False)
OptimizationParams = dict(
    coarse_iterations=0,
    iterations=10,
    densify_from_iter=100,
    densify_until_iter=0,
    pruning_from_iter=100,
    opacity_reset_interval=100,
    position_lr_max_steps=10,
)
ModelHiddenParams = dict(curve_num=20, ch_num=8, init_param=0.01)

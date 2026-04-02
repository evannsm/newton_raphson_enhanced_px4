# Newton-Raphson Flow for PX4-ROS2 Deployment with Enhanced Error Terms

A ROS 2 enhanced Newton-Raphson v2 trajectory tracking controller for quadrotors. Builds on [nr_standard](https://github.com/evannsm/nr_standard) by incorporating state Jacobian terms and reference rate derivatives into the control law for improved dynamic tracking performance.

## What's Enhanced

Compared to the standard Newton-Raphson controller:

- **Reference rate feed-forward** — incorporates trajectory derivative information into the error term for better anticipation of dynamic maneuvers
- **State Jacobian coupling** — uses both output and state Jacobians (`get_jac_pred_x_uinv()`) to account for state-dependent dynamics
- **Higher control gains** — tuned for tighter tracking: `ALPHA = [30, 40, 40, 40]` vs `[20, 30, 30, 30]` in standard

## Key Features

- **Enhanced NR control law** — augmented error formulation with reference rate and state Jacobian terms
- **Integral CBF safety constraints** — optional barrier functions for input constraint enforcement
- **JAX JIT-compiled** — all control computations are JIT-compiled for real-time performance
- **PX4 integration** — publishes attitude setpoints and offboard commands via `px4_msgs`
- **Structured logging** — optional CSV logging via ROS2Logger

## Controller Profiles

The package now exposes two explicit controller profiles via `--nr-profile`:

| Profile | Lookahead | Predictor | Iterations | `alpha` | Integral action |
|-----------|-----------|-----------|------------|---------|-----------------|
| `baseline` | `1.2 s` | ZOH | `1` | `[30, 40, 40, 40]` | Disabled |
| `workshop` | `0.8 s` | FOH | `2` | `[30, 40, 40, 40]` | Enabled with bounded anti-windup |

`baseline` preserves the original enhanced control law after workspace
integration. `workshop` adds the same structural fixes validated on the
standard Python controller: shorter lookahead, first-order-hold prediction,
bounded integral error injection, and two damped Newton updates per 100 Hz
cycle.

Measured Python comparison on April 1, 2026 (`fig8_horz`, headless SITL):

| Profile | Position RMSE (m) | Compute time (ms) |
|-----------|-------------------|-------------------|
| `baseline` | `0.35749240937686216` | `0.3395175933837423` |
| `workshop` | `0.23281274706265945` | `0.32021449162404647` |

That is a `34.88%` RMSE reduction with a slight `5.69%` compute-time decrease
on this run.

For the full workspace-level writeup and exact run commands, see
`docs/qmd/newton_raphson_workshop_profiles.qmd`.

## Control Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `ALPHA` | `[30, 40, 40, 40]` | Baseline control gains `[x, y, z, yaw]` |
| `USE_CBF` | `True` | Enable integral Control Barrier Functions |

## Usage

```bash
source install/setup.bash

# Fly a figure-8 in simulation
ros2 run newton_raphson_enhanced_px4 run_node --platform sim --trajectory fig8_horz

# Fly the validated workshop profile
ros2 run newton_raphson_enhanced_px4 run_node --platform sim --trajectory fig8_horz --nr-profile workshop --log

# Fly a helix on hardware with logging
ros2 run newton_raphson_enhanced_px4 run_node --platform hw --trajectory helix --log
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--platform {sim,hw}` | Target platform (required) |
| `--trajectory {hover,yaw_only,circle_horz,...}` | Trajectory type (required) |
| `--hover-mode {1..8}` | Hover sub-mode (1-4 for hardware) |
| `--log` | Enable CSV data logging |
| `--log-file NAME` | Custom log filename |
| `--double-speed` | 2x trajectory speed |
| `--short` | Short variant (fig8_vert) |
| `--spin` | Enable yaw rotation |
| `--flight-period SEC` | Custom flight duration |
| `--nr-profile {baseline,workshop}` | Select the enhanced NR profile |

## Workspace Integration

`newton_raphson_enhanced_px4` is now integrated into the shared
`quad_platforms`, `quad_trajectories`, and `src/workspace_tools/fly_pipeline.py`
infrastructure used by the rest of this workspace.

That means it now:

- uses the shared platform mass and force-to-throttle conversion path
- uses the shared trajectory registry and context handling
- writes analysis-compatible CSV logs into `src/data_analysis/log_files/`
- can be run through the same headless SITL pipeline as the other controllers

## Dependencies

- [quad_trajectories](https://github.com/evannsm/quad_trajectories) — trajectory definitions
- [quad_platforms](https://github.com/evannsm/quad_platforms) — platform abstraction
- [ROS2Logger](https://github.com/evannsm/ROS2Logger) — experiment logging
- [px4_msgs](https://github.com/PX4/px4_msgs) — PX4 ROS 2 message definitions
- JAX / jaxlib

## Package Structure

```
newton_raphson_enhanced_px4/
├── newton_raphson_enhanced_px4/
│   ├── run_node.py              # CLI entry point and argument parsing
│   └── ros2px4_node.py          # ROS 2 node (subscriptions, publishers, control loop)
└── newton_raphson_enhanced_px4_utils/
    ├── controller/
    │   ├── nr_enhanced.py       # Enhanced NR control law
    │   └── nr_utils.py          # Dynamics, Jacobians, CBF, state Jacobian
    ├── px4_utils/               # PX4 interface and flight phase management
    ├── transformations/         # Yaw adjustment utilities
    ├── main_utils.py            # Helper functions
    └── jax_utils.py             # JAX configuration
```

## Installation

```bash
# Inside a ROS 2 workspace src/ directory
git clone git@github.com:evannsm/newton_raphson_enhanced_px4.git
cd .. && colcon build --symlink-install
```

## License

MIT

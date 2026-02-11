# newton_raphson_enhanced_px4

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

## Control Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `ALPHA` | `[30, 40, 40, 40]` | Control gains `[x, y, z, yaw]` |
| `USE_CBF` | `True` | Enable integral Control Barrier Functions |

## Usage

```bash
source install/setup.bash

# Fly a figure-8 in simulation
ros2 run newton_raphson_enhanced_px4 run_node --platform sim --trajectory fig8_horz

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

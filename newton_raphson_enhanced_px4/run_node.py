"""Entry point for the Newton-Raphson Enhanced control ROS2 node."""

import argparse
import os
import traceback

import rclpy

from ros2_logger import Logger  # type: ignore
from pyJoules.handler.csv_handler import CSVHandler
from quad_platforms import PlatformType
from quad_trajectories import TrajectoryType

from .ros2px4_node import OffboardControl


def create_parser():
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Newton-Raphson Enhanced Offboard Control for Quadrotor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        """ + "==" * 60 + """
        Example usage:
        ros2 run newton_raphson_enhanced_px4 run_node --platform sim --trajectory helix --double-speed --spin --log
        ros2 run newton_raphson_enhanced_px4 run_node --platform sim --trajectory fig8_horz --nr-profile workshop --log
        """ + "==" * 60 + """
        """,
    )

    parser.add_argument(
        "--platform",
        type=PlatformType,
        choices=list(PlatformType),
        required=True,
        help="Platform type to use.",
    )
    parser.add_argument(
        "--trajectory",
        type=TrajectoryType,
        choices=list(TrajectoryType),
        required=True,
        help="Trajectory type to execute.",
    )
    parser.add_argument(
        "--hover-mode",
        type=int,
        choices=range(1, 9),
        help="Hover mode (required when --trajectory=hover).",
    )
    parser.add_argument("--log", action="store_true", help="Enable CSV data logging.")
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Custom log file name (without extension).",
    )
    parser.add_argument("--pyjoules", action="store_true", help="Enable PyJoules energy monitoring.")
    parser.add_argument("--double-speed", action="store_true", help="Use double speed (2x) trajectories.")
    parser.add_argument("--short", action="store_true", help="Use short fig8_vert trajectory variant.")
    parser.add_argument("--spin", action="store_true", help="Enable spin for circle_horz and helix.")
    parser.add_argument(
        "--flight-period",
        type=float,
        default=None,
        help="Set custom flight period in seconds.",
    )
    parser.add_argument("--ff", action="store_true", help="Enable feedforward for the trajectory.")
    parser.add_argument(
        "--nr-profile",
        choices=["baseline", "workshop"],
        default="baseline",
        help="Enhanced Newton-Raphson controller profile to run.",
    )
    return parser


def ensure_csv(filename: str) -> str:
    """Return filename that ends with exactly one '.csv' (case-insensitive)."""
    filename = filename.strip()
    if filename.lower().endswith(".csv"):
        return filename[:-4] + ".csv"
    return filename + ".csv"


def generate_log_filename(args) -> str:
    """Generate auto log filename based on configuration."""
    parts = [args.platform.value, "nr_enhanced", args.trajectory.value]
    if args.ff:
        parts.append("ff")
    if args.nr_profile != "baseline":
        parts.append(args.nr_profile)
    parts.append("2x" if args.double_speed else "1x")
    if args.short:
        parts.append("short")
    if args.spin:
        parts.append("spin")
    parts.append("py")
    return "_".join(parts)


def validate_args(args, parser: argparse.ArgumentParser) -> None:
    """Validate command-line arguments."""
    if args.trajectory == TrajectoryType.HOVER:
        if args.hover_mode is None:
            parser.error("--hover-mode is required when --trajectory=hover")
        if args.platform == PlatformType.HARDWARE and args.hover_mode not in range(1, 5):
            parser.error("--hover-mode must be 1-4 for --platform=hw")
        if args.platform == PlatformType.SIM and args.hover_mode not in range(1, 9):
            parser.error("--hover-mode must be 1-8 for --platform=sim")
    elif args.hover_mode is not None:
        parser.error("--hover-mode is only valid when --trajectory=hover")

    if args.log_file is not None and not args.log:
        parser.error("--log-file requires --log to be enabled")


def _logger_base_path(file_path: str, pkg_name: str) -> str:
    """Return the base path needed for ros2_logger path resolution."""
    path = os.path.abspath(file_path)
    parts = path.split(os.sep)
    for i, part in enumerate(parts[:-1]):
        if part in ("install", "src", "build") and parts[i + 1] == pkg_name:
            return os.sep.join(parts[:i + 2] + [pkg_name])
    return os.path.dirname(path)


def main():
    """Main entry point for the executable."""
    parser = create_parser()
    args = parser.parse_args()
    validate_args(args, parser)

    platform = args.platform
    trajectory = args.trajectory
    hover_mode = args.hover_mode
    logging_enabled = args.log
    pyjoules = args.pyjoules
    double_speed = args.double_speed
    short = args.short
    spin = args.spin
    flight_period = args.flight_period
    feedforward = args.ff
    nr_profile = args.nr_profile
    base_path = _logger_base_path(__file__, "newton_raphson_enhanced_px4")

    if logging_enabled:
        log_file_stem = args.log_file if args.log_file is not None else generate_log_filename(args)
        log_file = ensure_csv(log_file_stem)
    else:
        log_file = None

    print("\n" + "=" * 60)
    print("Newton-Raphson Enhanced Offboard Control Configuration")
    print("=" * 60)
    print(f"Platform:      {platform.value.upper()}")
    print("Controller:    ENHANCED")
    print(f"Trajectory:    {trajectory.value.upper()}")
    print(f"Hover Mode:    {hover_mode if hover_mode is not None else 'N/A'}")
    print(f"Speed:         {'Double (2x)' if double_speed else 'Regular (1x)'}")
    print(f"Short:         {'Enabled (fig8_vert)' if short else 'Disabled'}")
    print(f"Flight Period: {flight_period if flight_period is not None else 60.0 if platform == PlatformType.HARDWARE else 30.0} seconds")
    print(f"Spin:          {'Enabled (circle_horz, helix)' if spin else 'Disabled'}")
    print(f"Feedforward:   {'Enabled' if feedforward else 'Disabled'}")
    print(f"NR Profile:    {nr_profile}")
    print(f"Data Logging:  {'Enabled' if logging_enabled else 'Disabled'}")
    if logging_enabled:
        print(f"Log File:      {log_file}")
    print(f"PyJoules:      {'Enabled' if pyjoules else 'Disabled'}")
    print("=" * 60 + "\n")

    rclpy.init(args=None)
    offboard_control_node = OffboardControl(
        platform_type=platform,
        trajectory=trajectory,
        hover_mode=hover_mode,
        double_speed=double_speed,
        short=short,
        spin=spin,
        pyjoules=pyjoules,
        csv_handler=CSVHandler(ensure_csv(generate_log_filename(args) + "_energy") if not log_file else log_file, base_path) if pyjoules else None,
        logging_enabled=logging_enabled,
        flight_period_=flight_period,
        feedforward=feedforward,
        nr_profile=nr_profile,
    )

    logger = None

    def shutdown_logging(*_args):
        print("\nShutting down, triggering logging...")
        if logger and logging_enabled:
            logger.log(offboard_control_node)
        offboard_control_node.destroy_node()
        rclpy.shutdown()

    try:
        print("\nInitializing Offboard Control Node")
        if logging_enabled:
            logger = Logger(log_file, base_path)
        rclpy.spin(offboard_control_node)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt (Ctrl+C)")
    except Exception as exc:
        print(f"\nError: {exc}")
        traceback.print_exc()
    finally:
        if pyjoules and offboard_control_node.csv_handler:
            print(f"Saving PyJoules energy data to {offboard_control_node.csv_handler._filename}.")
            offboard_control_node.csv_handler.save_data()
        if logging_enabled:
            print("Saving log data...")
        shutdown_logging()
        print("\nNode shut down.")


if __name__ == "__main__":
    main()

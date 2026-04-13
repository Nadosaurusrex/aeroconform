"""CLI entry point for starting the AeroConform MCP server.

Launches the live monitoring loop and exposes the MCP API.
"""

from __future__ import annotations

import argparse

import numpy as np
import structlog

from aeroconform.config import AeroConformConfig
from aeroconform.data.preprocessing import NormStats
from aeroconform.inference.mcp_server import AeroConformMCPServer
from aeroconform.models.conformal import AdaptiveConformalDetector
from aeroconform.models.pipeline import AeroConformPipeline
from aeroconform.models.trajectory_model import TrajectoryTransformer
from aeroconform.inference.live_monitor import LiveAirspaceMonitor
from aeroconform.utils.logging import setup_logging

logger = structlog.get_logger(__name__)


def create_pipeline(config: AeroConformConfig, device: str = "cpu") -> AeroConformPipeline:
    """Create an AeroConform pipeline with a model and calibrated detector.

    Args:
        config: Configuration.
        device: Inference device.

    Returns:
        Configured pipeline.
    """
    model = TrajectoryTransformer.from_config(config)
    model.eval()

    detector = AdaptiveConformalDetector.from_config(config)
    # Initialize with synthetic calibration scores
    rng = np.random.default_rng(42)
    detector.calibrate(rng.exponential(scale=50.0, size=config.cal_window))

    norm_stats = NormStats(median=[0.0] * 6, iqr=[1.0] * 6)

    return AeroConformPipeline(
        model=model,
        detector=detector,
        config=config,
        norm_stats=norm_stats,
        device=device,
    )


def main() -> None:
    """Entry point for the serve CLI."""
    parser = argparse.ArgumentParser(description="AeroConform MCP Server")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Server host",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Server port",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Inference device",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
    )

    args = parser.parse_args()
    setup_logging(level=args.log_level)

    config = AeroConformConfig(
        serve_host=args.host,
        serve_port=args.port,
    )

    pipeline = create_pipeline(config, device=args.device)
    monitor = LiveAirspaceMonitor(pipeline=pipeline, config=config)
    server = AeroConformMCPServer(monitor=monitor, config=config)

    tools = server.get_tools()
    logger.info(
        "mcp_server_ready",
        n_tools=len(tools),
        tool_names=[t["name"] for t in tools],
        host=args.host,
        port=args.port,
    )

    # Print tool definitions for MCP registration
    for tool in tools:
        logger.info("mcp_tool", **tool)


if __name__ == "__main__":
    main()

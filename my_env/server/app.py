# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the protein folding OpenEnv environment."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import ProteinAction, ProteinObservation
    from .my_env_environment import ProteinFoldingEnvironment
except ModuleNotFoundError:
    from models import ProteinAction, ProteinObservation
    from server.my_env_environment import ProteinFoldingEnvironment


app = create_app(
    ProteinFoldingEnvironment,
    ProteinAction,
    ProteinObservation,
    env_name="my_env",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Run the protein folding server directly."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)

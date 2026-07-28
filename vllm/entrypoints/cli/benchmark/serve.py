# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse

from vllm.benchmarks.serve import add_cli_args
from vllm.benchmarks.serve import main as python_main
from vllm.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser

logger = init_logger(__name__)



class BenchmarkServingSubcommand(BenchmarkSubcommandBase):
    """The `serve` subcommand for `vllm bench`."""

    name = "serve"
    help = "Benchmark the online serving throughput."

    @classmethod
    def add_cli_args(cls, parser: FlexibleArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        python_main(args)

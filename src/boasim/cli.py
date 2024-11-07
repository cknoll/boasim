"""
Command line interface
"""

import argparse
from ipydex import IPS, activate_ips_on_exception
from . import core
from . import __version__

activate_ips_on_exception()


def main():

    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "cmd", help=f"main command (not yet defined)"
    # )
    parser.add_argument(
        "--version", help=f"print version and exit", action="store_true"
    )

    args = parser.parse_args()

    if args.version:
        print(__version__)

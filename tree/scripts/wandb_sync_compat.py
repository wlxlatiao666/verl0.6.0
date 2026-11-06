"""Run wandb sync with nullable server capabilities and error reporting."""

import sys
import threading

from wandb.cli.cli import cli
from wandb.sdk.internal.internal_api import Api
from wandb.sdk.internal.sender import SendManager


def main():
    original_gql = Api.gql
    original_error = SendManager._handle_error
    original_hook = threading.excepthook
    errors = []

    def compatible_gql(self, *args, **kwargs):
        result = original_gql(self, *args, **kwargs)
        if isinstance(result, dict) and "ServerSettingsType" in result:
            if result["ServerSettingsType"] is None:
                result = dict(result)
                result["ServerSettingsType"] = {"fields": []}
        return result

    def report_error(self, record, error, run):
        errors.append(error.message)
        print(f"W&B sync failed: {error.message}", file=sys.stderr)
        return original_error(self, record, error, run)

    def report_thread_error(args):
        errors.append(str(args.exc_value))
        original_hook(args)

    Api.gql = compatible_gql
    SendManager._handle_error = report_error
    threading.excepthook = report_thread_error
    try:
        cli.main(args=["sync", *sys.argv[1:]], standalone_mode=False)
    finally:
        Api.gql = original_gql
        SendManager._handle_error = original_error
        threading.excepthook = original_hook
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

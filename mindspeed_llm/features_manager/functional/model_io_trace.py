from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class ModelIOTraceFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__(feature_name="model-io-trace", optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument(
            "--model-io-trace",
            action="store_true",
            help="Enable full-model module I/O and batch-input tracing.",
        )
        group.add_argument(
            "--model-io-trace-config-path",
            type=str,
            default=None,
            help="Path to a custom model I/O trace config.json file.",
        )
        group.add_argument(
            "--model-io-trace-output-path",
            type=str,
            default=None,
            help="Directory to save model I/O trace results.",
        )

    def validate_args(self, args):
        super().validate_args(args)
        if args.model_io_trace and not args.model_io_trace_output_path:
            raise ValueError("`--model-io-trace-output-path` must be specified when model I/O tracing is enabled.")

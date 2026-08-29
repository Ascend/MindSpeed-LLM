from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class MsProbeFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__(feature_name="msprobe", optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument("--msprobe", action="store_true", help="Enable msProbe precision data collection.")
        group.add_argument(
            "--msprobe-config-path",
            type=str,
            default=None,
            help="Path to a custom msProbe config.json file.",
        )

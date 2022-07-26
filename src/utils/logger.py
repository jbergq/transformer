from datetime import datetime
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    def __init__(self, name="unnamed_experiment", log_dir="./experiments"):
        self.name = self._generate_experiment_name(name)

        self.log_dir = Path(log_dir)
        self.experiment_dir = self.log_dir / self.name
        self.experiment_dir.mkdir(exist_ok=True, parents=True)

        self.summary_writer = SummaryWriter(self.experiment_dir / "tensorboard")

    def _generate_experiment_name(self, name):
        datetime_str = datetime.today().strftime("%Y%m%d_%H%M%S")

        return f"{datetime_str}_{name}"

    def log_params(self, params):
        for key, val in params.items():
            self.summary_writer.add_scalar(key, val)

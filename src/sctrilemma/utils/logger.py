import logging
import logging.config
import os
from typing import Literal

from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table


def setup_logger(
        log_file: str = "./logs/training.log",
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
        to_console: bool = True,
) -> None:
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)

    handlers = {
        # File Handler (using RotatingFileHandler)
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "file_formatter",
            "filename": log_file,
            "maxBytes": 1024 * 1024 * 5,  # 5 MB
            "backupCount": 5,  # Keep maximum 5 log files
            "encoding": "utf-8",
        },
    }

    root_handlers = ["file"]

    if to_console:
        handlers["console"] = {
            "class": "rich.logging.RichHandler",
            "formatter": "console_formatter",
            "level": log_level,
            "rich_tracebacks": True,
            "show_path": False,
        }
        root_handlers.append("console")

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "file_formatter": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "console_formatter": {
                "format": "[%(levelname)s] %(message)s",
            },
        },
        "handlers": handlers,
        # Set Up Root logger
        "root": {
            "level": log_level,
            "handlers": root_handlers,
        },
    }

    logging.config.dictConfig(logging_config)
    logging.info("Logger has been configured successfully.")

def log_config(cfg: DictConfig, logger: logging.Logger) -> None:
    """Log configuration details to file and render a rich view in the console.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object to display and persist.
    logger : logging.Logger
        Logger used for writing the plain-text configuration to file.
    """
    # --- 1. Summary Table ---
    summary_table = Table(
        title="🚀 Key Experiment Parameters",
        show_header=True,
        header_style="bold magenta",
    )
    summary_table.add_column("Group", style="dim", width=20)
    summary_table.add_column("Parameter", style="dim", width=25)
    summary_table.add_column("Value", style="cyan")

    # Handle missing keys gracefully
    task_name = cfg.get("experiment_name", "N/A")

    # Extract model name from components.model._target_
    model_name = "N/A"
    if "components" in cfg and "model" in cfg.components:
        model_target = cfg.components.model.get("_target_", "")
        if model_target:
            # Extract class name from full path (e.g., "newgenex.models.sctranscriptomer.SCTranscriptomerModel" -> "SCTranscriptomerModel")
            model_name = model_target.split(".")[-1]

    seed_value = cfg.get("seed", "N/A")

    summary_table.add_row("Task", "Name", str(task_name))
    summary_table.add_row("Model", "Name", str(model_name))
    # ... (Add Something)
    summary_table.add_row("Global", "Seed", str(seed_value))

    # Full
    config_yaml = OmegaConf.to_yaml(cfg)
    _full_config_syntax = Syntax(config_yaml, "yaml", theme="monokai", line_numbers=True)  # noqa: F841

    # --- 2. Display table in terminal only ---
    display_console = Console()
    display_console.print(summary_table)

    # --- 3. Log detailed config to file ---
    logger.info("--- Experiment Configuration ---")
    logger.info("✨ Full Configuration Tree")
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("---------------------------------")

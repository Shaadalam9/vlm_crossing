"""Contains various function used throughout this project.

Adapted from common.py in crowd-dataset/crowd-city (Pavlo Bazilinskyy).

Attributes:
    root_dir (str): Folder of this repository.
    cache_dir (str): Folder for cached intermediate results.
    log_dir (str): Folder for log files.
    output_dir (str): Folder for tracking CSVs, metrics and figures.
    logger (CustomLogger): Module logger.
"""

from email.message import EmailMessage
from typing import Any, Dict, Tuple
import json
import os
import pickle
import smtplib
import subprocess
import sys

from custom_logger import CustomLogger


root_dir = os.path.dirname(__file__)
cache_dir = os.path.join(root_dir, '_cache')
log_dir = os.path.join(root_dir, '_logs')
output_dir = os.path.join(root_dir, '_output')

logger = CustomLogger(__name__)  # use custom logger


# ---------------------------------------------------------------------
# Configuration cache
# ---------------------------------------------------------------------
#
# Configuration does not change during a run, so config and default.config are
# read and validated once per process. Cache entries are process local:
# multiprocessing workers load their own configuration once when they start.
_CONFIG_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}


def _load_config_once(
    config_file_name: str = 'config',
    config_default_file_name: str = 'default.config',
):
    """Load, validate, and cache configuration once per Python process."""
    cache_key = (config_file_name, config_default_file_name)

    if cache_key in _CONFIG_CACHE:
        return _CONFIG_CACHE[cache_key]

    config_path = os.path.join(root_dir, config_file_name)
    default_path = os.path.join(root_dir, config_default_file_name)

    try:
        with open(config_path) as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.error(f"Config file {config_file_name} not found. Create it from {config_default_file_name}.")
        return None
    except json.decoder.JSONDecodeError:
        logger.error(
            "Config file badly formatted. Please update based on default.config."
        )
        return None

    try:
        with open(default_path) as f:
            default = json.load(f)
    except FileNotFoundError:
        logger.error(
            f"Default config file {config_default_file_name} not found."
        )
        return None
    except json.decoder.JSONDecodeError:
        logger.error(
            "Default config file badly formatted. "
            "Please update based on default.config."
        )
        return None

    # Every value is taken from config. default.config only defines which
    # settings must exist; its values are never used, so a setting missing
    # from config stops the run instead of silently falling back.
    missing_keys = [key for key in default if key not in config]
    if missing_keys:
        logger.error(
            f"Config file is missing {len(missing_keys)} variable(s): "
            f"{', '.join(missing_keys)}. Add them to {config_file_name}, "
            f"using {config_default_file_name} for the structure."
        )
        return None

    _CONFIG_CACHE[cache_key] = config
    return config


def clear_config_cache() -> None:
    """Clear cached configuration so the next lookup reloads it from disk."""
    _CONFIG_CACHE.clear()


def get_secrets(entry_name: str, secret_file_name: str = 'secret') -> Dict[str, str]:
    """
    Open the secrets file and return the requested entry.

    Args:
        entry_name (str): Key in the secrets file.
        secret_file_name (str, optional): Secrets filename.

    Returns:
        Dict[str, str]: Value stored under entry_name.
    """
    with open(os.path.join(root_dir, secret_file_name)) as f:
        return json.load(f)[entry_name]


def get_configs(
    entry_name: str,
    config_file_name: str = 'config',
    config_default_file_name: str = 'default.config',
):
    """
    Return a configuration value from the process-local cached configuration.

    The first lookup reads config and checks it against default.config, which
    only lists the settings that must exist. Every value comes from config.
    Subsequent lookups use the in-memory dictionary and perform no file I/O.

    Args:
        entry_name (str): Configuration key.
        config_file_name (str, optional): Main config filename.
        config_default_file_name (str, optional): Default config filename.

    Returns:
        Any: Value stored under entry_name.
    """
    content = _load_config_once(
        config_file_name=config_file_name,
        config_default_file_name=config_default_file_name,
    )
    if content is None:
        sys.exit()

    return content[entry_name]


def check_config(
    config_file_name: str = 'config',
    config_default_file_name: str = 'default.config',
):
    """
    Check whether config is valid.

    Validation is cached, so repeated calls do not reread either JSON file.

    Args:
        config_file_name (str, optional): Main config filename.
        config_default_file_name (str, optional): Default config filename.

    Returns:
        bool: True when the configuration is valid.
    """
    return (
        _load_config_once(
            config_file_name=config_file_name,
            config_default_file_name=config_default_file_name,
        )
        is not None
    )


def resolve_path(path: str) -> str:
    """
    Return an absolute path, resolving relative paths against the repository root.

    Args:
        path (str): Absolute path, or path relative to root_dir.

    Returns:
        str: Absolute path.
    """
    path = os.path.expanduser(str(path))
    return path if os.path.isabs(path) else os.path.join(root_dir, path)


def get_output_dir() -> str:
    """Return the configured output folder, created if missing."""
    path = resolve_path(get_configs("output"))
    os.makedirs(path, exist_ok=True)
    return path


def save_to_p(file, data, description_data='data'):
    """
    Save data to a pickle file in the cache folder.

    Args:
        file (str): Pickle file (*.p or *.pkl).
        data (tuple): Data tuple.
        description_data (str, optional): Description of data.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, file)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    logger.info('Saved ' + description_data + ' to pickle file {}.', file)


def load_from_p(file, description_data='data'):
    """Load data from a pickle file in the cache folder.

    Args:
        file (str): Pickle file (*.p or *.pkl).
        description_data (str, optional): Description of data.

    Returns:
        tuple: data tuple.
    """
    path = os.path.join(cache_dir, file)
    with open(path, 'rb') as f:
        data = pickle.load(f)
    logger.info('Loaded ' + description_data + ' from pickle file {}.', file)
    return data


# Pull changes from repository
def git_pull():
    """
    git pull changes from the repo with a terminal command.
    """
    try:
        logger.info("Attempting to pull latest changes from git repository...")
        result = subprocess.run(["git", "pull"], capture_output=True, text=True, check=True)
        logger.info(f"Git pull successful:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Git pull failed with error:\n{e.stderr}")


# Send email with certain message
def send_email(subject, content, sender, recipients):
    """
    Send email with certain message from sender to recipients.

    Args:
        subject (str): Subject of email.
        content (str): Email body.
        sender (str): Email address to send from.
        recipients (list): Email addresses to receive the message.
    """
    msg = EmailMessage()
    msg.set_content(content)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)

    try:
        with smtplib.SMTP_SSL(get_secrets("email_smtp"), 465) as smtp:
            smtp.login(get_secrets("email_account"), get_secrets("email_password"))
            smtp.send_message(msg)
            logger.info(f"Sent email to: {recipients}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")


os.makedirs(log_dir, exist_ok=True)

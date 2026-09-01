import copy
import json
import logging
import tomllib
from pathlib import Path
from utils.paths import Paths

logger = logging.getLogger(__name__)

#: Distinguishes "no default given" from a default of None, so an optional key whose legitimate
#: value is None still reads as absent.
_MISSING = object()


def parse_toml_value(text: str):
    """Parse an override's right-hand side with TOML's own literal rules."""
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


def to_toml_literal(value) -> str:
    """Render a Python value as TOML source, the inverse of :func:`parse_toml_value`."""
    if isinstance(value, bool):  # before int: bool is an int subclass
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(to_toml_literal(item) for item in value) + "]"
    raise TypeError(f"Cannot render {type(value).__name__} as a TOML literal: {value!r}")


class Config:
    def __init__(self, config_path):
        self._config_data = None
        self._overrides = []

        with open(config_path, 'rb') as f:
            self._config_data = tomllib.load(f)

    def get_train_paths(self):
        return self._config_data["train_path"]

    def get_val_paths(self):
        return self._config_data["val_path"]

    def get_test_paths(self) -> dict | None:
        return self._config_data.get("test_path", None)

    def get_training_param(self, param: str, default=_MISSING):
        """Read a key from [training_params], raising KeyError unless a default is given."""
        if default is _MISSING:
            return self._config_data["training_params"][param]
        return self._config_data.get("training_params", {}).get(param, default)

    def get_model_config(self):
        return self._config_data["model"]

    def get_param(self, param_name: str, default=None):
        return self._config_data.get(param_name, default)

    def set_by_path(self, dotted_key: str, value) -> None:
        """Set ``a.b.c`` to ``value``, creating intermediate tables that do not exist yet.

        Creation matters for regime switches: pointing [train_path] at a source the base config
        never mentioned must work without editing the file.
        """
        parts = self._split_key(dotted_key)
        node = self._config_data
        for depth, part in enumerate(parts[:-1]):
            child = node.get(part)
            if child is None:
                child = {}
                node[part] = child
            elif not isinstance(child, dict):
                traversed = ".".join(parts[:depth + 1])
                raise ValueError(
                    f"Cannot set {dotted_key!r}: {traversed} is a {type(child).__name__}, not a table"
                )
            node = child
        node[parts[-1]] = value

    def unset_by_path(self, dotted_key: str) -> bool:
        """Remove ``a.b.c``. A key that is not there is a warning, not an error.

        A sweep arm may unset a dataset that another arm never enabled, and that should not abort
        a chain of runs half way through.
        """
        parts = self._split_key(dotted_key)
        node = self._config_data
        for part in parts[:-1]:
            node = node.get(part) if isinstance(node, dict) else None
            if not isinstance(node, dict):
                logger.warning(f"unset {dotted_key}: {part} is not a table, nothing removed")
                return False
        if parts[-1] not in node:
            logger.warning(f"unset {dotted_key}: key not present, nothing removed")
            return False
        del node[parts[-1]]
        return True

    def apply_overrides(self, sets=None, unsets=None) -> list:
        for key in unsets or []:
            key = key.strip()
            self.unset_by_path(key)
            self._overrides.append(f"unset {key}")
            logger.info(f"Override: unset {key}")

        for item in sets or []:
            key, separator, raw_value = item.partition('=')
            if not separator:
                raise ValueError(f"Override {item!r} is not of the form key=value")
            key = key.strip()
            value = parse_toml_value(raw_value)
            self.set_by_path(key, value)
            self._overrides.append(f"{key}={to_toml_literal(value)}")
            logger.info(f"Override: {key} = {value!r}")

        return list(self._overrides)

    def get_overrides(self) -> list:
        return list(self._overrides)

    def as_dict(self) -> dict:
        return copy.deepcopy(self._config_data)

    @staticmethod
    def _split_key(dotted_key: str) -> list:
        parts = [part.strip() for part in dotted_key.split('.')]
        if not parts or any(not part for part in parts):
            raise ValueError(f"Malformed config key: {dotted_key!r}")
        return parts


def load_config(config_path):
    config_path = Path(config_path)
    if not Path.is_absolute(config_path):
        config_path = Paths.config_dir() / config_path

    with open(config_path, 'rb') as f:
       model_config = tomllib.load(f)
    return model_config

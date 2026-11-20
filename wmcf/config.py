"""Small YAML config helper with dotted ``--set`` overrides."""

from __future__ import annotations

from pathlib import Path

import yaml


class DotDict(dict):
    """dict with attribute access, used for nested config sections."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def _wrap(obj):
    if isinstance(obj, dict):
        return DotDict({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _plain(obj):
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_plain(v) for v in obj]
    return obj


def _set_nested(cfg: DotDict, dotted: str, value) -> None:
    keys = dotted.split(".")
    node = cfg
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = DotDict()
        node = node[key]
    node[keys[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> DotDict:
    """Load ``path`` and apply ``key=value`` overrides such as ``data.confound=false``."""

    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    cfg = _wrap(raw)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        _set_nested(cfg, key.strip(), _wrap(yaml.safe_load(value)))
    return cfg


def save_config(cfg, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(_plain(cfg), handle, allow_unicode=True, sort_keys=False)

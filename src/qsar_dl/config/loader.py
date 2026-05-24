"""Project-wide configuration loading for QSAR experiments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Any


class ConfigError(ValueError):
    """Raised when a configuration file is missing, invalid, or incomplete."""


_PLACEHOLDER_PATTERN = re.compile(r"\$\{([^}]+)\}")

_BASE_DEFAULTS: dict[str, Any] = {
    "project": {
        "root": "",
    },
    "experiment": {
        "id": "",
        "seed": 20260524,
        "output_dir": "",
    },
    "data": {
        "clean_sqlite": "outputs/databases/ecotox_clean.sqlite",
        "modeling_table": "outputs/tables/modeling_toxicity_long.parquet",
    },
    "target": {
        "column": "target_ptox",
        "unit_family": "water_mg_l",
    },
    "runtime": {
        "device": "auto",
        "num_workers": 0,
    },
    "logging": {
        "level": "INFO",
        "save_resolved_config": True,
    },
}

_REQUIRED_KEYS = (
    ("project", "root"),
    ("experiment", "id"),
    ("experiment", "seed"),
    ("experiment", "output_dir"),
    ("data", "clean_sqlite"),
    ("data", "modeling_table"),
    ("target", "column"),
    ("target", "unit_family"),
    ("runtime", "device"),
    ("runtime", "num_workers"),
    ("logging", "level"),
    ("logging", "save_resolved_config"),
)

_PATH_KEY_NAMES = {
    "root",
    "output_dir",
    "clean_sqlite",
    "modeling_table",
}

_PATH_KEY_SUFFIXES = (
    "_dir",
    "_path",
    "_file",
    "_sqlite",
    "_parquet",
    "_csv",
    "_tsv",
    "_xlsx",
    "_xls",
    "_yaml",
    "_yml",
    "_json",
    "_table",
)

_PATH_VALUE_SUFFIXES = (
    ".sqlite",
    ".db",
    ".parquet",
    ".csv",
    ".tsv",
    ".xlsx",
    ".xls",
    ".yaml",
    ".yml",
    ".json",
    ".pkl",
    ".joblib",
    ".pt",
    ".pth",
    ".png",
    ".pdf",
    ".svg",
    ".tif",
    ".tiff",
)


def _yaml_module() -> Any:
    try:
        import yaml  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ConfigError(
            "PyYAML is required to read configuration files. "
            "Install it with `conda install pyyaml` or `pip install PyYAML`."
        ) from exc
    return yaml


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file with UTF-8 encoding."""

    yaml_path = Path(path).expanduser()
    if not yaml_path.exists():
        raise ConfigError(f"YAML config file not found: {yaml_path.resolve(strict=False)}")
    if not yaml_path.is_file():
        raise ConfigError(f"YAML config path is not a file: {yaml_path.resolve(strict=False)}")

    yaml = _yaml_module()
    try:
        with yaml_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except yaml.YAMLError as exc:  # type: ignore[attr-defined]
        raise ConfigError(f"Invalid YAML in {yaml_path.resolve(strict=False)}: {exc}") from exc

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(
            "YAML config must contain a mapping at the top level: "
            f"{yaml_path.resolve(strict=False)}"
        )
    return loaded


def resolve_config(config_path: Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load includes, merge configs, resolve project-relative paths, validate keys."""

    config_file = _absolute_path(Path(config_path))
    experiment_config = load_yaml(config_file)
    project_root = get_project_root(config_file)

    merged = deepcopy(_BASE_DEFAULTS)
    merged["project"]["root"] = project_root.as_posix()

    for include_name, include_path in _iter_include_paths(experiment_config, project_root):
        included_config = load_yaml(include_path)
        if "includes" in included_config:
            raise ConfigError(
                "Nested includes are not supported; remove `includes` from "
                f"{include_path.as_posix()} referenced as `{include_name}`."
            )
        merged = _deep_merge(merged, included_config)

    experiment_body = deepcopy(experiment_config)
    experiment_body.pop("includes", None)
    merged = _deep_merge(merged, experiment_body)

    if overrides:
        merged = _deep_merge(merged, _expand_dotted_keys(overrides))

    merged["project"]["root"] = project_root.as_posix()
    interpolated = _interpolate_config(merged)
    resolved = _resolve_paths(interpolated, project_root)
    resolved["project"]["root"] = project_root.as_posix()
    _validate_resolved_config(resolved)
    _ensure_no_placeholders(resolved)
    return resolved


def write_resolved_config(config: dict[str, Any], output_path: Path) -> None:
    """Write the exact runtime config to YAML."""

    _ensure_no_placeholders(config)
    yaml = _yaml_module()
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(
            config,
            handle,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )


def get_project_root(start: Path | None = None) -> Path:
    """Return repository root containing .git or pyproject.toml."""

    candidate = Path.cwd() if start is None else Path(start).expanduser()
    candidate = candidate.resolve(strict=False)
    if candidate.is_file():
        candidate = candidate.parent

    for path in (candidate, *candidate.parents):
        if (path / ".git").exists() or (path / "pyproject.toml").is_file():
            return path

    raise ConfigError(
        "Could not find project root containing `.git` or `pyproject.toml` "
        f"from start path: {candidate.as_posix()}"
    )


def _absolute_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve(strict=False)
    return (Path.cwd() / expanded).resolve(strict=False)


def _iter_include_paths(
    config: dict[str, Any], project_root: Path
) -> list[tuple[str, Path]]:
    includes = config.get("includes", {})
    if includes is None:
        return []
    if not isinstance(includes, dict):
        raise ConfigError("`includes` must be a mapping from include name to YAML path.")

    paths: list[tuple[str, Path]] = []
    for include_name, include_value in includes.items():
        if not isinstance(include_name, str):
            raise ConfigError("All `includes` keys must be strings.")
        if not isinstance(include_value, (str, Path)):
            raise ConfigError(
                f"Include `{include_name}` must be a YAML file path, "
                f"got {type(include_value).__name__}."
            )
        include_path = Path(include_value).expanduser()
        if not include_path.is_absolute():
            include_path = project_root / include_path
        paths.append((include_name, include_path.resolve(strict=False)))
    return paths


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _expand_dotted_keys(overrides: dict[str, Any]) -> dict[str, Any]:
    expanded: dict[str, Any] = {}
    for key, value in overrides.items():
        if isinstance(key, str) and "." in key:
            _set_nested(expanded, key.split("."), value)
        elif isinstance(value, dict):
            expanded[key] = _expand_dotted_keys(value)
        else:
            expanded[key] = deepcopy(value)
    return expanded


def _set_nested(target: dict[str, Any], parts: list[str], value: Any) -> None:
    current = target
    for part in parts[:-1]:
        next_value = current.setdefault(part, {})
        if not isinstance(next_value, dict):
            raise ConfigError(f"Override path `{'.'.join(parts)}` conflicts with a scalar value.")
        current = next_value
    current[parts[-1]] = deepcopy(value)


def _interpolate_config(config: dict[str, Any]) -> dict[str, Any]:
    current = deepcopy(config)
    for _ in range(10):
        updated, changed = _interpolate_value(current, current)
        current = updated
        if not changed:
            return current
    raise ConfigError("Config interpolation did not converge after 10 passes.")


def _interpolate_value(value: Any, root: dict[str, Any]) -> tuple[Any, bool]:
    if isinstance(value, dict):
        changed = False
        result: dict[str, Any] = {}
        for key, item in value.items():
            resolved_item, item_changed = _interpolate_value(item, root)
            result[key] = resolved_item
            changed = changed or item_changed
        return result, changed
    if isinstance(value, list):
        changed = False
        result_list: list[Any] = []
        for item in value:
            resolved_item, item_changed = _interpolate_value(item, root)
            result_list.append(resolved_item)
            changed = changed or item_changed
        return result_list, changed
    if not isinstance(value, str):
        return value, False

    matches = list(_PLACEHOLDER_PATTERN.finditer(value))
    if not matches:
        return value, False

    if len(matches) == 1 and matches[0].span() == (0, len(value)):
        return deepcopy(_lookup_placeholder(root, matches[0].group(1))), True

    resolved = value
    for match in matches:
        placeholder_value = _lookup_placeholder(root, match.group(1))
        if isinstance(placeholder_value, (dict, list)):
            raise ConfigError(
                f"Interpolation `${{{match.group(1)}}}` cannot be embedded because it "
                "resolves to a non-scalar value."
            )
        resolved = resolved.replace(match.group(0), str(placeholder_value))
    return resolved, True


def _lookup_placeholder(root: dict[str, Any], reference: str) -> Any:
    current: Any = root
    for part in reference.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ConfigError(f"Unknown config interpolation reference: `${{{reference}}}`")
        current = current[part]
    return current


def _resolve_paths(value: Any, project_root: Path, key_path: tuple[str, ...] = ()) -> Any:
    if isinstance(value, dict):
        return {
            key: _resolve_paths(item, project_root, (*key_path, str(key)))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_resolve_paths(item, project_root, key_path) for item in value]
    if isinstance(value, Path):
        value = value.as_posix()
    if isinstance(value, str) and _should_resolve_path(key_path, value):
        return _resolve_path_string(value, project_root)
    return value


def _should_resolve_path(key_path: tuple[str, ...], value: str) -> bool:
    if not value or "://" in value or "${" in value:
        return False
    key = key_path[-1].lower() if key_path else ""
    lower_value = value.lower()
    if key in _PATH_KEY_NAMES or key.endswith(_PATH_KEY_SUFFIXES):
        return True
    if lower_value.endswith(_PATH_VALUE_SUFFIXES):
        return True
    if value.startswith(("~/", "~\\", "./", ".\\", "../", "..\\")):
        return True
    return Path(value).is_absolute()


def _resolve_path_string(value: str, project_root: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve(strict=False).as_posix()


def _validate_resolved_config(config: dict[str, Any]) -> None:
    for key_path in _REQUIRED_KEYS:
        try:
            value = _lookup_required(config, key_path)
        except KeyError as exc:
            raise ConfigError(f"Missing required config key: {'.'.join(key_path)}") from exc
        if value is None:
            raise ConfigError(f"Required config key cannot be null: {'.'.join(key_path)}")

    if not str(config["experiment"]["id"]).strip():
        raise ConfigError("`experiment.id` must be set in the experiment config or overrides.")
    if not str(config["experiment"]["output_dir"]).strip():
        raise ConfigError(
            "`experiment.output_dir` must be set in the experiment config or overrides."
        )
    if not isinstance(config["experiment"]["seed"], int):
        raise ConfigError("`experiment.seed` must be an integer for reproducible data splits.")
    if not isinstance(config["runtime"]["num_workers"], int):
        raise ConfigError("`runtime.num_workers` must be an integer.")
    if not isinstance(config["logging"]["save_resolved_config"], bool):
        raise ConfigError("`logging.save_resolved_config` must be a boolean.")


def _lookup_required(config: dict[str, Any], key_path: tuple[str, ...]) -> Any:
    current: Any = config
    for key in key_path:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(key)
        current = current[key]
    return current


def _ensure_no_placeholders(value: Any) -> None:
    path = _find_placeholder_path(value)
    if path is not None:
        raise ConfigError(
            "Unresolved config interpolation placeholder remains at "
            f"`{path[0]}`: {path[1]}"
        )


def _find_placeholder_path(value: Any, key_path: tuple[str, ...] = ()) -> tuple[str, str] | None:
    if isinstance(value, dict):
        for key, item in value.items():
            found = _find_placeholder_path(item, (*key_path, str(key)))
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = _find_placeholder_path(item, (*key_path, str(index)))
            if found is not None:
                return found
    elif isinstance(value, str) and "${" in value:
        return ".".join(key_path), value
    return None

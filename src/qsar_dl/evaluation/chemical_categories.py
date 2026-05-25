"""Rule-based pollutant category assignment for evaluation splits."""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


CATEGORY_ORDER = (
    "PFAS",
    "organophosphates",
    "PAHs",
    "TPHs",
    "metals_metalloids",
    "surfactants",
    "dyes",
    "pesticides",
    "pharmaceuticals_personal_care_products",
    "phenols",
    "chlorinated_organics",
    "solvents_vocs",
    "other_unknown",
)

DEFAULT_CONFIG: dict[str, Any] = {
    "chemical_id_column": "chemical_id",
    "identifier_columns": ["chemical_id", "casrn", "cas_number", "dtxsid", "smiles"],
    "name_columns": [
        "chemical_name",
        "preferred_name",
        "name",
        "common_name",
        "substance_name",
        "cas_name",
    ],
    "smiles_column": "smiles",
    "existing_category_columns": [
        "chemical_category",
        "category",
        "chemical_class",
        "chemical_group",
        "ecotox_group",
        "pollutant_class",
        "use_class",
    ],
    "output_category_column": "chemical_category",
    "unknown_category": "other_unknown",
}

_CATEGORY_PATTERNS: dict[str, tuple[str, ...]] = {
    "PFAS": (
        r"\bPFAS\b",
        r"\bPFOS\b",
        r"\bPFOA\b",
        r"\bPFNA\b",
        r"\bPFHxS\b",
        r"\bPFBS\b",
        r"\bGenX\b",
        r"\bperfluoro",
        r"\bpolyfluoro",
        r"\bfluorotelomer",
        r"\bfluorinated\s+surfactant",
    ),
    "organophosphates": (
        r"\borganophosphate\b",
        r"\borganophosphorus\b",
        r"\bphosphate\s+ester\b",
        r"\bphosphonate\b",
        r"\bchlorpyrifos\b",
        r"\bdiazinon\b",
        r"\bmalathion\b",
        r"\bparathion\b",
        r"\bdichlorvos\b",
        r"\bdimethoate\b",
        r"\bacephate\b",
        r"\bglyphosate\b",
        r"\btriphenyl\s+phosphate\b",
        r"\bTCEP\b",
        r"\bTDCPP\b",
    ),
    "PAHs": (
        r"\bPAH(s)?\b",
        r"\bpolycyclic\s+aromatic\s+hydrocarbon",
        r"\bnaphthalene\b",
        r"\banthracene\b",
        r"\bphenanthrene\b",
        r"\bfluoranthene\b",
        r"\bpyrene\b",
        r"\bchrysene\b",
        r"\bacenaphthene\b",
        r"\bfluorene\b",
        r"\bbenzo\[?a\]?\s*pyrene\b",
        r"\bbenz\[?a\]?\s*anthracene\b",
        r"\bdibenz\[?a,h\]?\s*anthracene\b",
        r"\bindeno\[?1,2,3-cd\]?\s*pyrene\b",
    ),
    "TPHs": (
        r"\bTPH(s)?\b",
        r"\btotal\s+petroleum\s+hydrocarbon",
        r"\bpetroleum\s+hydrocarbon",
        r"\bcrude\s+oil\b",
        r"\bfuel\s+oil\b",
        r"\bdiesel\b",
        r"\bgasoline\b",
        r"\bkerosene\b",
        r"\bmineral\s+oil\b",
        r"\blubricating\s+oil\b",
        r"\bmotor\s+oil\b",
        r"\boil\s+range\s+organics\b",
    ),
    "metals_metalloids": (
        r"\barsenic\b",
        r"\bcadmium\b",
        r"\blead\b",
        r"\bmercury\b",
        r"\bchromium\b",
        r"\bcopper\b",
        r"\bnickel\b",
        r"\bzinc\b",
        r"\bselenium\b",
        r"\bantimony\b",
        r"\bbarium\b",
        r"\bsilver\b",
        r"\baluminum\b",
        r"\baluminium\b",
        r"\bcobalt\b",
        r"\bmanganese\b",
        r"\biron\b",
        r"\bvanadium\b",
        r"\bmolybdenum\b",
        r"\bthallium\b",
        r"\buranium\b",
        r"\btin\b",
        r"\btitanium\b",
        r"\bboron\b",
        r"\bmetals?\b",
        r"\bmetalloids?\b",
    ),
    "surfactants": (
        r"\bsurfactant\b",
        r"\bdetergent\b",
        r"\bSDS\b",
        r"\bsodium\s+dodecyl\s+sulfate\b",
        r"\blinear\s+alkylbenzene\s+sulfonate\b",
        r"\bLAS\b",
        r"\balkyl\s+sulfate\b",
        r"\balkylbenzene\s+sulfonate\b",
        r"\bquaternary\s+ammonium\b",
        r"\bbenzalkonium\b",
        r"\bcetyltrimethylammonium\b",
        r"\bnonylphenol\s+ethoxylate\b",
    ),
    "dyes": (
        r"\bdyes?\b",
        r"\bazo\s+dye\b",
        r"\banthraquinone\s+dye\b",
        r"\bmethylene\s+blue\b",
        r"\bcrystal\s+violet\b",
        r"\bmalachite\s+green\b",
        r"\bmethyl\s+orange\b",
        r"\brhodamine\b",
        r"\breactive\s+black\b",
        r"\bcongo\s+red\b",
    ),
    "pesticides": (
        r"\bpesticide\b",
        r"\bherbicide\b",
        r"\binsecticide\b",
        r"\bfungicide\b",
        r"\bbiocide\b",
        r"\batrazine\b",
        r"\bsimazine\b",
        r"\bcarbaryl\b",
        r"\bcarbofuran\b",
        r"\bpermethrin\b",
        r"\bcypermethrin\b",
        r"\bimidacloprid\b",
        r"\bDDT\b",
        r"\bendosulfan\b",
        r"\blindane\b",
        r"\balachlor\b",
        r"\bmetolachlor\b",
        r"\b2,4-D\b",
    ),
    "pharmaceuticals_personal_care_products": (
        r"\bPPCP(s)?\b",
        r"\bpharmaceutical\b",
        r"\bpersonal\s+care\s+product",
        r"\bdrug\b",
        r"\bantibiotic\b",
        r"\banalgesic\b",
        r"\bibuprofen\b",
        r"\bdiclofenac\b",
        r"\bcarbamazepine\b",
        r"\bacetaminophen\b",
        r"\bparacetamol\b",
        r"\bcaffeine\b",
        r"\btriclosan\b",
        r"\bsulfamethoxazole\b",
        r"\btetracycline\b",
        r"\bnaproxen\b",
    ),
    "phenols": (
        r"\bphenols?\b",
        r"\bphenolic\b",
        r"\bcresol\b",
        r"\bbisphenol\b",
        r"\bchlorophenol\b",
        r"\bnitrophenol\b",
        r"\bnonylphenol\b",
        r"\bresorcinol\b",
        r"\bhydroquinone\b",
    ),
    "chlorinated_organics": (
        r"\bchlorinated\s+organic",
        r"\bchlorinated\s+solvent",
        r"\borganochlorine\b",
        r"\bPCB(s)?\b",
        r"\bpolychlorinated\b",
        r"\bdioxin\b",
        r"\bfuran\b",
        r"\btrichloroethylene\b",
        r"\btetrachloroethylene\b",
        r"\bperchloroethylene\b",
        r"\bchloroform\b",
        r"\bcarbon\s+tetrachloride\b",
        r"\bdichloromethane\b",
        r"\bchlorobenzene\b",
        r"\bchloroalkane\b",
    ),
    "solvents_vocs": (
        r"\bsolvents?\b",
        r"\bVOC(s)?\b",
        r"\bvolatile\s+organic",
        r"\bBTEX\b",
        r"\bbenzene\b",
        r"\btoluene\b",
        r"\bethanol\b",
        r"\bmethanol\b",
        r"\bacetone\b",
        r"\bethy(?:l)?benzene\b",
        r"\bxylene\b",
        r"\bstyrene\b",
        r"\bhexane\b",
        r"\bethyl\s+acetate\b",
    ),
    "other_unknown": (
        r"\bother\b",
        r"\bunknown\b",
        r"\bunclassified\b",
        r"\bunspecified\b",
        r"\bnot\s+available\b",
    ),
}

_CATEGORY_ALIASES: dict[str, tuple[str, ...]] = {
    "chlorinated organics": ("chlorinated_organics",),
    "chlorinated-organics": ("chlorinated_organics",),
    "pharmaceuticals/ppcps": ("pharmaceuticals_personal_care_products",),
    "pharmaceuticals and personal care products": (
        "pharmaceuticals_personal_care_products",
    ),
    "pharmaceuticals_personal_care_products": (
        "pharmaceuticals_personal_care_products",
    ),
    "ppcp": ("pharmaceuticals_personal_care_products",),
    "ppcps": ("pharmaceuticals_personal_care_products",),
    "metals/metalloids": ("metals_metalloids",),
    "metals_metalloids": ("metals_metalloids",),
    "solvents/vocs": ("solvents_vocs",),
    "solvents_vocs": ("solvents_vocs",),
    "other/unknown": ("other_unknown",),
    "other_unknown": ("other_unknown",),
}

_COMPILED_PATTERNS = {
    category: tuple(re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns)
    for category, patterns in _CATEGORY_PATTERNS.items()
}

_METAL_SYMBOLS = {
    "As",
    "Cd",
    "Pb",
    "Hg",
    "Cr",
    "Cu",
    "Ni",
    "Zn",
    "Se",
    "Sb",
    "Ba",
    "Ag",
    "Al",
    "Co",
    "Mn",
    "Fe",
    "V",
    "Mo",
    "Tl",
    "U",
    "Sn",
    "Ti",
    "B",
}


def assign_chemical_categories(
    chemical_table: pd.DataFrame,
    config: Mapping[str, Any] | str | Path | None = None,
) -> pd.DataFrame:
    """Assign literature-relevant pollutant classes using traceable rules.

    The rule order favors specific chemical classes over broad use classes. If
    a reliable non-unknown category already exists in the input, that category
    is preserved and normalized to the canonical labels used by the evaluation
    protocol.
    """

    if not isinstance(chemical_table, pd.DataFrame):
        raise TypeError("chemical_table must be a pandas DataFrame.")

    cfg = _load_config(config)
    output = chemical_table.copy()
    records: list[dict[str, object]] = []
    for row_index, row in output.iterrows():
        assignment = _assign_row_category(row, cfg)
        records.append(
            {
                str(cfg["chemical_id_column"]): _chemical_id_from_row(row, cfg, row_index),
                str(cfg["output_category_column"]): assignment.category,
                "category_confidence": assignment.confidence,
                "category_evidence": assignment.evidence,
                "category_source": assignment.source,
            }
        )

    assignments = pd.DataFrame.from_records(records)
    for column in assignments.columns:
        output[column] = assignments[column].to_numpy()
    return output


class _CategoryAssignment:
    def __init__(
        self,
        category: str,
        confidence: float,
        evidence: str,
        source: str,
    ) -> None:
        self.category = category
        self.confidence = confidence
        self.evidence = evidence
        self.source = source


def _assign_row_category(row: pd.Series, cfg: Mapping[str, Any]) -> _CategoryAssignment:
    existing_text = _joined_column_text(row, cfg["existing_category_columns"])
    if existing_text:
        match = _match_text(existing_text, include_unknown=False)
        if match is not None:
            category, evidence = match
            return _CategoryAssignment(category, 1.0, f"existing_category:{evidence}", "existing")

    name_text = _joined_column_text(row, cfg["name_columns"])
    if name_text:
        match = _match_text(name_text, include_unknown=False)
        if match is not None:
            category, evidence = match
            return _CategoryAssignment(category, 0.85, f"name_rule:{evidence}", "name_rule")

    smiles = _clean_text(row.get(str(cfg["smiles_column"])))
    if smiles:
        match = _match_smiles(smiles)
        if match is not None:
            category, evidence = match
            return _CategoryAssignment(category, 0.70, f"smiles_rule:{evidence}", "smiles_rule")

    fallback_match = _match_text(f"{existing_text} {name_text}", include_unknown=True)
    if fallback_match is not None:
        category, evidence = fallback_match
        if category == str(cfg.get("unknown_category", "other_unknown")):
            return _CategoryAssignment(category, 0.20, f"unknown_rule:{evidence}", "unknown")

    return _CategoryAssignment(
        str(cfg.get("unknown_category", "other_unknown")),
        0.10,
        "no reliable category rule matched",
        "fallback",
    )


def _match_text(text: str, *, include_unknown: bool) -> tuple[str, str] | None:
    normalized = _normalize_category_label(text)
    if normalized is not None and (include_unknown or normalized != "other_unknown"):
        return normalized, normalized

    categories = CATEGORY_ORDER if include_unknown else CATEGORY_ORDER[:-1]
    for category in categories:
        for pattern in _COMPILED_PATTERNS[category]:
            match = pattern.search(text)
            if match is not None:
                return category, match.group(0)
    return None


def _match_smiles(smiles: str) -> tuple[str, str] | None:
    text = smiles.strip()
    compact = text.replace(" ", "")
    if _looks_like_metal_smiles(compact):
        return "metals_metalloids", "metal/metalloid symbol in SMILES"
    if compact.count("F") >= 4 and ("C(F)(F)" in compact or "C(F)F" in compact):
        return "PFAS", "highly fluorinated carbon pattern"
    if "P" in compact and re.search(r"P(?:\([=O]\)|=O|O|S)", compact):
        return "organophosphates", "phosphorus ester/phosphonate pattern"
    if "Cl" in compact or "Br" in compact:
        return "chlorinated_organics", "halogenated organic SMILES"
    if _looks_like_pah_smiles(compact):
        return "PAHs", "fused aromatic ring heuristic"
    if _looks_like_phenol_smiles(compact):
        return "phenols", "aromatic hydroxyl heuristic"
    return None


def _looks_like_metal_smiles(smiles: str) -> bool:
    bracket_tokens = re.findall(r"\[([A-Z][a-z]?)", smiles)
    if any(token in _METAL_SYMBOLS for token in bracket_tokens):
        return True
    return smiles in _METAL_SYMBOLS


def _looks_like_pah_smiles(smiles: str) -> bool:
    aromatic_carbons = smiles.count("c")
    ring_digits = set(re.findall(r"\d", smiles))
    return aromatic_carbons >= 10 and len(ring_digits) >= 2 and not any(
        element in smiles for element in ("N", "O", "S", "P")
    )


def _looks_like_phenol_smiles(smiles: str) -> bool:
    return bool(re.search(r"c1[a-zA-Z0-9@\[\]\(\)=#\\/]*O", smiles)) or bool(
        re.search(r"Oc1", smiles)
    )


def _normalize_category_label(text: object) -> str | None:
    cleaned = _clean_text(text)
    if not cleaned:
        return None
    key = cleaned.lower().strip()
    key = key.replace("&", " and ")
    key = re.sub(r"\s+", " ", key)
    compact = key.replace("-", "_").replace(" ", "_").replace("/", "_")

    for category in CATEGORY_ORDER:
        category_key = category.lower()
        if key == category_key or compact == category_key.lower():
            return category

    if key in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[key][0]
    if compact in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[compact][0]
    return None


def _joined_column_text(row: pd.Series, columns: Sequence[str]) -> str:
    values: list[str] = []
    for column in columns:
        if column not in row.index:
            continue
        text = _clean_text(row.get(column))
        if text:
            values.append(text)
    return " | ".join(values)


def _chemical_id_from_row(row: pd.Series, cfg: Mapping[str, Any], row_index: Any) -> str:
    for column in cfg.get("identifier_columns", []):
        if column not in row.index:
            continue
        text = _clean_text(row.get(column))
        if text:
            return text
    return f"row_{row_index}"


def _load_config(config: Mapping[str, Any] | str | Path | None) -> dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)
    if config is None:
        return cfg
    if isinstance(config, (str, Path)):
        loaded = _read_yaml(Path(config))
    elif isinstance(config, Mapping):
        loaded = dict(config)
    else:
        raise TypeError("config must be a mapping, path, string, or None.")

    if isinstance(loaded.get("evaluation"), Mapping):
        loaded = dict(loaded["evaluation"])
    if isinstance(loaded.get("category_assignment"), Mapping):
        category_assignment = dict(loaded.pop("category_assignment"))
        _deep_update(loaded, category_assignment)
    return _deep_update(cfg, loaded)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - depends on packaging.
        raise RuntimeError("PyYAML is required to load evaluation configs.") from exc

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Evaluation config must contain a mapping: {path}")
    return dict(data)


def _deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return ""
    return text

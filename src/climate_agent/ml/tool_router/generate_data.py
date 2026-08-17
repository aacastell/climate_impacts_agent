import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
TRAIN_FILE = DATA_DIR / "train.jsonl"
EVAL_FILE = DATA_DIR / "eval.jsonl"
EVAL_FRACTION = 0.15
SEED = 42

SYSTEM_PROMPT = (
    "You are a query router for a climate impacts assistant. Given a user question, extract: "
    "region (place name), sector (Agriculture or Biome), gwl_mode (gwl, heat_extreme, or "
    "precip_extreme), and gwl_value (a number). Respond with ONLY a JSON object with keys: "
    "region, sector, gwl_mode, gwl_value."
)

# Real place names — countries, states/provinces, and named sub-regions — spanning multiple
# continents, matching the "geocode must handle arbitrary regions, not a fixed list" requirement.
REGIONS = [
    "Tolima, Colombia", "Iowa, United States", "Punjab, India", "the Nile Delta, Egypt",
    "the Mekong Delta, Vietnam", "Bavaria, Germany", "Queensland, Australia",
    "São Paulo, Brazil", "Sichuan, China", "Kenya", "Ethiopia", "California, United States",
    "Ukraine", "Argentina", "Nigeria", "Indonesia", "Thailand", "France", "Spain",
    "Kansas, United States", "the Punjab region of Pakistan", "Bangladesh",
    "the Prairie provinces, Canada", "Andalusia, Spain", "Gujarat, India",
]

SECTORS = ["Agriculture", "Biome"]

# (mode, value_range, unit, templates) — templates use {value}, {sector}, {region}
MODE_SPECS = [
    (
        "gwl",
        (1.0, 4.0),
        "°C",
        [
            "How would {value}°C of warming affect {sector} in {region}?",
            "What happens to {sector} in {region} at {value} degrees of warming?",
            "Impact of {value}C warming on {sector} in {region}",
            "At {value}°C above pre-industrial levels, how is {sector} in {region} affected?",
            "{region} {sector} outlook under {value} degrees of global warming",
        ],
    ),
    (
        "heat_extreme",
        (5.0, 30.0),
        " more heat-extreme days/year",
        [
            "How does {sector} in {region} change with {value} more heat-extreme days a year?",
            "If {region} sees {value} additional days above 35°C, how is {sector} affected?",
            "{sector} impact in {region} from {value} extra extreme-heat days annually",
            "What if heat waves added {value} more scorching days a year in {region} — effect on {sector}?",
        ],
    ),
    (
        "precip_extreme",
        (5.0, 25.0),
        "% more extreme rainfall",
        [
            "How would {sector} in {region} be affected if extreme rainfall increased by {value}%?",
            "{region} {sector} impact from a {value}% increase in precipitation extremes",
            "What happens to {sector} in {region} if heavy-rain events rise {value} percent?",
            "Effect on {sector} in {region} from {value}% more intense storm rainfall",
        ],
    ),
]


def _generate_example(region: str, sector: str, mode: str, value_range: tuple, templates: list[str]) -> dict:
    """One (query, target) training pair for a given region/sector/gwl_mode.

    Args: region, sector — real values to fill into the template. mode — gwl_mode key.
    value_range — (low, high) for the sampled numeric value. templates — phrasing options.
    Returns: {"query": ..., "target": {"region", "sector", "gwl_mode", "gwl_value"}}.
    """
    value = round(random.uniform(*value_range), 1)
    template = random.choice(templates)
    query = template.format(value=value, sector=sector.lower(), region=region)
    return {
        "query": query,
        "target": {"region": region, "sector": sector, "gwl_mode": mode, "gwl_value": value},
    }


def generate_dataset(n_per_combination: int = 2) -> list[dict]:
    """All (region, sector, gwl_mode) combinations, each sampled n_per_combination times.

    Args: n_per_combination — number of examples generated per (region, sector, mode) combo,
    each with a re-sampled value and re-chosen template for variety.
    Returns: list of {query, target} dicts.
    """
    examples = []
    for region in REGIONS:
        for sector in SECTORS:
            for mode, value_range, _unit, templates in MODE_SPECS:
                for _ in range(n_per_combination):
                    examples.append(_generate_example(region, sector, mode, value_range, templates))
    random.shuffle(examples)
    return examples


def save_split(examples: list[dict]) -> None:
    """Split into train/eval and write both as JSONL to DATA_DIR."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    n_eval = int(len(examples) * EVAL_FRACTION)
    eval_examples, train_examples = examples[:n_eval], examples[n_eval:]

    TRAIN_FILE.write_text("\n".join(json.dumps(e) for e in train_examples))
    EVAL_FILE.write_text("\n".join(json.dumps(e) for e in eval_examples))
    print(f"Wrote {len(train_examples)} train / {len(eval_examples)} eval examples to {DATA_DIR}")


def main() -> None:
    random.seed(SEED)
    examples = generate_dataset()
    save_split(examples)


if __name__ == "__main__":
    main()

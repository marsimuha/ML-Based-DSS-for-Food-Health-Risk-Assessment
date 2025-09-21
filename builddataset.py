# Build dataset
import re
import pandas as pd
from pathlib import Path

INPUT_XLSX = Path(r"C:\Users\Lenovo\Desktop\Final Thesis Script\Final Database.xlsx")
SHEET_NAME = "Sheet1"
OUTPUT_CSV = Path(r"C:\Users\Lenovo\Desktop\Final Thesis Script\Final Dataset.csv")

ING_COL   = "Ingredients list"
CODE_COL  = "Product Code"
TARGET_COL= "Nova Score"

# Groups
COLORANTS = {
    "e100","e101","e102","e104","e110","e120","e122","e123","e124","e127","e129",
    "e131","e132","e133","e140","e141","e142","e150a","e150b","e150c","e150d",
    "e151","e153","e155","e160a","e160b","e160c","e160d","e160e","e161b","e161g",
    "e162","e163","e170","e171","e172","e173","e174","e175","e180"
}

SWEETENERS = {
    "e420","e421","e950","e951","e952","e953","e954","e955","e957","e959",
    "e960a","e960b","e960c","e961","e962","e964","e965","e966","e967","e968","e969"
}
EMULSIFIERS = {
    "e322","e400","e401","e402","e403","e404","e405","e406","e407","e407a","e410","e412",
    "e413","e414","e415","e416","e417","e418","e425","e426","e427","e432","e433","e434",
    "e435","e436","e440","e442","e444","e445","e460","e461","e462","e463","e464","e465",
    "e466","e468","e469","e470a","e470b","e471","e472a","e472b","e472c","e472d","e472e",
    "e472f","e473","e474","e475","e476","e477","e479b","e481","e482","e483","e491","e492",
    "e493","e494","e495","e1103"
}

# E-number handling 
E_REGEX_FULL = re.compile(
    r"\b[eE]\s*[-]?\s*(\d{3,4})\s*([a-zA-Z])?\s*(?:\(\s*[ivxlcdmIVXLCDM]+\s*\))?",
    flags=re.UNICODE
)

def extract_enumbers_set(text: str) -> set:
    if not isinstance(text, str):
        return set()
    return {
        f"e{m.group(1).lower()}{(m.group(2) or '').lower()}"
        for m in E_REGEX_FULL.finditer(text)
    }

# Count ANY E-number occurrence (each appearance), including letter suffix
E_ADD_COUNT_REGEX = re.compile(
    r"\b[eE]\s*[-]?\s*\d{3,4}[a-zA-Z]?\b(?:\(\s*[ivxlcdmIVXLCDM]+\s*\))?"
)
def count_additives(text: str) -> int:
    return len(E_ADD_COUNT_REGEX.findall(str(text)))

def count_ingredients(text: str) -> int:
    # Ingredients are comma-separated
    return sum(1 for p in str(text).split(",") if p.strip())

# Helpers =
def any_in_set(target: set, present: set) -> int:
    return int(bool(target.intersection(present)))

def minmax_norm(series: pd.Series) -> pd.Series:
    s_min, s_max = series.min(), series.max()
    if pd.isna(s_min) or pd.isna(s_max) or s_max == s_min:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - s_min) / (s_max - s_min)

def any_kw(text: str, words_or_patterns) -> int:
    t = text if isinstance(text, str) else ""
    for w in words_or_patterns:
        if re.search(w, t, flags=re.IGNORECASE):
            return 1
    return 0

def any_word(text: str, words) -> int:
    t = text if isinstance(text, str) else ""
    return int(any(w in t for w in words))

# Added sugar / oil-and-fat 
ADDED_SUGAR_PATTERNS = [r"\badded[-\s]?sugars?\b"]
OIL_AND_FAT_PATTERNS = [r"\boil[-\s]?and[-\s]?fat(s)?\b"]

# Carcinogens, Allergens, Cardio-risk
CARCINOGEN_ECODES = {
    "e950","e962","e969","e951","e320","e321","e407","e310","e250","e251","e955","e319"
}
CARCINOGEN_KEYWORDS = [
    r"\bsaccharin\b", r"\bsuccharin\b",
    r"\bpotassium\s+bromate\b",
    r"\bpotassium\s+iodate\b",
    r"\bcyclamate\b",
    r"\bazodicarbonamide\b",
    r"\bartificial\s+flavou?r(?:ing)?s?\b",
    r"\bnatural\s+flavou?r(?:ing)?s?\b",
    r"\baloe\s+vera\b"
]
ALLERGEN_KEYWORDS = [
    r"\bannatto\b|annatto",  # robust for typos + correct spelling
    r"\bautoly(z|s)ed\s+yeast\s+extract\b",
    r"\bbrazzein\b",
    r"\b(carmine|cochineal|carminic\s+acid)\b",
    r"\bcasein\b|\bsodium\s+caseinate\b|\bcaseinate\b",
    r"\blactose\b",
    r"\b(msg|monosodium\s+glutamate)\b",
    r"\bmycoprotein\b|\bquorn\b",
    r"\bpropylene\s+glycol\b",
    r"\bquinine\b",
    r"\bsodium\s+benzoate\b|\bbenzoic\s+acid\b",
    r"\bsulfites?\b|\bsulphites?\b|\bsulfur\s+dioxide\b|\bsodium\s+bisulfite\b|\bsodium\s+bisulphite\b",
    r"\bgum\s+arabic\b|\barabic\s+gum\b|\bfurcelleran\b|\bgellan\b|\bghatti\b|\bguar\b|\bkaraya\b|\blocust\s+bean\s+gum\b|\btragacanth\b|\bxanthan\b"
]
CARDIO_RISK_KEYWORDS = [
    r"\bfructose\b",
    r"\bhigh[-\s]?fructose\s+corn\s+syrup\b|\bhfcs\b",
    r"\bphosphoric\s+acid\b|\bphosphates?\b",
    r"\bsalt\b|\bsea\s+salt\b",
    r"\bsugar\b|\bsucrose\b",
    r"\btrans\s+fat(s)?\b|\bpartially\s+hydrogenated\s+vegetable\s+oil\b|\bpartially\s+hydrogenated\b|\bhydrogenated\b"
]

# NOVA-3 targeting feature vocabularies 
CHEESE_MARKERS  = ["cheese","yogurt","cream cheese","curd","feta","mozzarella","cheddar"]
SUGAR_TERMS = ["sugar","glucose","fructose","sucrose","dextrose","maltose","lactose","syrup","molasses"]
SALT_TERMS  = ["salt","sea salt"]
NOVA1_FOODS = ["wheat","flour","milk","potato","rice","bean","oat","corn","maize"]

# BUILD 
def main():
    if not INPUT_XLSX.exists():
        raise FileNotFoundError(f"Cannot find {INPUT_XLSX.resolve()}. Update INPUT_XLSX.")
    df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME)

    if ING_COL not in df.columns:
        raise ValueError(f"Column '{ING_COL}' not found in sheet '{SHEET_NAME}'.")

    # Normalize text
    df[ING_COL] = df[ING_COL].astype(str).str.lower()

    # E-numbers present per row (set) + counts
    ecode_sets = df[ING_COL].apply(extract_enumbers_set)
    num_ingredients = df[ING_COL].apply(count_ingredients).astype(float)
    num_additives   = df[ING_COL].apply(count_additives).astype(float)
    additives_per_ingredient = (num_additives / num_ingredients.replace(0, pd.NA)).fillna(0.0).astype(float)

    # Base output frame
    out = pd.DataFrame({
        CODE_COL:   df.get(CODE_COL, pd.Series(range(len(df)))),
        TARGET_COL: df.get(TARGET_COL, pd.Series([None]*len(df))),
    })

    # Existing binary flags (specific E + groups + text flags) 
    out["has_e330"] = ecode_sets.apply(lambda s: int("e330" in s))
    out["has_e500"] = ecode_sets.apply(lambda s: int("e500" in s))
    out["has_e202"] = ecode_sets.apply(lambda s: int("e202" in s))
    out["has_e322"] = ecode_sets.apply(lambda s: int("e322" in s))

    out["has_colorant"]    = ecode_sets.apply(lambda s: any_in_set(COLORANTS, s))
    out["has_sweetener"]   = ecode_sets.apply(lambda s: any_in_set(SWEETENERS, s))
    out["has_emulsifier"]  = ecode_sets.apply(lambda s: any_in_set(EMULSIFIERS, s))

    out["has_added_sugar"] = df[ING_COL].apply(lambda t: any_kw(t, ADDED_SUGAR_PATTERNS))
    out["has_oil_and_fat"] = df[ING_COL].apply(lambda t: any_kw(t, OIL_AND_FAT_PATTERNS))

    carcinogen_from_ecodes = ecode_sets.apply(lambda s: any_in_set(CARCINOGEN_ECODES, s))
    carcinogen_from_words  = df[ING_COL].apply(lambda t: any_kw(t, CARCINOGEN_KEYWORDS))
    out["has_carcinogen"] = ((carcinogen_from_ecodes == 1) | (carcinogen_from_words == 1)).astype(int)

    out["has_allergen"] = df[ING_COL].apply(lambda t: any_kw(t, ALLERGEN_KEYWORDS))
    out["has_cardiovascular_risk_ingredient"] = df[ING_COL].apply(lambda t: any_kw(t, CARDIO_RISK_KEYWORDS))

    # Numeric features (normalized)
    out["number_of_ingredients_norm"] = minmax_norm(num_ingredients)
    out["number_of_additives_norm"]   = minmax_norm(num_additives)
    out["additives_per_ingredient_norm"] = minmax_norm(additives_per_ingredient)

    # NOVA-3 targeted features
    out["has_cheese_marker"]                = df[ING_COL].apply(lambda t: any_word(t, CHEESE_MARKERS))

    # Complexity but not additives: helps distinguish NOVA 3 from 2 & 4
    complex_score = (num_ingredients - num_additives).astype(float)
    out["complex_non_additive_score_norm"] = minmax_norm(complex_score)

    # Salt + staple food combo (e.g., "salt" with wheat/milk/potato…)
    out["salt_and_food_combo"] = df[ING_COL].apply(
        lambda t: int(("salt" in t) and any(w in t for w in NOVA1_FOODS))
    )

    # Save
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV.resolve()}  (rows: {len(out)}, cols: {out.shape[1]})")

if __name__ == "__main__":
    main()


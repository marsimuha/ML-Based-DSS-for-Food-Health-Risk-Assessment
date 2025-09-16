import streamlit as st
import joblib
import re
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# Load the trained model
model = joblib.load("final_logreg_model.joblib")

# Risk patterns
CARCINOGEN_ECODES = {
    "e950", "e962", "e969", "e951", "e320", "e321", "e407", "e310",
    "e250", "e251", "e955", "e319"
}
CARCINOGEN_KEYWORDS = [
    r"acesulfame\s+potassium", r"aloe\s+vera", r"artificial\s+(and\s+)?natural\s+flavou?r(?:ing)?s?",
    r"aspartame", r"equal", r"aminosweet", r"azodicarbonamide", r"bha",
    r"butylated\s+hydroxyanisole", r"butylated\s+hydroxytoluene", r"bht",
    r"caramel\s+color(?:ing)?", r"carrageenan", r"cyclamate", r"ginkgo\s+biloba",
    r"potassium\s+bromate", r"potassium\s+iodate", r"propyl\s+gallate", r"saccharin",
    r"succharin", r"sodium\s+nitrite", r"nitrate", r"nitrite", r"sucralose", r"splenda",
    r"tbhq", r"tert[-\s]?butylhydroquinone"
]
ALLERGEN_KEYWORDS = [
    r"annatto", r"autoly(z|s)ed\s+yeast\s+extract", r"brazzein",
    r"(carmine|cochineal|carminic\s+acid)", r"casein", r"sodium\s+caseinate", r"lactose",
    r"(msg|monosodium\s+glutamate)", r"mycoprotein", r"quorn", r"propylene\s+glycol",
    r"quinine", r"sodium\s+benzoate", r"benzoic\s+acid", r"sulfites?", r"sulphites?",
    r"sulfur\s+dioxide", r"sodium\s+bisulfite", r"sodium\s+bisulphite",
    r"gum\s+arabic", r"arabic\s+gum", r"furcelleran", r"gellan", r"ghatti", r"guar",
    r"karaya", r"locust\s+bean\s+gum", r"tragacanth", r"xanthan"
]
CARDIO_RISK_KEYWORDS = [
    r"fructose", r"high[-\s]?fructose\s+corn\s+syrup", r"hfcs",
    r"phosphoric\s+acid", r"phosphates?", r"salt", r"sea\s+salt",
    r"sugar", r"sucrose", r"trans\s+fat(s)?", r"partially\s+hydrogenated", r"hydrogenated"
]

# Common ingredient words for typo correction
COMMON_INGREDIENTS = [
    "sugar", "salt", "fructose", "lecithin", "emulsifier", "citrate",
    "flour", "water", "oil", "syrup", "flavoring", "sweetener",
    "preservative", "additive", "colorant", "gluten", "fat", "milk",
    "lactose", "phosphate", "casein", "carmine", "msg", "benzoate"
]

def correct_typos(ingredient_str, threshold=90):
    words = re.split(r'[,\n]', ingredient_str.lower())
    cleaned_words = []
    for word in words:
        word = word.strip()
        if not word:
            continue
        match, score = process.extractOne(word, COMMON_INGREDIENTS)
        if score >= threshold:
            cleaned_words.append(match)
        else:
            cleaned_words.append(word)
    return ', '.join(cleaned_words)


def find_matches(text, patterns, return_matches=False):
    matches = set()
    for pat in patterns:
        found = re.findall(pat, text)
        matches.update([m.strip() for m in found if m])
    if return_matches:
        return (len(matches) > 0), sorted(matches)
    return int(bool(matches))

def has_e_code(text, code):
    return int(code.lower() in text.lower())

def extract_features(ingredient_text):
    ing = correct_typos(ingredient_text.lower())
    ingredients = [i.strip() for i in re.split(r'[,\n;]', ing) if i.strip()]
    total_ingredients = len(ingredients)

    additive_matches = re.findall(r"\be\d{3,4}\b", ing)
    additive_set = set(additive_matches)
    total_additives = len(additive_set)

    additives_per_ingredient = total_additives / total_ingredients if total_ingredients > 0 else 0
    complex_score = (total_ingredients - total_additives) / total_ingredients if total_ingredients > 0 else 0

    has_carc, matched_carc = find_matches(ing, CARCINOGEN_KEYWORDS + list(CARCINOGEN_ECODES), return_matches=True)
    has_aller, matched_aller = find_matches(ing, ALLERGEN_KEYWORDS, return_matches=True)
    has_cardio, matched_cardio = find_matches(ing, CARDIO_RISK_KEYWORDS, return_matches=True)

    feats = {
        "has_e330": has_e_code(ing, "e330"),
        "has_e500": has_e_code(ing, "e500"),
        "has_e202": has_e_code(ing, "e202"),
        "has_e322": has_e_code(ing, "e322"),
        "has_colorant": int("color" in ing or "colour" in ing),
        "has_sweetener": int("sweetener" in ing),
        "has_emulsifier": int("emulsifier" in ing),
        "has_added_sugar": int("sugar" in ing),
        "has_oil_and_fat": int("oil" in ing or "fat" in ing),
        "has_carcinogen": int(has_carc),
        "has_allergen": int(has_aller),
        "has_cardiovascular_risk_ingredient": int(has_cardio),
        "number_of_ingredients_norm": total_ingredients / 50,
        "number_of_additives_norm": total_additives / 25,
        "additives_per_ingredient_norm": additives_per_ingredient,
        "has_cheese_marker": int("cheese" in ing),
        "complex_non_additive_score_norm": complex_score,
        "salt_and_food_combo": int("salt" in ing and total_ingredients > 2)
    }

    return feats, matched_carc, matched_aller, matched_cardio

def risk_level(nova):
    return "Low" if nova in [1, 2] else "Moderate" if nova == 3 else "High"

# Streamlit App
st.set_page_config(page_title="NOVA DSS", layout="centered")
st.title("Food Safety Control: NOVA Score & Health Risk Assessment 🧠")
st.subheader("A Machine Learning-Based Decision Support System")
st.markdown("Enter a food product's **ingredient list** to assess its **NOVA score** and health risk profile.")

input_text = st.text_area("Ingredient List", height=150, placeholder="e.g., water, sugar, E322, salt, artificial flavoring")
def is_valid_ingredient_input(text):
    if len(text.strip()) < 10:
        return False, "Input is too short to be a valid ingredient list."
    if not re.search(r'[,\n]', text):  # Ingredient lists usually contain commas or new lines
        return False, "Input does not appear to be a valid ingredient list. Please separate ingredients with commas."
    if re.search(r"\bi\s+am\s+(stupid|dumb|idiot|cool|a\s+banana)\b", text.lower()):
        return False, "Please enter a valid ingredient list, not unrelated text."
    return True, ""

# Perform validation
is_valid, error_msg = is_valid_ingredient_input(input_text)

if not is_valid and input_text:
    st.error(error_msg)

if st.button("Predict NOVA Score") and input_text and is_valid:
    feats, matched_carc, matched_aller, matched_cardio = extract_features(input_text)
    X_input = pd.DataFrame([feats], dtype=float)

    # Ensure all columns present
    for col in model.feature_names_in_:
        if col not in X_input.columns:
            X_input[col] = 0
    X_input = X_input[model.feature_names_in_]

    pred = int(model.predict(X_input)[0])
    risk = risk_level(pred)

    st.subheader("🧪 Prediction Results")
    st.write(f"**Predicted NOVA Score:** {pred}")
    if pred == 1:
        st.write("This food is unprocessed or minimally processed.")
    elif pred == 2:
        st.write("This food is processed culinary ingredient.")
    elif pred == 3:
        st.write("This food is processed.")
    else:
        st.write("This food is ultra-processed.")
    st.write(f"**Health Risk Level:** {risk}")

    st.subheader("⚠️ Health Risk Factor Detection")

    if matched_carc:
        st.write("❌ **Carcinogens detected**:", ", ".join(matched_carc))
    else:
        st.write("✅ No carcinogens detected.")

    if matched_aller:
        st.write("❌ **Allergens detected**:", ", ".join(matched_aller))
    else:
        st.write("✅ No allergens detected.")

    if matched_cardio:
        st.write("❌ **Cardiovascular risk ingredients detected**:", ", ".join(matched_cardio))
    else:
        st.write("✅ No cardiovascular risk ingredients detected.")

elif input_text:
    st.info("Click the **Predict NOVA Score** button to get the results.")

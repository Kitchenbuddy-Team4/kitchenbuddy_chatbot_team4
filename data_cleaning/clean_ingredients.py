import json
import re

# === Step 1: Load ingredients from JSON ===
with open("../resources/ingredients/unique_ingredients.json", "r") as f:
    ingredients = json.load(f)

# === Step 2: Define the cleaning function ===
def clean_ingredient(ingredient):
    ingredient = ingredient.replace('\n', ' ').replace('\t', ' ')
    ingredient = re.sub(r'^[\s\-–—•\d\/\.]+', '', ingredient)   # strip leading formatting
    ingredient = re.sub(r'\b\d+[\w\/.]*\b', '', ingredient)     # remove embedded numbers
    ingredient = re.sub(r'\s+', ' ', ingredient)                # collapse whitespace
    return ingredient.lower().strip()

# === Step 3: Clean all ingredients and remove duplicates ===
cleaned_ingredients = set(clean_ingredient(i) for i in ingredients if i.strip())
cleaned_ingredients = sorted(cleaned_ingredients)  # optional: sort alphabetically

# === Step 4: Write cleaned list back to JSON ===
with open("../resources/ingredients/unique_ingredients.json", "w") as f:
    json.dump(cleaned_ingredients, f, indent=4)

print("✅ Cleaned ingredients saved to unique_ingredients.json")

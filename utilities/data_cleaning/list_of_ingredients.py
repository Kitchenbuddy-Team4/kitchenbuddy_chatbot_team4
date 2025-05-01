import pandas as pd
import ast
import json

# Load dataset
recipes_df = pd.read_csv("../../resources/dataset/RecipeNLG_dataset.csv")

# Convert ner strings to Python lists
recipes_df['NER'] = recipes_df['NER'].apply(ast.literal_eval)

# Flatten and deduplicate all ner ingredients
all_ingredients = set(
    ingredient.lower().strip()
    for row in recipes_df['NER']
    for ingredient in row
)

# Sort and write to JSON
unique_ingredients = sorted(all_ingredients)

with open("../../resources/ingredients/unique_ingredients.json", "w") as f:
    json.dump(unique_ingredients, f, indent=4)

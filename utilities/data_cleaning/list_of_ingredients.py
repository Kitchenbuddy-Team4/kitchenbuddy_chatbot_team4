import pandas as pd
import ast
import json

# import the data
recipes_df = pd.read_csv("../../resources/dataset/RecipeNLG_dataset.csv")

recipes_df['ingredients'] = recipes_df['ingredients'].apply(ast.literal_eval)

all_ingredients = set(ingredient for row in recipes_df['ingredients'] for ingredient in row)
unique_ingredients = sorted(all_ingredients)

# Write to a JSON file
with open("../../resources/ingredients/unique_ingredients.json", "w") as f:
    json.dump(unique_ingredients, f, indent=4)

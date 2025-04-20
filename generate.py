import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 1000

# Features
temperature = np.random.uniform(30, 60, n_samples)  # °C
pH = np.random.normal(7.5, 0.5, n_samples).clip(6.5, 8.5)
moisture_content = np.random.uniform(80, 95, n_samples)  # %

# Food waste composition ratios (3 categories)
food_waste_ratios = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2])

# Define ratio_map HERE (before using it)
ratio_map = {
    0: "70% fruits/vegetables, 20% grains, 10% dairy",
    1: "50% fruits, 30% vegetables, 20% bakery waste",
    2: "80% pre-consumer waste, 15% post-consumer waste, 5% oils"
}

# Methane yield calculation (enhanced formula)
base_yield = np.where(
    food_waste_ratios == 0, 0.85,
    np.where(food_waste_ratios == 1, 0.92, 1.0)
)

methane_yield = (
    base_yield +
    0.08 * (temperature - 35) +    # Temperature sensitivity
    1.2 * (pH - 7.0) +             # pH sensitivity
    0.03 * (moisture_content - 85) +  # Moisture sensitivity
    np.random.normal(0, 0.08, n_samples)  # Process noise
)
methane_yield = np.clip(methane_yield, 0.6, 1.3)  # m³ CH4/kg VS

# Create DataFrame
df = pd.DataFrame({
    "temperature": temperature,
    "pH": pH,
    "moisture_content": moisture_content,
    "food_waste_ratio": food_waste_ratios,
    "methane_yield": methane_yield
})

# Add human-readable ratio descriptions (ratio_map is now defined)
df["feedstock_composition"] = df["food_waste_ratio"].map(ratio_map)

# Save to CSV
df.to_csv("data.csv", index=False)

# Show sample data
print(df.head())
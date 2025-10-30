import requests
import json

# NUSMods API URL
url = "https://api.nusmods.com/v2/2025-2026/moduleInfo.json"

# Send GET request
response = requests.get(url)

# Ensure success
response.raise_for_status()

# Parse JSON
modules = response.json()

# Print how many modules and one sample
print(f"Total modules: {len(modules)}")
print(json.dumps(modules[0], indent=2))

# Optionally save to a local file
with open("../data/nusmods_2025_2026_moduleInfo.json", "w", encoding="utf-8") as f:
    json.dump(modules, f, indent=2, ensure_ascii=False)
import yaml
import json
import requests
import base64
import re
import math

# --- Load YAML ---
with open("test.yaml", "r") as f:
    config = yaml.safe_load(f)

api_config = config["providers"][0]["config"]
url = api_config["url"]
method = api_config["method"].upper()
body_file = api_config["body"].replace("file://", "")

# --- Load body content (question.txt) ---
with open(body_file, "r") as f:
    body_data = f.read()

# --- Send Request ---
response = requests.request(method, url, data=body_data)
output = response.text.strip()

print("\n🧪 Raw API Response:")
print(output)

# --- Parse response ---
try:
    parsed = json.loads(output)
    assert isinstance(parsed, list), "Output is not a list"
    assert len(parsed) == 4, "Output does not have 4 items"
except Exception as e:
    print(f"❌ Structural check failed: {e}")
    exit(1)

score = 0

# --- Check #1: First item must be 1 ---
if parsed[0] == 1:
    score += 4
    print("✅ Check 1 passed")
else:
    print("❌ Check 1 failed")

# --- Check #2: Second item must contain 'Titanic' (case-insensitive) ---
if re.search(r'titanic', str(parsed[1]), re.I):
    score += 4
    print("✅ Check 2 passed")
else:
    print("❌ Check 2 failed")

# --- Check #3: Third item must be ~0.485782 (±0.001) ---
try:
    if abs(float(parsed[2]) - 0.485782) <= 0.001:
        score += 4
        print("✅ Check 3 passed")
    else:
        print("❌ Check 3 failed")
except Exception:
    print("❌ Check 3 failed — not a number")

# --- Check #4: Fourth item must be base64 image ---
try:
    if isinstance(parsed[3], str) and parsed[3].startswith("data:image/png;base64,"):
        img_data = parsed[3].split(",")[1]
        size_kb = len(base64.b64decode(img_data)) / 1024
        if size_kb < 100:
            score += 8
            print("✅ Check 4 passed (image size < 100 KB)")
        else:
            print("❌ Check 4 failed (image too big)")
    else:
        print("❌ Check 4 failed (not a base64 image string)")
except Exception as e:
    print(f"❌ Check 4 error: {e}")

# --- Final Score ---
print(f"\n🎯 Final Score: {score}/20")

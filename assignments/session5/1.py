from pathlib import Path
import pandas as pd
import requests


p = Path(__file__).parent.parent.parent / "data" / "penguins_1.csv"
df = pd.read_csv(p).dropna().sample(100, random_state=42)
inputs = df.to_dict(orient="records")

response = requests.post("http://0.0.0.0:8080/invocations", json={"inputs": inputs})
print(response.text)
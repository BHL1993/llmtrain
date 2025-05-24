# single table
from io import StringIO

import pandas as pd

EXAMPLE_CSV_CONTENT = """
"Loss","Date","Score","Opponent","Record","Attendance"
"Hampton (14–12)","September 25","8–7","Padres","67–84","31,193"
"Speier (5–3)","September 26","3–1","Padres","67–85","30,711"
"Elarton (4–9)","September 22","3–1","@ Expos","65–83","9,707"
"Lundquist (0–1)","September 24","15–11","Padres","67–83","30,774"
"Hampton (13–11)","September 6","9–5","Dodgers","61–78","31,407"
"""

csv_file = StringIO(EXAMPLE_CSV_CONTENT)
df = pd.read_csv(csv_file)

str = df.head(5).to_string(index=False)

print(df.head(5).to_string(index=False))




if __name__ == "__main__":
    for t in reversed(range(8)):
        print(t)
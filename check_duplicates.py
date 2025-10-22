import pandas as pd

# wczytaj swój plik wynikowy
df = pd.read_csv("C:/Users/Bartek/Downloads/MOT16_results/MOT16-02/det.txt", header=None)

# znajdź wszystkie duplikaty (frame, track_id)
dupes = df[df.duplicated(subset=[0,1], keep=False)]

if dupes.empty:
    print("✅ Brak duplikatów ID w tym pliku.")
else:
    print("⚠️ Znaleziono duplikaty:")
    print(dupes)

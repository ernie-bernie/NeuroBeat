import pandas as pd

path = r"C:\Users\evyne\Documents\NeuroBeat\NeuroBeat\code\phase2\data\participant_ratings.xls"
df = pd.read_excel(path)

print(df.head())

filtered = df[df['Participant_id'] == 1]
print(filtered)


print(f"\nParticipant 1 — {len(filtered)} trials")
print(f"Valence range: {filtered['Valence'].min():.1f} to {filtered['Valence'].max():.1f}")
print(f"Arousal range: {filtered['Arousal'].min():.1f} to {filtered['Arousal'].max():.1f}")
print(f"Average valence: {filtered['Valence'].mean():.2f}")
print(f"Average arousal: {filtered['Arousal'].mean():.2f}")


#------------------------
# Run using python code/phase2/load_labels.py
#------------------------
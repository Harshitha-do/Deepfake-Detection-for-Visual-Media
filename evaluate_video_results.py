import pandas as pd
import os

df = pd.read_csv('video_predictions.csv')
# Normalize path separators to forward slash
df['video'] = df['video'].apply(lambda x: x.replace('\\', '/'))
df['folder'] = df['video'].apply(lambda x: x.split('/')[0])

ground_truth = {
    'Celeb-synthesis': 'FAKE',
    'Celeb-real': 'REAL',
    'YouTube-real': 'REAL'
}

df['true_label'] = df['folder'].map(ground_truth)
# Drop rows where folder not in ground_truth (e.g., test_videos)
df = df.dropna(subset=['true_label'])
if len(df) == 0:
    print("No matching folders found. Available folders:", df['folder'].unique())
else:
    df['correct'] = df['prediction'] == df['true_label']
    accuracy = df['correct'].mean()
    print(f"Overall accuracy: {accuracy:.2%}")
    for folder in ground_truth:
        sub = df[df['folder'] == folder]
        if len(sub) > 0:
            acc = (sub['prediction'] == sub['true_label']).mean()
            print(f"{folder:20} {acc:.2%} ({len(sub)} videos)")
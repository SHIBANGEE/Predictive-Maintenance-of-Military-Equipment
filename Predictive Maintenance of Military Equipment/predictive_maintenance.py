import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg

# Step 1: Load data
df = pd.read_csv("equipment_data.csv")
print("✅ Data Sample:\n", df.head())

# Step 2: Prepare features and label
X = df[['temperature', 'vibration', 'pressure', 'usage_hours']]
y = df['failure']

# Step 3: Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Predict and evaluate
y_pred = model.predict(X_test)
print("\n🔍 Classification Report:\n", classification_report(y_test, y_pred))

# Step 6: Feature Importance Bar Plot (Animated + Colorful)
importances = model.feature_importances_
features = X.columns
sorted_indices = importances.argsort()

# Bright colors for each bar
bright_colors = ['#FF6F61', '#6B5B95', '#88B04B', '#F7CAC9']

fig, ax = plt.subplots(figsize=(8, 6))

# Optional: set background color
fig.patch.set_facecolor('#f0f8ff')  # soft blue background
ax.set_facecolor('#fef9f4')  # creamy background for axes

# Optional: Add background image (uncomment if you have one)
# img = mpimg.imread("background.jpg")
# ax.imshow(img, extent=[0, 1, 0, 1], aspect='auto', zorder=-1, alpha=0.2)

bars = ax.barh(range(len(features)), [0]*len(features), 
               tick_label=features[sorted_indices], 
               color=[bright_colors[i] for i in sorted_indices])

ax.set_xlim(0, max(importances) + 0.05)
ax.set_title("Feature Importance", fontsize=14, fontweight='bold', color='#333')
ax.set_xlabel("Importance", fontsize=12, color='#444')
ax.tick_params(colors='#555')

def update(frame):
    for i, b in enumerate(bars):
        b.set_width(importances[sorted_indices][i] * (frame / 10))

ani = FuncAnimation(fig, update, frames=10, interval=200, repeat=False)
plt.tight_layout()
plt.show()

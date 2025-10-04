import matplotlib.pyplot as plt
import pandas as pd

# Define tasks and timeline (start and duration in weeks)
tasks = [
    ("Literature Review & Research Design", 1, 3),
    ("Data Acquisition & Preprocessing", 2, 4),
    ("Model Design & Implementation", 4, 5),
    ("Preliminary Experiments & Hyperparameter Tuning", 7, 4),
    ("Comprehensive Experiments & Model Evaluation", 11, 6),
    ("Explainability Analysis (Counterfactuals, XAI)", 14, 4),
    ("Results Interpretation & Comparative Study", 16, 4),
    ("Paper Writing & Refinement", 18, 4),
    ("Manuscript Finalization & Submission", 20, 2)
]

# Convert to DataFrame
df = pd.DataFrame(tasks, columns=["Task", "Start Week", "Duration"])
df["End Week"] = df["Start Week"] + df["Duration"]

# Assign colors for better visualization
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]


# Improve figure aesthetics
fig, ax = plt.subplots(figsize=(12, 6))

# Plot each task as a horizontal bar with improved style
for i, (task, start, duration) in enumerate(tasks):
    ax.barh(task, duration, left=start, color=colors[i], edgecolor='black', alpha=0.85, height=0.5)

# Format plot aesthetics
ax.set_xlabel("Project Weeks", fontsize=12, fontweight='bold')
ax.set_ylabel("Thesis Tasks", fontsize=12, fontweight='bold')
ax.set_title("Thesis Project Gantt Chart", fontsize=14, fontweight='bold')

# Improve grid visibility
ax.set_xticks(range(1, 26))  # Weeks from 1 to 25
ax.xaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)  # Ensure grid lines stay behind bars
ax.invert_yaxis()  # Flip y-axis to have first task on top

# Adjust tick labels
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)

# Add subtle background color
fig.patch.set_facecolor('#f7f7f7')

# Show plot
plt.show()

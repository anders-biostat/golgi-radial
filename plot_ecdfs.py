import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the results
results = pd.read_pickle('results.pkl')
print(f'Loaded {len(results)} cells from {results["filename"].nunique()} files')

# Check conditions
print(f'Conditions: {results["cond"].value_counts().to_dict()}')

# Set up the plot
plt.figure(figsize=(12, 8))

# Set up colors for each condition
conditions = results['cond'].unique()
colors = plt.cm.Set1(np.linspace(0, 1, len(conditions)))
condition_colors = dict(zip(conditions, colors))

# Plot all ECDF curves
for idx, row in results.iterrows():
    if len(row['statistic']) > 0:  # Make sure we have data
        ecdf = row['statistic']
        x_values = np.arange(1, len(ecdf) + 1)  # 1-based indexing like R
        
        # Limit to x-axis range 0-200
        mask = x_values <= 200
        x_plot = x_values[mask]
        y_plot = ecdf[mask] if len(ecdf) > len(x_plot) else ecdf[:len(x_plot)]
        
        # Plot with condition-specific color
        plt.plot(x_plot, y_plot, 
                color=condition_colors[row['cond']], 
                alpha=0.3, 
                linewidth=0.5)

# Add legend by plotting one line per condition
legend_lines = []
legend_labels = []
for cond in conditions:
    line = plt.plot([], [], color=condition_colors[cond], linewidth=2, label=cond)[0]
    legend_lines.append(line)
    legend_labels.append(f"{cond} (n={sum(results['cond'] == cond)})")

plt.legend(legend_lines, legend_labels, loc='lower right')
plt.xlim(0, 200)
plt.ylim(0, 1)
plt.xlabel('Radius (pixels)')
plt.ylabel('Cumulative Distribution')
plt.title('Radial ECDF Curves by Condition')
plt.grid(True, alpha=0.3)

# Save the plot
plt.tight_layout()
plt.savefig('ecdf_curves_by_condition.png', dpi=150, bbox_inches='tight')
plt.close()

print("ECDF plot saved as ecdf_curves_by_condition.png")

# Also create a summary plot showing mean ECDF per condition
plt.figure(figsize=(12, 8))

for cond in conditions:
    subset = results[results['cond'] == cond]
    
    # Collect all ECDFs for this condition
    ecdfs = []
    for _, row in subset.iterrows():
        if len(row['statistic']) > 0:
            ecdf = row['statistic']
            # Pad or truncate to 200 points
            if len(ecdf) >= 200:
                ecdfs.append(ecdf[:200])
            else:
                # Pad with the last value
                padded = np.concatenate([ecdf, np.full(200 - len(ecdf), ecdf[-1] if len(ecdf) > 0 else 0)])
                ecdfs.append(padded)
    
    if len(ecdfs) > 0:
        ecdfs_array = np.array(ecdfs)
        mean_ecdf = np.mean(ecdfs_array, axis=0)
        std_ecdf = np.std(ecdfs_array, axis=0)
        
        x_values = np.arange(1, 201)
        
        # Plot mean with confidence band
        plt.plot(x_values, mean_ecdf, color=condition_colors[cond], linewidth=2, label=f"{cond} (n={len(ecdfs)})")
        plt.fill_between(x_values, 
                        mean_ecdf - std_ecdf, 
                        mean_ecdf + std_ecdf, 
                        color=condition_colors[cond], 
                        alpha=0.2)

plt.xlim(0, 200)
plt.ylim(0, 1)
plt.xlabel('Radius (pixels)')
plt.ylabel('Cumulative Distribution')
plt.title('Mean Radial ECDF Curves by Condition (with ±1 SD)')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mean_ecdf_curves_by_condition.png', dpi=150, bbox_inches='tight')
plt.close()

print("Mean ECDF plot saved as mean_ecdf_curves_by_condition.png")
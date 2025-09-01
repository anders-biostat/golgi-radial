import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the results
results = pd.read_pickle('results.pkl')
print(f'Loaded {len(results)} cells from {results["filename"].nunique()} files')

# Function to find 70% quantile from ECDF
def find_70pct_quantile(ecdf):
    """Find the radius where ECDF reaches 0.7 (70%)"""
    if len(ecdf) == 0:
        return np.nan
    
    # Find first index where ECDF >= 0.7
    indices = np.where(ecdf >= 0.7)[0]
    if len(indices) == 0:
        # If never reaches 0.7, return the maximum radius
        return len(ecdf)
    else:
        # Return the radius (1-based indexing like R)
        return indices[0] + 1

# Extract 70% quantiles for each cell
quantiles_70pct = []
conditions = []
frames = []
genotypes = []

for idx, row in results.iterrows():
    if len(row['statistic']) > 0:  # Make sure we have data
        q70 = find_70pct_quantile(row['statistic'])
        if not np.isnan(q70):
            quantiles_70pct.append(q70)
            conditions.append(row['cond'])
            frames.append(row['frame'])
            genotypes.append(row['genotype'])

# Create dataframe for plotting
plot_data = pd.DataFrame({
    '70pct_quantile': quantiles_70pct,
    'condition': conditions,
    'frame': frames,
    'genotype': genotypes
})

print(f'Extracted {len(plot_data)} 70% quantiles')
print(f'Conditions: {plot_data["condition"].value_counts().to_dict()}')
print(f'Mean 70% quantiles by condition:')
print(plot_data.groupby('condition')['70pct_quantile'].agg(['mean', 'std', 'count']))

# Create beeswarm plot
plt.figure(figsize=(10, 8))

# Use seaborn for beeswarm plot
sns.swarmplot(data=plot_data, x='condition', y='70pct_quantile', size=5, alpha=0.7)

# Add summary statistics
for i, cond in enumerate(plot_data['condition'].unique()):
    subset = plot_data[plot_data['condition'] == cond]
    mean_val = subset['70pct_quantile'].mean()
    # Add horizontal line for mean
    plt.hlines(mean_val, i-0.25, i+0.25, colors='red', linewidth=2, alpha=0.8)

plt.xlabel('Condition')
plt.ylabel('70% Quantile Radius (pixels)')
plt.title('70% Quantile of Radial ECDF by Condition\n(Red lines show means)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('beeswarm_70pct_quantiles.png', dpi=150, bbox_inches='tight')
plt.close()

print("Beeswarm plot saved as beeswarm_70pct_quantiles.png")

# Also create a violin plot for comparison
plt.figure(figsize=(10, 8))

sns.violinplot(data=plot_data, x='condition', y='70pct_quantile', inner='box')
sns.swarmplot(data=plot_data, x='condition', y='70pct_quantile', size=3, alpha=0.6, color='white')

plt.xlabel('Condition')
plt.ylabel('70% Quantile Radius (pixels)')
plt.title('70% Quantile of Radial ECDF by Condition\n(Violin + Swarm Plot)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('violin_70pct_quantiles.png', dpi=150, bbox_inches='tight')
plt.close()

print("Violin plot saved as violin_70pct_quantiles.png")

# Print some summary statistics
print("\nSummary statistics:")
summary = plot_data.groupby('condition')['70pct_quantile'].describe()
print(summary)
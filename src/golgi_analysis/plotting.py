from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIGURATION & PALETTES
# =============================================================================

GENOTYPE_COLORS = {
    'WT': '#FF99FF',
    'CEP68KO': '#5757F9',
    'CEP128KO': '#FD8008',
    'CEP250KO': '#99FFCC',
    'CEP128250dKO': '#336666',
    'CEP68128dKO': '#666699',
    'CROCCKO': '#003366',
    'NINKO': '#CCCCFF',
    'CEP250rescue': '#005A00',
    'AKAPKO': '#800080'
}

DEFAULT_GENOTYPE_ORDER = [
    'WT', 'CEP68KO', 'CEP128KO', 'CEP250KO', 'CEP250rescue',
    'CEP68128dKO', 'CEP128250dKO', 'AKAPKO', 'CROCCKO', 'NINKO'
]

DEFAULT_CONDITION_ORDER = ["nodrug", "2hdrug", "30minWO", "2hWO"]

FIGURE_SIZE = (12, 6)
PLOT_DPI = 300


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_numpy_array(val) -> np.ndarray:
    """Parses string representations or existing NumPy arrays safely."""
    if isinstance(val, np.ndarray):
        return val
    if isinstance(val, (list, tuple)):
        return np.array(val, dtype=np.float64)
    if isinstance(val, str):
        cleaned = val.replace('[', '').replace(']', '').replace(',', ' ').strip()
        if not cleaned:
            return np.array([], dtype=np.float64)
        try:
            return np.array([float(x) for x in cleaned.split()], dtype=np.float64)
        except ValueError:
            return np.array([], dtype=np.float64)
    return np.array([], dtype=np.float64)


def _plot_categorical_distribution(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_order: list,
    color_or_palette,
    title: str,
    ylabel: str,
    save_path: Path,
    plot_type: str = 'swarm',
    fig_size: tuple = FIGURE_SIZE,
    dpi: int = PLOT_DPI
):
    """Core rendering function powering both Function 2 and Function 3."""
    plt.figure(figsize=fig_size)

    if plot_type == 'swarm':
        if isinstance(color_or_palette, dict):
            sns.swarmplot(
                data=data, x=x_col, y=y_col, order=x_order,
                palette=color_or_palette, size=3.5, alpha=0.75
            )
        else:
            sns.swarmplot(
                data=data, x=x_col, y=y_col, order=x_order,
                color=color_or_palette, size=5, alpha=0.8
            )
    elif plot_type == 'violin':
        color = color_or_palette if isinstance(color_or_palette, str) else None
        sns.violinplot(
            data=data, x=x_col, y=y_col, order=x_order,
            color=color, inner=None, alpha=0.3
        )
        sns.swarmplot(
            data=data, x=x_col, y=y_col, order=x_order,
            color='white', edgecolor='black', linewidth=0.5, size=4, alpha=0.9
        )

    # Overlay Mean Bars (Red Lines)
    for i, cat in enumerate(x_order):
        subset = data[data[x_col] == cat]
        if not subset.empty:
            mean_val = subset[y_col].mean()
            plt.hlines(mean_val, xmin=i - 0.3, xmax=i + 0.3, colors='red', linewidth=2.5, zorder=10)

    plt.xlabel(x_col.replace('_', ' ').title(), fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path.name}")


# =============================================================================
# PUBLIC PLOTTING FUNCTIONS
# =============================================================================

def plot_mean_curves(
    df: pd.DataFrame,
    output_dir: Path,
    mode: str = 'Centrosome',
    target_conditions: list = None,
    ecdf_xlim: int = 100,
    sd_band: float = 2.0
):
    """
    Function 1: Plots Mean Radial ECDF curves +/- SD comparing GENOTYPES for conditions.
    """
    df = df.copy()
    df['statistic'] = df['statistic'].apply(parse_numpy_array)

    conditions = target_conditions or df['cond'].unique()

    for cond in conditions:
        subset_cond = df[df['cond'] == cond]
        if subset_cond.empty:
            continue

        genotypes = sorted(subset_cond['genotype'].unique())
        plt.figure(figsize=(10, 8))

        for geno in genotypes:
            subset_geno = subset_cond[subset_cond['genotype'] == geno]
            color = GENOTYPE_COLORS.get(geno, 'gray')

            ecdfs_list = []
            for arr in subset_geno['statistic']:
                if len(arr) == 0:
                    continue
                arr_cut = arr[:ecdf_xlim]
                if len(arr_cut) < ecdf_xlim:
                    padding = np.full(ecdf_xlim - len(arr_cut), arr_cut[-1])
                    arr_cut = np.concatenate([arr_cut, padding])
                ecdfs_list.append(arr_cut)

            if ecdfs_list:
                matrix = np.array(ecdfs_list)
                mean_curve = np.mean(matrix, axis=0)
                std_curve = np.std(matrix, axis=0)
                x_axis = np.arange(1, ecdf_xlim + 1)

                plt.plot(x_axis, mean_curve, color=color, linewidth=2.5, label=f"{geno} (n={len(subset_geno)})")

                lower = np.maximum(mean_curve - (sd_band * std_curve), 0)
                upper = np.minimum(mean_curve + (sd_band * std_curve), 1)
                plt.fill_between(x_axis, lower, upper, color=color, alpha=0.15)

        plt.xlim(0, ecdf_xlim)
        plt.ylim(0, 1)
        plt.xlabel('Radius (pixels)', fontsize=12)
        plt.ylabel('Cumulative Distribution', fontsize=12)
        plt.title(f'Mean Radial ECDF: {cond} ({mode.title()})\nShaded area: ±{sd_band} SD', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = output_dir / f'Genotype_Comparison_{mode.title()}_{cond}_updated.png'
        plt.savefig(save_path, dpi=PLOT_DPI)
        plt.close()
        print(f"Saved: {save_path.name}")


def plot_beeswarm_cross_genotype(
    df: pd.DataFrame,
    output_dir: Path,
    mode: str = 'Centrosome',
    target_conditions: list = None
):
    """
    Function 2: Compares GENOTYPES on X-axis for a given experimental Condition.
    """
    df = df.copy()
    if 'std_dev' not in df.columns and 'variance' in df.columns:
        df['std_dev'] = np.sqrt(df['variance'])

    conditions = target_conditions or df['cond'].unique()

    metrics = [
        {'col': 'std_dev', 'title': 'Standard Deviation', 'ylabel': 'Standard Deviation (pixels)'},
        {'col': '70pct_quantile', 'title': '70% Quantile', 'ylabel': '70% Quantile Radius (pixels)'}
    ]

    for cond in conditions:
        subset = df[df['cond'] == cond]
        if subset.empty:
            continue

        present_genotypes = list(subset['genotype'].unique())
        final_order = [g for g in DEFAULT_GENOTYPE_ORDER if g in present_genotypes] + \
                      [g for g in present_genotypes if g not in DEFAULT_GENOTYPE_ORDER]

        for m in metrics:
            col_name = m['col']
            if col_name not in subset.columns:
                continue

            clean_subset = subset.dropna(subset=[col_name])
            if clean_subset.empty:
                continue

            save_path = output_dir / f'Beeswarm_{m["title"].replace(" ", "")}_{mode.title()}_{cond}.png'

            _plot_categorical_distribution(
                data=clean_subset,
                x_col='genotype',
                y_col=col_name,
                x_order=final_order,
                color_or_palette=GENOTYPE_COLORS,
                title=f'{m["title"]} by Genotype: {cond} ({mode.title()})',
                ylabel=m['ylabel'],
                save_path=save_path,
                plot_type='swarm'
            )


def plot_beeswarm_by_genotype(
    df: pd.DataFrame,
    output_dir: Path,
    mode: str = 'Centrosome',
    plot_violins: bool = False
):
    """
    Function 3: Compares CONDITIONS on X-axis for each Genotype.
    """
    df = df.copy()
    if 'std_dev' not in df.columns and 'variance' in df.columns:
        df['std_dev'] = np.sqrt(df['variance'])

    unique_genotypes = df['genotype'].unique()

    metrics = [
        {'col': 'std_dev', 'title': 'StdDev', 'ylabel': 'Standard Deviation (pixels)'},
        {'col': '70pct_quantile', 'title': 'Quantile', 'ylabel': '70% Quantile Radius (pixels)'}
    ]

    for genotype in unique_genotypes:
        subset = df[df['genotype'] == genotype]
        if subset.empty:
            continue

        geno_color = GENOTYPE_COLORS.get(genotype, '#808080')
        present_conds = [c for c in DEFAULT_CONDITION_ORDER if c in subset['cond'].unique()]

        for m in metrics:
            col_name = m['col']
            if col_name not in subset.columns:
                continue

            clean_subset = subset.dropna(subset=[col_name])
            if clean_subset.empty:
                continue

            # Beeswarm Plot
            save_path = output_dir / f'Beeswarm_{m["title"]}_{genotype}.png'
            _plot_categorical_distribution(
                data=clean_subset,
                x_col='cond',
                y_col=col_name,
                x_order=present_conds,
                color_or_palette=geno_color,
                title=f'Radial {m["title"]}: {genotype} ({mode.title()})',
                ylabel=m['ylabel'],
                save_path=save_path,
                plot_type='swarm'
            )

            # Optional Violin Plot
            if plot_violins and col_name == '70pct_quantile':
                v_save_path = output_dir / f'Violin_{m["title"]}_{genotype}.png'
                _plot_categorical_distribution(
                    data=clean_subset,
                    x_col='cond',
                    y_col=col_name,
                    x_order=present_conds,
                    color_or_palette=geno_color,
                    title=f'Violin Distribution ({m["title"]}): {genotype} ({mode.title()})',
                    ylabel=m['ylabel'],
                    save_path=v_save_path,
                    plot_type='violin'
                )
from typing import Dict
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the iris dataset, it's just a dataframe! Seaborn maintains some public datasets to go with their plotting examples
df: pd.DataFrame = sns.load_dataset('iris')

# Create the FacetGrid with species as the facet (similar to df.groupby(species))
g: sns.FacetGrid = sns.FacetGrid(df, col="species")

# For each facet in the grid, create a scatter plot using the resulting sub_df.
# This works similar to df.groupby(species).apply(lambda sub_df: plt.scatter(sub_df, 'sepal_length', 'sepal_width')), but
# the grid knows how to maintain the plot grouping instead of having N individual plots.
g.map(plt.scatter, "sepal_length", "sepal_width")

# You can add a title to the whole plot through g.fig.suptitle
# g.fig is a matplotlib.figure.Figure
g.fig.suptitle("Sepal Length vs Width by Species in the Iris Dataset", fontsize=16)
g.fig.subplots_adjust(top=0.75)  # Adjust the top to make room for the title

# Save the plot
plt.savefig('iris_facet_grid_example.png')


# =================== A Second Example: Adding per-facet colors =====================

# Create the FacetGrid with species as the facet and specify colors
palette: Dict[str, str] = {'setosa': 'green', 'versicolor': 'purple', 'virginica': 'orange'}

# hue is used to control color by some attribute, and pallate will look up the value of that attribute in pallete to pick a color
# so make sure pallette has a key for all facets
g: sns.FacetGrid = sns.FacetGrid(df, col="species", hue='species', palette=palette)

# Map a scatter plot for each species
g.map(plt.scatter, "sepal_length", "sepal_width")

# Adding an informative title
g.fig.suptitle("Sepal Length vs Width by Species in the Iris Dataset", fontsize=16)
g.fig.subplots_adjust(top=0.75)  # Adjust the top to make room for the title

# Adding a legend
g.add_legend()

plt.savefig('iris_facet_grid_example_multicolor.png')
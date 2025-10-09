# matplot3.py

`matplot3.py` is a versatile command-line tool for generating various types of plots from standard input data. It leverages `matplotlib`, `typer`, and other scientific Python libraries to provide functionalities for plotting time-series, scatter plots, histograms, jitter plots, ROC curves, UMAP embeddings, chord diagrams, and Venn diagrams.

## Installation

This script relies on several Python packages. You can install them using pip:

```bash
pip install typer rich numpy scipy scikit-learn pillow umap-learn pycirclize pyvenn
```

Note: `pycirclize` and `pyvenn` are specifically required for the `chord-diagram` and `venn-diagram` commands, respectively. If you don't plan to use these commands, you can omit their installation.

## Usage

The script is built with `typer` and offers several subcommands for different plot types.

To get general help:
```bash
./matplot3.py --help
```

To get help for a specific subcommand, e.g., `plot`:
```bash
./matplot3.py plot --help
```

### Global Options (plot_setup)

These options apply globally to all commands and can be set before any subcommand:

*   `--xlabel <TEXT>`: The label for the x-axis (default: "x")
*   `--ylabel <TEXT>`: The label for the y-axis (default: "y")
*   `--semilog-x`: If True, set the x-axis to a logarithmic scale.
*   `--semilog-y`: If True, set the y-axis to a logarithmic scale.
*   `--grid`: If True, display a grid on the plot.
*   `--aspect-ratio <TEXT>`: Set the figure size (e.g., '10 5' for 10x5 inches).
*   `--subplots <TEXT>`: The number of subplots in the format "rows columns" (default: "1 1")
*   `--sharex`: If True, share the x-axis among subplots.
*   `--sharey`: If True, share the y-axis among subplots.
*   `--titles <TEXT>`: The titles for the subplots, separated by spaces.
*   `--debug`: If True, enable debug mode.

### Commands

#### `plot`
Plot data from standard input.

**Options:**
*   `--fields <TEXT>`: x: The x field, y: The y field, xt: The xtick labels field, ts: The x field is a timestamp (in seconds since epoch) (default: "x y")
*   `--labels <TEXT>`: The labels to use for the data (default: "")
*   `--moving-avg <INTEGER>`: The size of the moving average window (default: 0)
*   `--delimiter <TEXT>`: The delimiter to use to split the data
*   `--fmt <TEXT>`: The format string to use for the plot (default: "")
*   `--alpha <FLOAT>`: The alpha value for the plot (default: 1.0)
*   `--rotation <INTEGER>`: The rotation of the xtick labels in degrees (default: 45)
*   `--save <TEXT>`: The filename to save the plot to (default: "")
*   `--xmin <FLOAT>`: The minimum x value for the plot
*   `--xmax <FLOAT>`: The maximum x value for the plot
*   `--ymin <FLOAT>`: The minimum y value for the plot
*   `--ymax <FLOAT>`: The maximum y value for the plot
*   `--shade <TEXT>`: Give 0 (no shade) or 1 (shade) to shade the area under the curve. Give 1 value per y field. e.g. if --fields x y y, shade can be 0 1 to only shade the area under the second y field
*   `--alpha-shade <FLOAT>`: The alpha value for the shaded area (default: 0.2)
*   `--test`: Generate random data for testing
*   `--test-npts <INTEGER>`: The number of points to generate for testing (default: 1000)
*   `--test-ndata <INTEGER>`: The number of datasets to generate for testing (default: 2)
*   `--equal-aspect`: Set the aspect ratio of the plot to equal

#### `scatter`
Create a scatter plot from data in standard input.

**Options:**
*   `--fields <TEXT>`: x: The x field, y: The y field, c: A sequence of numbers to be mapped to colors using cmap (see: --cmap), s: The marker size in points**2, il: a particular field with labels to display for interactive mode, t: a field with text labels to display on the plot, xt: the xticks labels (default: "x y")
*   `--labels <TEXT>`: The labels to use for the data (default: "")
*   `--delimiter <TEXT>`: The delimiter to use to split the data
*   `--alpha <FLOAT>`: The alpha value for the plot (default: 1.0)
*   `--cmap <TEXT>`: The colormap to use for the plot (default: "viridis")
*   `--pcr`: Principal component regression (see: https://en.wikipedia.org/wiki/Principal_component_regression)
*   `--save <TEXT>`: The filename to save the plot to (default: "")
*   `--xmin <FLOAT>`: The minimum x value for the plot
*   `--xmax <FLOAT>`: The maximum x value for the plot
*   `--ymin <FLOAT>`: The minimum y value for the plot
*   `--ymax <FLOAT>`: The maximum y value for the plot
*   `--colorbar`: Add a colorbar to the plot
*   `--test`: Generate random data for testing
*   `--test-npts <INTEGER>`: The number of points to generate for testing (default: 1000)
*   `--test-ndata <INTEGER>`: The number of datasets to generate for testing (default: 2)
*   `--equal-aspect`: Set the aspect ratio of the plot to equal

#### `hist`
Create a histogram from data in standard input.

**Options:**
*   `--fields <TEXT>`: The fields to read (default: "y")
*   `--labels <TEXT>`: The labels to use for the data (default: "")
*   `--delimiter <TEXT>`: The delimiter to use to split the data
*   `--bins <TEXT>`: The number of bins to use for the histogram (default: "auto")
*   `--alpha <FLOAT>`: The alpha value for the plot (default: 1.0)
*   `--density`: Normalize the histogram
*   `--save <TEXT>`: The filename to save the plot to (default: "")
*   `--xmin <FLOAT>`: The minimum x value for the plot
*   `--xmax <FLOAT>`: The maximum x value for the plot
*   `--ymin <FLOAT>`: The minimum y value for the plot
*   `--ymax <FLOAT>`: The maximum y value for the plot
*   `--test`: Generate random data for testing
*   `--test-npts <INTEGER>`: The number of points to generate for testing (default: 1000)
*   `--test-ndata <INTEGER>`: The number of datasets to generate for testing (default: 2)
*   `--equal-aspect`: Set the aspect ratio of the plot to equal

#### `jitter`
Create a jitter plot from data in standard input.

**Options:**
*   `--fields <TEXT>`: x: The x field, y: The y field, xt: The xtick labels field, c: The color field, il: The interactive labels field (default: "x y")
*   `--labels <TEXT>`: The labels to use for the data (default: "")
*   `--delimiter <TEXT>`: The delimiter to use to split the data
*   `--xjitter <FLOAT>`: The amount of jitter to add to the x values (default: 0.1)
*   `--yjitter <FLOAT>`: The amount of jitter to add to the y values (default: 0.0)
*   `--size <INTEGER>`: The size of the markers in the plot (default: 10)
*   `--alpha <FLOAT>`: The alpha value for the plot (default: 1.0)
*   `--kde`: Use kernel density estimation to color the points
*   `--kde-subset <INTEGER>`: The number of points to use for the KDE (default: 1000)
*   `--kde-normalize`: Normalize the KDE values
*   `--cmap <TEXT>`: The colormap to use for the plot (default: "viridis")
*   `--median`: Plot the median of the data
*   `--median-size <INTEGER>`: The size of the median markers in the plot (default: 100)
*   `--median-color <TEXT>`: The color of the median markers in the plot (default: "black")
*   `--median-marker <TEXT>`: The marker to use for the median markers in the plot (default: "_")
*   `--median-sort`: Sort by median values
*   `--save <TEXT>`: The filename to save the plot to (default: "")
*   `--xmin <FLOAT>`: The minimum x value for the plot
*   `--xmax <FLOAT>`: The maximum x value for the plot
*   `--ymin <FLOAT>`: The minimum y value for the plot
*   `--ymax <FLOAT>`: The maximum y value for the plot
*   `--rotation <INTEGER>`: The rotation of the xtick labels in degrees (default: 45)
*   `--colorbar`: Add a colorbar to the plot
*   `--cbar-label <TEXT>`: The label for the colorbar
*   `--test`: Generate random data for testing
*   `--test-npts <INTEGER>`: The number of points to generate for testing (default: 1000)
*   `--test-ndata <INTEGER>`: The number of datasets to generate for testing (default: 3)
*   `--equal-aspect`: Set the aspect ratio of the plot to equal

#### `roc`
Create a ROC curve from data in standard input.

**Options:**
*   `--fields <TEXT>`: y: The value (the lower the better by default), a: 1 for active, 0 for inactive (default: "y a")
*   `--labels <TEXT>`: The labels to use for the data (default: "")
*   `--delimiter <TEXT>`: The delimiter to use to split the data
*   `--save <TEXT>`: The filename to save the plot to (default: "")
*   `--xmin <FLOAT>`: The minimum x value for the plot (default: 0.0)
*   `--xmax <FLOAT>`: The maximum x value for the plot (default: 1.0)
*   `--ymin <FLOAT>`: The minimum y value for the plot (default: 0.0)
*   `--ymax <FLOAT>`: The maximum y value for the plot (default: 1.0)
*   `--test`: Generate random data for testing
*   `--test-npts <INTEGER>`: The number of points to generate for testing (default: 1000)
*   `--test-ndata <INTEGER>`: The number of datasets to generate for testing (default: 2)
*   `--equal-aspect`: Set the aspect ratio of the plot to equal

#### `umap`
Create a UMAP plot from data in standard input.

**Options:**
*   `--n-neighbors <INTEGER>`: The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation (default: 15)
*   `--min-dist <FLOAT>`: The effective minimum distance between embedded points (default: 0.1)
*   `--metric <TEXT>`: The metric to use to compute distance in high dimensional space (default: euclidean, precomputed, cosine, manhattan, hamming, etc.) (default: "euclidean")
*   `--test`: Generate random data for testing
*   `--save <TEXT>`: The filename to save the plot to (default: "")
*   `--npy <TEXT>`: Load data from a numpy file (default: "")
*   `--npz <TEXT>`: Load data from a numpy file (compressed) (default: "")
*   `--data-key <TEXT>`: The key to use to load data from the npz file (default: "data")
*   `--labels-key <TEXT>`: The key to use to load labels from the npz file (default: "")
*   `--ilabels-key <TEXT>`: The key to use to load interactive labels from the npz file (default: "")
*   `--legend`: Add a legend to the plot (default: True)
*   `--colorbar`: Add a colorbar to the plot
*   `--cmap <TEXT>`: The colormap to use for the plot (default: "viridis")
*   `--size <INTEGER>`: The size of the markers in the plot (default: 10)
*   `--alpha <FLOAT>`: The transparency of the markers in the plot (default: 1.0)
*   `--xmin <FLOAT>`: The minimum x value for the plot
*   `--xmax <FLOAT>`: The maximum x value for the plot
*   `--ymin <FLOAT>`: The minimum y value for the plot
*   `--ymax <FLOAT>`: The maximum y value for the plot

#### `read-metadata`
Read metadata from a PNG file.

**Options:**
*   `--filename <TEXT>`: The filename to read the metadata from

#### `chord-diagram`
Create a chord diagram from data in standard input.

**Options:**
*   `--fields <TEXT>`: d: The data field (matrix values), r: The row labels field, c: The column labels field (default: "d r c")
*   `--labels <TEXT>`: The labels to use for the data (default: "")
*   `--delimiter <TEXT>`: The delimiter to use to split the data
*   `--test`: If True, generate random data for testing
*   `--save <TEXT>`: The filename to save the plot to (default: "")

#### `venn-diagram`
Create a Venn diagram from data in standard input or generated test data.

**Options:**
*   `--fields <TEXT>`: d: The data field (set components), l: The set label field (Unique label for each set, maximum 6 labels, 6 sets) (default: "d l")
*   `--labels-fill <TEXT>`: Comma-separated options for filling labels: 'number', 'logic', 'percent', 'elements'. E.g., 'number,percent' (default: "number")
*   `--save <TEXT>`: The filename to save the plot to (default: "")
*   `--test`: If True, generate random data for testing
*   `--test-ndata <INTEGER>`: The number of sets to generate for testing (2 to 6) (default: 3)
*   `--test-npts <INTEGER>`: The number of points in each test set (default: 10)
*   `--delimiter <TEXT>`: The delimiter to use to split the data
*   `--colors <TEXT>`: Comma-separated list of colors (e.g., 'red,blue,green'). Uses default colors if not specified. (default: "")
*   `--figsize <TEXT>`: Figure size in inches (e.g., '9 7'). Uses default if not specified. (default: "")
*   `--dpi <INTEGER>`: Resolution of the figure in dots per inch. (default: 96)
*   `--fontsize <INTEGER>`: Font size for labels. (default: 14)
*   `--sortkey <INTEGER>`: Key to sort elements in 'elements' fill option. Defaults to 0 for string slicing. (default: 0)

## Examples

### Plot a simple line graph
```bash
echo "1 10\n2 12\n3 5" | ./matplot3.py plot
```

### Scatter plot with custom labels and PCR
```bash
echo "1 10\n2 12\n3 5" | ./matplot3.py scatter --pcr --labels "MyData"
```

### Jitter plot with KDE coloring
```bash
seq 1 100 | awk '{print ($1%2) " " ($1*0.1 + rand())}' | ./matplot3.py jitter --kde
```

### ROC curve
```bash
# Example data for ROC: value and activity (1 or 0)
echo "0.1 1\n0.2 0\n0.3 1\n0.4 1\n0.5 0" | ./matplot3.py roc
```

### UMAP from test data
```bash
./matplot3.py umap --test --save umap_test.png
```

### Chord Diagram with test data
```bash
./matplot3.py chord-diagram --test --save chord_test.png
```

### Venn Diagram with test data
```bash
./matplot3.py venn-diagram --test --test-ndata 3 --labels-fill "number,percent" --save venn_test.png
```

### Read metadata from a plot
```bash
# First create a plot and save it as PNG
echo "1 10\n2 12\n3 5" | ./matplot3.py plot --save test.png
# Then read its metadata
./matplot3.py read-metadata --filename test.png
```

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Define custom color palette
colors = ['#a0c5d8', '#ee80a3', '#8aaa5a', '#bdd3e9', '#e0b8f0', '#6bba3c', '#ece0cf', '#e4e5e6']  # blue, pink, green, blue_2, red_2, green_2, beige, light_grey

def plot_box_plot(
    df, 
    value_col, 
    category_col=None, 
    custom_legend_names=None, 
    color_map=None,
    label_font_size=11, 
    y_tick_font_size=11, 
    y_tick_intervals=(0, 900, 100),
    plot_title=None, 
    y_label=None,
    box_color='lightgrey',       # Fill color for box plots
    edge_color='black'           # Edge color for box plots
    ):
    """
    Plots comparative box plots for different categories in one panel.
    It supports value distribution for multiple categories like city/countryside or renovated/unrenovated.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with value data (e.g., price).
    - value_col (str): Column name for the value to plot (e.g., price).
    - category_col (str): Column name for the category (e.g., city/countryside or renovation status). Default is None.
    - custom_legend_names (dict): Custom names for the categories. Default is None.
    - color_map (dict): Custom colors for the different categories. Default is None.
    - label_font_size (int): Font size for the y-axis labels. Default is 11.
    - y_tick_font_size (int): Font size for the y-axis tick labels. Default is 11.
    - y_tick_intervals (tuple): Tick interval (start, end, step) for the y-axis.
    - plot_title (str): Title for the plot.
    - y_label (str): Custom label for the y-axis. Default is None.
    - box_color (str): Fill color for the box plots. Default is 'lightgrey'.
    - edge_color (str): Edge color for the box plots. Default is 'black'.

    Returns:
    - None: The function creates and shows the box plots.
    """
    
    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Ensure that df contains no missing values in the relevant columns
    if category_col:
        df_copy = df_copy.dropna(subset=[value_col, category_col])
    else:
        df_copy = df_copy.dropna(subset=[value_col])

    # If a category column is provided, plot based on categories
    if category_col:
        # Replace category names with custom legend names if provided
        if custom_legend_names:
            df_copy[category_col] = df_copy[category_col].map(custom_legend_names)
        
        # Get unique categories for plotting
        unique_categories = df_copy[category_col].unique()

        # Handle the color map. If none is provided, set all colors to box_color (lightgrey)
        if color_map is None:
            palette_colors = [box_color] * len(unique_categories)
            color_map = {category: box_color for category in unique_categories}
        else:
            # Ensure all categories have a defined color
            if custom_legend_names:
                # Update the color map to match the custom legend names
                color_map = {custom_legend_names.get(key, key): color for key, color in color_map.items()}

            # Check if all categories are present in the color_map
            if not all(cat in color_map for cat in unique_categories):
                raise ValueError(f"color_map must include colors for all categories: {unique_categories}")
            
            # Assign colors based on the color_map
            palette_colors = [color_map[cat] for cat in unique_categories]

        # Create the figure and axes
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot the boxplot for each category with hue assigned
        sns.boxplot(
            x=category_col,
            y=value_col,
            hue=category_col,  # Assign 'hue' to 'category_col'
            data=df_copy,
            palette=palette_colors,  # Apply the light grey palette
            ax=ax,
            linewidth=1,
            flierprops=dict(
                marker='.', 
                markerfacecolor=edge_color, 
                markeredgecolor=edge_color, 
                markersize=6,
                linestyle='none'
            ),
            whiskerprops=dict(color=edge_color, linewidth=1),
            medianprops=dict(color=edge_color, linewidth=2),
            showcaps=False,
            width=0.2
        )
        
        # Remove the redundant legend if it exists
        if ax.get_legend() is not None:
            ax.get_legend().remove()

        # Set the x-axis label to empty as it's already clear from the categories
        ax.set_xlabel('')  # This removes the x-axis label
        
        # Set the y-axis limits and ticks
        start, end, step = y_tick_intervals
        ax.set_ylim(start, end)
        ax.set_yticks(range(start, end + 1, step))

        # Set y-axis tick label font size
        ax.tick_params(axis='y', labelsize=y_tick_font_size)

        # Set y-axis label
        if y_label:
            ax.set_ylabel(y_label, fontsize=label_font_size)
        else:
            ax.set_ylabel(value_col.replace('_', ' ').title(), fontsize=label_font_size)

        # Set the plot title if provided
        if plot_title:
            ax.set_title(plot_title, fontsize=label_font_size + 2)

        # Remove the top and right axes for a cleaner look
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Set the linewidth of the left and bottom axes
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)

    else:
        # If no category column is provided, just plot a single box plot for the entire dataset
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot the boxplot
        sns.boxplot(
            y=value_col,
            data=df_copy,
            boxprops=dict(edgecolor=edge_color, facecolor=box_color),  # Use facecolor and edgecolor
            ax=ax,
            linewidth=1,
            flierprops=dict(
                marker='.', 
                markerfacecolor=edge_color, 
                markeredgecolor=edge_color, 
                markersize=6,
                linestyle='none'
            ),
            whiskerprops=dict(color=edge_color, linewidth=1),
            medianprops=dict(color=edge_color, linewidth=2),
            showcaps=False,
            width=0.2
        )
        
        # Set the y-axis limits and ticks
        start, end, step = y_tick_intervals
        ax.set_ylim(start, end)
        ax.set_yticks(range(start, end + 1, step))

        # Set y-axis tick label font size
        ax.tick_params(axis='y', labelsize=y_tick_font_size)

        # Set y-axis label
        if y_label:
            ax.set_ylabel(y_label, fontsize=label_font_size)
        else:
            ax.set_ylabel(value_col.replace('_', ' ').title(), fontsize=label_font_size)

        # Set the plot title if provided
        if plot_title:
            ax.set_title(plot_title, fontsize=label_font_size + 2)

        # Remove the top and right axes
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Set the linewidth of the left and bottom axes
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_violin_plot(
    df, 
    value_col, 
    category_col=None, 
    custom_legend_names=None, 
    color_map=None,
    label_font_size=11, 
    y_tick_font_size=11, 
    y_tick_intervals=(0, 900, 100),
    plot_title=None, 
    scale='width', 
    split=False,
    violin_width=0.3, 
    y_label=None,
    box_color='lightgrey',       # Fill color for violins
    edge_color='black'           # Edge color for violins
    ):
    """
    Plots comparative violin plots for different categories in one panel.
    It supports value distribution for multiple categories like city/countryside or renovated/unrenovated.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with value data (e.g., price).
    - value_col (str): Column name for the value to plot (e.g., price).
    - category_col (str): Column name for the category (e.g., city/countryside or renovation status). Default is None.
    - custom_legend_names (dict): Custom names for the categories. Default is None.
    - color_map (dict): Custom colors for the different categories. Default is None.
    - label_font_size (int): Font size for the y-axis labels. Default is 11.
    - y_tick_font_size (int): Font size for the y-axis tick labels. Default is 11.
    - y_tick_intervals (tuple): Tick interval (start, end, step) for the y-axis.
    - plot_title (str): Title for the plot.
    - scale (str): Determines the method for the width of the violins. Default is 'width'.
                     Other options are 'area', 'count'.
    - split (bool): If True, it splits the violins when the hue is used.
    - violin_width (float): Controls the width of the violins. Default is 0.3.
    - y_label (str): Custom label for the y-axis. Default is None.
    - box_color (str): Fill color for the violins. Default is 'lightgrey'.
    - edge_color (str): Edge color for the violins. Default is 'black'.

    Returns:
    - None: The function creates and shows the violin plots.
    """
    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Ensure that df contains no missing values in the relevant columns
    if category_col:
        df_copy = df_copy.dropna(subset=[value_col, category_col])
    else:
        df_copy = df_copy.dropna(subset=[value_col])

    # If a category column is provided, plot based on categories
    if category_col:
        # Replace category names with custom legend names if provided
        if custom_legend_names:
            df_copy[category_col] = df_copy[category_col].map(custom_legend_names)
        
        # Get unique categories for plotting
        unique_categories = df_copy[category_col].unique()

        # Handle the color map. If none is provided, set all colors to box_color (lightgrey)
        if color_map is None:
            palette_colors = [box_color] * len(unique_categories)
            color_map = {category: box_color for category in unique_categories}
        else:
            # Ensure all categories have a defined color
            if custom_legend_names:
                # Update the color map to match the custom legend names
                color_map = {custom_legend_names.get(key, key): color for key, color in color_map.items()}

            if not all(cat in color_map for cat in unique_categories):
                raise ValueError(f"color_map must include colors for all categories: {unique_categories}")
            
            palette_colors = [color_map[cat] for cat in unique_categories]

        # Create the figure and axes
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot the violin plot for each category with hue assigned
        sns.violinplot(
            x=category_col,
            y=value_col,
            hue=category_col,  # Assign 'hue' to 'category_col'
            data=df_copy,
            palette=palette_colors,  # Apply the light grey palette
            ax=ax,
            density_norm='width',  # Replaces 'scale'
            split=split,
            linewidth=1,
            inner=None,  # No inner quartiles or medians
            width=violin_width  # Set the width of the violins
        )

        # Remove the redundant legend if it exists
        if ax.get_legend() is not None:
            ax.get_legend().remove()

        # Set the x-axis label to empty as it's already clear from the categories
        ax.set_xlabel('')  # This removes the x-axis label

        # Set the y-axis limits and ticks
        start, end, step = y_tick_intervals
        ax.set_ylim(start, end)
        ax.set_yticks(range(start, end + 1, step))

        # Set y-axis tick label font size
        ax.tick_params(axis='y', labelsize=y_tick_font_size)

        # Set y-axis label
        if y_label:
            ax.set_ylabel(y_label, fontsize=label_font_size)
        else:
            ax.set_ylabel(value_col.replace('_', ' ').title(), fontsize=label_font_size)

        # Set the plot title if provided
        if plot_title:
            ax.set_title(plot_title, fontsize=label_font_size + 2)

        # Remove the top and right axes for a cleaner look
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Set the linewidth of the left and bottom axes
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)

        # Ensure edges of violins are black
        for violin in ax.findobj(PolyCollection):
            violin.set_edgecolor(edge_color)
            violin.set_linewidth(1)

    else:
        # If no category column is provided, just plot a single violin plot for the entire dataset
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot the violin plot without palette and hue
        sns.violinplot(
            y=value_col,
            data=df_copy,
            color=box_color,  # Use facecolor for single violin
            ax=ax,
            density_norm='width',  # Replaces 'scale'
            linewidth=1,
            inner=None,  # No inner quartiles or medians
            width=violin_width  # Set the width of the violins
        )

        # Customize spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)

        # Set y-axis label
        if y_label:
            ax.set_ylabel(y_label, fontsize=label_font_size)
        else:
            ax.set_ylabel(value_col.replace('_', ' ').title(), fontsize=label_font_size)

        # Set the y-axis limits and ticks
        start, end, step = y_tick_intervals
        ax.set_ylim(start, end)
        ax.set_yticks(range(start, end + 1, step))

        # Set y-axis tick label font size
        ax.tick_params(axis='y', labelsize=y_tick_font_size)

        # Set the plot title if provided
        if plot_title:
            ax.set_title(plot_title, fontsize=label_font_size + 2)

        # Ensure edges of violins are black
        for violin in ax.findobj(PolyCollection):
            violin.set_edgecolor(edge_color)
            violin.set_linewidth(1)

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_histogram(
    df, 
    column, 
    bins=10, 
    color=colors[0],
    x_tick_interval=None, 
    y_tick_interval=None, 
    x_label=None, 
    y_label=None, 
    edgecolor=None, 
    alpha=1.0, 
    rwidth=0.95, 
    linewidth=1,
    figsize=(10, 6),
    transparent=False  # Option to control background transparency
    ):
    """
    Creates a histogram to visualize the distribution of a specific numeric column in the DataFrame using Matplotlib,
    maintaining the same style as the plot_histograms function, with an option for transparency and customizable x-label.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the column to plot.
    column (str): The name of the column to plot.
    bins (int or sequence): Number of bins or the bin edges.
    color (str or tuple): Color of the histogram bars.
    x_tick_interval (tuple): A tuple (start, end, step) for custom x-axis tick intervals. If None, defaults are used.
    y_tick_interval (tuple): A tuple (start, end, step) for custom y-axis tick intervals. If None, defaults are used.
    x_label (str): Custom x-axis label. If None, no x-label is displayed.
    y_label (str): Custom y-axis label. If None, 'Count' is used.
    edgecolor (str or tuple): Color of the edges of the bars. If None, no edge color is applied.
    alpha (float): Transparency level of the bars.
    rwidth (float): Relative width of the bars.
    linewidth (float): Width of the bar edges.
    figsize (tuple): Size of the figure.
    transparent (bool): Whether to make the figure background transparent.

    Returns:
    None: The function creates and shows the histogram.
    """

    # Check if the column exists in the DataFrame
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    # Check if the column is numeric
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"Column '{column}' must be numeric to plot a histogram.")

    # Create a single subplot
    fig, ax = plt.subplots(figsize=figsize)

    # Set figure background transparency if needed
    if transparent:
        fig.patch.set_alpha(0)  # Make the figure background transparent

    # Apply styling similar to plot_histograms
    # Modify X-axis spine to match the grid style
    ax.spines['bottom'].set_color(colors[-1])  # Set X-axis color to light grey
    ax.spines['bottom'].set_linewidth(1)       # Match the grid line width

    # Set ticks on the bottom, with the same style as gridlines
    ax.tick_params(
        axis='x', 
        which='both', 
        bottom=True, 
        top=False, 
        direction='out', 
        length=6, 
        width=1, 
        color=colors[-1]
    )  # Match tick style to grid lines

    # Set ticks on the y-axis, styled the same way as the x-axis
    ax.tick_params(
        axis='y', 
        which='both', 
        left=True, 
        right=False, 
        direction='out', 
        length=6, 
        width=1, 
        color=colors[-1]
    )  # Match tick style to grid lines on y-axis

    # Remove other spines (top, right, left)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Enable only horizontal grid lines
    ax.grid(axis='y', color=colors[-1], linestyle='-', linewidth=1)  # Enable horizontal grid lines only
    ax.grid(axis='x', visible=False)  # Disable vertical grid lines
    ax.set_axisbelow(True)  # Ensure grid is behind the data

    # Plot the histogram
    ax.hist(
        df[column].dropna(),  # Drop NaN values for plotting
        bins=bins, 
        color=color, 
        edgecolor=edgecolor, 
        alpha=alpha, 
        rwidth=rwidth, 
        linewidth=linewidth, 
        zorder=3
    )

    # Set x and y limits if tick intervals are provided
    if x_tick_interval:
        ax.set_xlim(x_tick_interval[0], x_tick_interval[1])
        ax.set_xticks(range(x_tick_interval[0], x_tick_interval[1] + 1, x_tick_interval[2]))
    else:
        ax.set_xlim(auto=True)

    if y_tick_interval:
        ax.set_ylim(y_tick_interval[0], y_tick_interval[1])
        ax.set_yticks(range(y_tick_interval[0], y_tick_interval[1] + 1, y_tick_interval[2]))
    else:
        ax.set_ylim(auto=True)

    # Set labels
    if x_label:  # Use the provided custom x-axis label if available
        ax.set_xlabel(x_label, fontsize=10)
    if y_label:  # Use the provided custom y-axis label if available
        ax.set_ylabel(y_label, fontsize=10)
    else:
        ax.set_ylabel('Count', fontsize=10)

    # Show the plot
    plt.show()

    # # Save the figure with transparency if needed
    # if transparent:
    #     fig.savefig('histogram.png', transparent=True)

def plot_box_plots(
    df, 
    columns_to_plot, 
    label_font_size=11, 
    y_tick_font_size=11, 
    y_tick_intervals=None, 
    y_labels=None, 
    plots_per_fig=12, 
    figsize=(20, 15),
    box_color='#e4e5e6',
    edge_color='black'
    ):
    """
    Creates vertical box plots to visualize the distribution of numeric columns in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the numeric columns to plot.
    - columns_to_plot (list of str): The list of columns to plot.
    - label_font_size (int): Font size for the y-axis labels. Default is 11.
    - y_tick_font_size (int): Font size for the y-axis tick labels. Default is 11.
    - y_tick_intervals (dict): A dictionary where keys are column names and values are tuples (start, end, step) for y-axis ticks.
                               If None, automatic tick intervals will be used.
    - y_labels (dict): A dictionary where keys are column names and values are custom labels for y-axis.
                       If None, column names will be used as labels.
    - plots_per_fig (int): Number of box plots per figure. Default is 12.
    - figsize (tuple): Size of each figure. Default is (20, 15).
    - box_color (str): Fill color for the box plots. Default is '#e4e5e6'.
    - edge_color (str): Edge color for the box plots. Default is 'black'.

    Returns:
    - None: The function creates and displays the box plots.
    """
    # Define default y_labels if not provided
    if y_labels is None:
        y_labels = {col: col for col in columns_to_plot}

    # Calculate the number of figures needed
    total_cols = len(columns_to_plot)
    num_figs = math.ceil(total_cols / plots_per_fig)

    for fig_num in range(num_figs):
        start_idx = fig_num * plots_per_fig
        end_idx = min(start_idx + plots_per_fig, total_cols)
        current_cols = columns_to_plot[start_idx:end_idx]
        num_plots = len(current_cols)
        
        # Determine grid size (e.g., 4x3 for 12 plots)
        cols = 4
        rows = math.ceil(num_plots / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()  # Flatten in case of multiple rows

        for i, col in enumerate(current_cols):
            ax = axes[i]
            sns.boxplot(
                y=df[col],
                ax=ax,
                palette=None,  # Explicitly set palette to None to prevent warnings
                boxprops=dict(edgecolor=edge_color, facecolor=box_color),  # Changed 'color' to 'edgecolor'
                linewidth=1,
                flierprops=dict(marker='.', color=edge_color, markersize=6),
                whiskerprops=dict(color=edge_color, linewidth=1),
                medianprops=dict(color=edge_color, linewidth=2),
                showcaps=False,
                width=0.2
            )

            # Customize spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_linewidth(1)
            ax.spines['bottom'].set_linewidth(1)

            # Set y-axis label
            ax.set_ylabel(y_labels[col], fontsize=label_font_size)

            # Set y-axis ticks if provided
            if y_tick_intervals and col in y_tick_intervals:
                start, end, step = y_tick_intervals[col]
                ax.set_ylim(start, end)
                ax.set_yticks(range(start, end + 1, step))

            # Set y-axis tick label font size
            ax.tick_params(axis='y', labelsize=y_tick_font_size)

            # Remove x-axis labels and titles
            ax.set_xticklabels([])
            ax.set_xlabel('')

        # Remove any unused subplots
        for j in range(num_plots, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

def plot_violin_plots(
    df, 
    columns_to_plot, 
    label_font_size=11, 
    y_tick_font_size=11, 
    y_tick_intervals=None, 
    y_labels=None, 
    plots_per_fig=12, 
    figsize=(20, 15),
    violin_color=colors[-1],  
    edge_color='black'         
    ):
    """
    Creates vertical violin plots to visualize the distribution of numeric columns in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the numeric columns to plot.
    - columns_to_plot (list of str): The list of columns to plot.
    - label_font_size (int): Font size for the y-axis labels. Default is 11.
    - y_tick_font_size (int): Font size for the y-axis tick labels. Default is 11.
    - y_tick_intervals (dict): A dictionary where keys are column names and values are tuples (start, end, step) for y-axis ticks.
                                If None, automatic tick intervals will be used.
    - y_labels (dict): A dictionary where keys are column names and values are custom labels for y-axis.
                       If None, column names will be used as labels.
    - plots_per_fig (int): Number of violin plots per figure. Default is 12.
    - figsize (tuple): Size of each figure. Default is (20, 15).
    - violin_color (str): Fill color for the violin plots. Default is 'lightblue'.
    - edge_color (str): Edge color for the violin plots. Default is 'black'.

    Returns:
    - None: The function creates and displays the violin plots.
    """
    # Define default y_labels if not provided
    if y_labels is None:
        y_labels = {col: col.replace('_', ' ').title() for col in columns_to_plot}

    # Calculate the number of figures needed
    total_cols = len(columns_to_plot)
    num_figs = math.ceil(total_cols / plots_per_fig)

    for fig_num in range(num_figs):
        start_idx = fig_num * plots_per_fig
        end_idx = min(start_idx + plots_per_fig, total_cols)
        current_cols = columns_to_plot[start_idx:end_idx]
        num_plots = len(current_cols)
        
        # Determine grid size (e.g., 4x3 for 12 plots)
        cols = 4
        rows = math.ceil(num_plots / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()  # Flatten in case of multiple rows

        for i, col in enumerate(current_cols):
            ax = axes[i]
            
            # Plot the violin plot
            sns.violinplot(
                y=df[col],
                ax=ax,
                inner=None,  # Removes the mini box plot inside the violin
                linewidth=1,  # Line thickness for the plot outline
                color=violin_color,
                edgecolor=edge_color  # Initial edge color
            )

            # Ensure edges of violins are black by iterating over PolyCollections
            for violin in ax.findobj(PolyCollection):
                violin.set_edgecolor(edge_color)
                violin.set_linewidth(1)
            
            # Customize spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_linewidth(1)
            ax.spines['bottom'].set_linewidth(1)

            # Set y-axis label
            ax.set_ylabel(y_labels[col], fontsize=label_font_size)

            # Set y-axis ticks if provided
            if y_tick_intervals and col in y_tick_intervals:
                start, end, step = y_tick_intervals[col]
                ax.set_ylim(start, end)
                ax.set_yticks(range(start, end + 1, step))

            # Set y-axis tick label font size
            ax.tick_params(axis='y', labelsize=y_tick_font_size)

            # Remove x-axis labels and titles
            ax.set_xticklabels([])
            ax.set_xlabel('')  # Removing the x-axis title

        # Remove any unused subplots
        for j in range(num_plots, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

def plot_histograms(
    df, 
    columns_to_plot, 
    x_tick_intervals=None, 
    y_tick_intervals=None, 
    x_labels=None, 
    y_labels=None, 
    plots_per_fig=6, 
    figsize=(20, 10),
    bin_settings=None,
    hist_color=colors[0],
    edge_color=None
    ):
    """
    Creates histograms to visualize the distribution of numeric columns in the DataFrame using Matplotlib.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the numeric columns to plot.
    - columns_to_plot (list of str): The list of columns to plot.
    - x_tick_intervals (dict): A dictionary where keys are column names and values are tuples (start, end, step) for x-axis ticks.
                                If None, automatic tick intervals will be used.
    - y_tick_intervals (dict): A dictionary where keys are column names and values are tuples (start, end, step) for y-axis ticks.
                                If None, automatic tick intervals will be used.
    - x_labels (dict): A dictionary where keys are column names and values are custom labels for x-axis.
                       If None, column names will be used as labels.
    - y_labels (dict): A dictionary where keys are column names and values are custom labels for y-axis.
                       If None, default to 'Count'.
    - plots_per_fig (int): Number of histograms per figure. Default is 6.
    - figsize (tuple): Size of each figure. Default is (20, 10).
    - bin_settings (dict): A dictionary where keys are column names and values are the number of bins or bin ranges.
                           If None, automatic binning will be used.
    - hist_color (str): Fill color for the histograms. Default is 'skyblue'.
    - edge_color (str): Edge color for the histograms. Default is 'black'.

    Returns:
    - None: The function creates and displays the histograms.
    """
    # Define a default color palette to mimic the original styling
    default_colors = ['skyblue', 'lightgreen', 'salmon', 'lightgrey', 'orange']

    # If colors list is referenced, ensure it exists
    # In this function, we will use default_colors to avoid undefined 'colors' variable

    # Define default y_labels if not provided
    if y_labels is None:
        y_labels = {col: 'Count' for col in columns_to_plot}
    
    # Define default x_labels if not provided
    if x_labels is None:
        x_labels = {col: col.replace('_', ' ').capitalize() for col in columns_to_plot}
    
    # Calculate the number of figures needed
    total_cols = len(columns_to_plot)
    num_figs = math.ceil(total_cols / plots_per_fig)
    
    # Define default bin settings if not provided
    if bin_settings is None:
        bin_settings = {col: 'auto' for col in columns_to_plot}  # 'auto' lets matplotlib decide
    
    for fig_num in range(num_figs):
        start_idx = fig_num * plots_per_fig
        end_idx = min(start_idx + plots_per_fig, total_cols)
        current_cols = columns_to_plot[start_idx:end_idx]
        num_plots = len(current_cols)
        
        # Determine grid size (e.g., 2x3 for 6 plots)
        cols = 3
        rows = math.ceil(num_plots / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()  # Flatten in case of multiple rows
        
        for i, col in enumerate(current_cols):
            ax = axes[i]
            
            # Check if the column exists in the DataFrame
            if col not in df.columns:
                print(f"Warning: Column '{col}' does not exist in the DataFrame. Skipping.")
                fig.delaxes(ax)
                continue
            
            # Determine number of bins or bin ranges
            bins = bin_settings[col]
            
            # Plot histogram
            ax.hist(
                df[col].dropna(),  # Drop NaN values to avoid errors
                bins=bins,
                color=hist_color,
                edgecolor=edge_color,
                alpha=1.0,
                rwidth=0.95,
                linewidth=1,
                zorder=3
            )
            
            # Customize spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_color('lightgrey')  # lightgrey
            ax.spines['bottom'].set_color('lightgrey')  # lightgrey
            ax.spines['left'].set_linewidth(1)
            ax.spines['bottom'].set_linewidth(1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Set tick parameters
            ax.tick_params(axis='x', which='both', bottom=True, top=False, 
                           direction='out', length=6, width=1, labelsize=11, color='lightgrey')
            ax.tick_params(axis='y', which='both', left=True, right=False, 
                           direction='out', length=6, width=1, labelsize=11, color='lightgrey')
            
            # Set grid lines for y-axis only
            ax.grid(axis='y', color='lightgrey', linestyle='-', linewidth=1, zorder=0)
            ax.set_axisbelow(True)  # Ensure grid is below the bars
            
            # Set x-axis label
            ax.set_xlabel(x_labels[col], fontsize=11)
            
            # Set y-axis label
            ax.set_ylabel(y_labels[col], fontsize=11)
            
            # Set x-axis ticks if provided
            if y_tick_intervals and col in y_tick_intervals:
                # Handle y_tick_intervals
                start, end, step = y_tick_intervals[col]
                ax.set_ylim(start, end)
                ax.set_yticks(range(start, end + 1, step))
            elif x_tick_intervals and col in x_tick_intervals:
                # Handle x_tick_intervals
                start, end, step = x_tick_intervals[col]
                ax.set_xlim(start, end)
                ax.set_xticks(range(start, end + 1, step))
            
            # Set tick label font sizes
            ax.tick_params(axis='x', labelsize=11)
            ax.tick_params(axis='y', labelsize=11)
        
        # Remove any unused subplots
        for j in range(num_plots, len(axes)):
            fig.delaxes(axes[j])
        
        # Adjust layout
        plt.tight_layout()
        plt.show()

def plot_correlation_matrix_heatmap(corr_mtrx, figsize=(14, 12), annot=True, linewidths=0.5):
    """
    Plots a non-redundant correlation matrix heatmap (lower left triangle only).
    
    Parameters:
    - corr_mtrx (pd.DataFrame): The correlation matrix to plot.
    - figsize (tuple): The size of the figure. Default is (14, 12).
    - annot (bool): Whether to annotate the heatmap with correlation values. Default is True.
    - linewidths (float): Width of the lines that will divide each cell in the heatmap. Default is 0.5.
    
    Returns:
    None: The function will display the heatmap.
    """

    # Define the custom color map for the heatmap
    colors_map = [colors[0], colors[-1], colors[1]]
    custom_diverging_palette = LinearSegmentedColormap.from_list("custom_palette", colors_map)
    
    # Create a mask for the upper triangle (we only want to show the lower left triangle)
    mask = np.triu(np.ones_like(corr_mtrx, dtype=bool))
    
    # Set up the figure and axis
    plt.figure(figsize=figsize)
    
    # Plot the heatmap with the mask
    ax = sns.heatmap(
        corr_mtrx,
        mask=mask,  # Apply the mask
        cmap=custom_diverging_palette,  # Custom color palette
        annot=annot,  # Annotate with correlation values
        linewidths=linewidths,  # Line width for cell borders
        cbar_kws={"shrink": .8}  # Shrink colorbar slightly
    )
    
    # Remove the 'site_latitude' label from the y-axis and 'day_of_week' from the x-axis
    x_labels = ax.get_xticklabels()  # Get current x-axis labels
    y_labels = ax.get_yticklabels()  # Get current y-axis labels

    # Identify the indices of 'day_of_week' and 'site_latitude'
    x_tick_pos = [i for i, label in enumerate(x_labels) if label.get_text() == 'day_of_week']
    y_tick_pos = [i for i, label in enumerate(y_labels) if label.get_text() == 'site_latitude']

    # Set new x-axis labels and remove the corresponding tick for 'day_of_week'
    ax.set_xticklabels(
        [label.get_text() if label.get_text() != 'day_of_week' else '' for label in x_labels],
        rotation=45,
        ha='right'
    )
    ax.set_yticklabels(
        [label.get_text() if label.get_text() != 'site_latitude' else '' for label in y_labels],
        rotation=0
    )

    # Remove the ticks for 'day_of_week' on the x-axis and 'site_latitude' on the y-axis
    if x_tick_pos:
        ax.set_xticks([tick for i, tick in enumerate(ax.get_xticks()) if i not in x_tick_pos])
    if y_tick_pos:
        ax.set_yticks([tick for i, tick in enumerate(ax.get_yticks()) if i not in y_tick_pos])
    
    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_scatter_plots(df, columns_to_plot, target_variable='pm2_5', figsize=(20, 15), plots_per_fig=4):
    """
    Function to create customized scatter plots with Seaborn and Matplotlib for all features in 'columns_to_plot' against the target variable.
    
    Parameters:
    - df : DataFrame
        The DataFrame containing the data to be plotted.
    - columns_to_plot : list
        List of column names to plot against the target variable.
    - target_variable : str
        The name of the target variable to plot against. Default is 'pm2_5'.
    - figsize : tuple
        Size of the figure (width, height). Default is (20, 15).
    - plots_per_fig : int
        Number of scatter plots per figure. Default is 4.
    
    Returns:
    - None: The function will display the scatter plots.
    """
    
    # Validate that the target variable exists in the DataFrame
    if target_variable not in df.columns:
        raise ValueError(f"The target variable '{target_variable}' is not present in the DataFrame.")
    
    # Remove the target variable from columns_to_plot to prevent plotting it against itself
    columns_to_plot = [col for col in columns_to_plot if col != target_variable]
    
    # Calculate the number of figures needed
    total_plots = len(columns_to_plot)
    num_figs = math.ceil(total_plots / plots_per_fig)
    
    # Iterate over each figure
    for fig_num in range(num_figs):
        # Determine the start and end indices for the current figure
        start_idx = fig_num * plots_per_fig
        end_idx = min(start_idx + plots_per_fig, total_plots)
        
        # Get the subset of columns for the current figure
        current_cols = columns_to_plot[start_idx:end_idx]
        num_plots = len(current_cols)
        
        # Determine the layout (rows and columns)
        cols = 2 if num_plots > 2 else num_plots  # Adjust columns based on number of plots
        rows = math.ceil(num_plots / cols)
        
        # Create subplots
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        # If there's only one plot, axes might not be an array
        if num_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()  # Flatten in case of multiple rows
        
        for i, col in enumerate(current_cols):
            ax = axes[i]
            
            # Check if the column exists in the DataFrame
            if col not in df.columns:
                print(f"Warning: Column '{col}' does not exist in the DataFrame. Skipping.")
                fig.delaxes(ax)
                continue
            
            # Plot scatter plot
            sns.scatterplot(
                x=df[col], 
                y=df[target_variable], 
                ax=ax, 
                color=colors[0], 
                zorder=3, 
                alpha=0.7
            )
            
            # Set labels
            ax.set_xlabel(col.replace('_', ' ').title(), color='black', fontsize=12)
            ax.set_ylabel(target_variable.replace('_', ' ').title(), color='black', fontsize=12)
            
            # Remove the title from each subplot
            ax.set_title('')
            
            # Customize the axes and gridlines
            # Remove the top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Set color and thickness for the bottom and left spines
            ax.spines['bottom'].set_color(colors[-1])   # Change to desired color
            ax.spines['left'].set_color(colors[-1])     # Change to desired color
            ax.spines['bottom'].set_linewidth(1)        # Change to desired thickness
            ax.spines['left'].set_linewidth(1)          # Change to desired thickness
            
            # Set tick parameters: color light grey for ticks, black for tick labels
            ax.tick_params(color=colors[-1], labelcolor='black', width=1, length=8)
            
            # Add gridlines (x and y) in light grey, with linewidth of 1, behind the datapoints
            ax.grid(True, which='both', axis='both', color=colors[-1], linewidth=1, zorder=1)
        
        # Remove any unused subplots
        for j in range(num_plots, len(axes)):
            fig.delaxes(axes[j])
        
        # Adjust the layout to make sure labels and spacing don't overlap
        plt.tight_layout()
        
        # Show the plot
        plt.show()

def error_analysis_plot(y_test, y_pred_test, window_size=30):
    """Generates true vs. predicted values and residual scatter plot for models."""
    residuals = y_test - y_pred_test

    # Create the figure with two subplots
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    plt.subplots_adjust(right=1)

    # True vs. Predicted Values scatter plot
    sns.scatterplot(x=y_pred_test, y=y_test, ax=ax[0], color=colors[0], alpha=0.7, zorder=3)
    ax[0].plot([-400, 350], [-400, 350], color="grey", linestyle='-', zorder=2)
    ax[0].set_xlabel("Predicted Values", fontsize=11)
    ax[0].set_ylabel("True Values", fontsize=11)
    ax[0].set_xlim((y_pred_test.min() - 10), (y_pred_test.max() + 10))
    ax[0].set_ylim((y_test.min() - 40), (y_test.max() + 40))

    # Residuals scatter plot
    sns.scatterplot(x=y_pred_test, y=residuals, ax=ax[1], color=colors[0], alpha=0.7, zorder=3)
    ax[1].plot([-400, 350], [0, 0], color="grey", linestyle='-', zorder=2)
    ax[1].set_xlabel("Predicted Values", fontsize=11)
    ax[1].set_ylabel("Residuals", fontsize=11)
    ax[1].set_xlim((y_pred_test.min() - 10), (y_pred_test.max() + 10))
    ax[1].set_ylim((residuals.min() - 10), (residuals.max() + 10))

    # Customizing the appearance for both plots
    for a in ax:
        # Remove top and right spines
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        
        # Customize bottom and left spines
        a.spines['bottom'].set_color("#193251")
        a.spines['left'].set_color("#193251")
        a.spines['bottom'].set_linewidth(1)
        a.spines['left'].set_linewidth(1)
        
        # Customize tick parameters
        a.tick_params(color="#193251", labelcolor="black", width=1, length=6)
        
        # Add gridlines (behind scatter points)
        a.grid(True, which='both', axis='both', color="#E5E5E5", linewidth=1, zorder=0)

    # Adjust the layout
    plt.tight_layout()
    plt.show()

def plot_cat_bar_plot(
    df_metrics,
    metric='RMSE_test',
    winner_label='Zindi Winning Solution',
    winner_value=26.0997,
    model_type_col='Model Type',
    search_type_col='Search Type',
    x_axis_title='Root Mean Squared Error (RMSE)',
    figsize=(12, 10),
    color_winner='orange',
    color_others='skyblue',
    grid_alpha=0.7,
    grid_linestyle='-',
    fontsize=10,
    x_limit_padding=2,
    title='',
    show=True
    ):
    """
    Plots a horizontal bar chart of the specified metric for different models,
    highlighting the Zindi winning solution.

    Parameters:
    -----------
    df_metrics : pd.DataFrame
        The DataFrame containing model metrics. Must include columns for model type, search type, and the metric.
        
    metric : str, optional
        The name of the metric column to plot (default is 'RMSE_test').
        
    winner_label : str, optional
        The label for the winning solution to be highlighted (default is 'Zindi Winning Solution').
        
    winner_value : float, optional
        The metric value for the winning solution (default is 26.0997).
        
    model_type_col : str, optional
        The column name for model types in df_metrics (default is 'Model Type').
        
    search_type_col : str, optional
        The column name for search types in df_metrics (default is 'Search Type').
        
    x_axis_title : str, optional
        The label for the x-axis (default is 'Root Mean Squared Error (RMSE)').
        
    figsize : tuple, optional
        The size of the figure in inches (default is (12, 10)).
        
    color_winner : str, optional
        The color for the winning solution bar (default is 'orange').
        
    color_others : str, optional
        The color for the other model bars (default is 'skyblue').
        
    grid_alpha : float, optional
        The transparency level for grid lines (default is 0.7).
        
    grid_linestyle : str, optional
        The style of the grid lines (default is '-').
        
    fontsize : int, optional
        The font size for tick labels (default is 10).
        
    x_limit_padding : float, optional
        The padding added to the x-axis limit (default is 2).
        
    title : str, optional
        The title of the plot (default is empty).
        
    show : bool, optional
        Whether to display the plot immediately (default is True).

    Returns:
    --------
    plt.Figure
        The matplotlib figure object containing the plot.
    """
    
    # Combine 'Model Type' and 'Search Type' for clearer labels
    df_metrics = df_metrics.copy()  # To avoid SettingWithCopyWarning
    df_metrics['Model_Search'] = df_metrics[model_type_col] + ' (' + df_metrics[search_type_col] + ')'
    
    # Create a DataFrame for the Zindi winning solution
    zindi_winner = pd.DataFrame({
        'Model_Search': [winner_label],
        metric: [winner_value]
    })
    
    # Append the Zindi winner to the main DataFrame
    df_plot = pd.concat([df_metrics[['Model_Search', metric]], zindi_winner], ignore_index=True)
    
    # Sort the DataFrame based on the specified metric (ascending order: best to worst)
    df_sorted = df_plot.sort_values(by=metric, ascending=True).reset_index(drop=True)
    
    # Assign colors to the Zindi winning solution and to all other models
    colors = [
        color_winner if model.strip().lower() == winner_label.strip().lower() else color_others 
        for model in df_sorted['Model_Search']
    ]
    
    # Flip the Order of the Bars to have the best-performing models at the top
    df_flipped = df_sorted[::-1].reset_index(drop=True)
    colors_flipped = colors[::-1]
    
    # Plotting with Matplotlib
    plt.figure(figsize=figsize)
    
    # Create the horizontal bar plot
    bars = plt.barh(
        df_flipped['Model_Search'], 
        df_flipped[metric], 
        color=colors_flipped, 
        edgecolor='none', 
        zorder=2
    )
    
    # Add solid gridlines for the x-axis behind the bars
    plt.grid(axis='x', linestyle=grid_linestyle, alpha=grid_alpha, zorder=0)
    
    # Customize Axes and Spines
    plt.title(title)
    plt.ylabel('')
    plt.xlabel(x_axis_title, fontsize=12)
    
    # Get the current Axes instance
    ax = plt.gca()
    
    # Remove the left, bottom, top, and right spines (axis lines)
    for spine in ['left', 'bottom', 'top', 'right']:
        ax.spines[spine].set_visible(False)
    
    # Remove the tick marks but keep the tick labels
    ax.tick_params(axis='both', which='both', length=0)
    
    # Keep the axis tick labels with specified font sizes
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    # Adjust Plot Limits and Layout
    n_bars = len(df_flipped)
    ax.set_ylim(-0.5, 22.5)
    plt.xlim(left=0, right=df_sorted[metric].max() + x_limit_padding)
    
    # Ensure the layout is tight so labels fit within the figure area
    plt.tight_layout()
    
    # Display the Plot if required
    if show:
        plt.show()

    return plt.gcf()  # Return the current figure
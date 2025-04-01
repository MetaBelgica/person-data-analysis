import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import display, Markdown
import json

UNCERTAIN_CHARACTERS = ['[', '?', '~', '%', 'X', '{']

def isCertainEDTFDate(value):
    if pd.isna(value):
        return np.nan
    if value != '':
        if any(char in str(value) for char in UNCERTAIN_CHARACTERS):
            return False
        else:
            return True
    else:
        return False
    
def getDatePrecision(value):
    if pd.isna(value) or value == '':
        return 'no_date'
    elif any(char in str(value) for char in UNCERTAIN_CHARACTERS):
        return 'uncertain'
    elif len(value) == 7 and '-' in value:
        return 'month'
    elif len(value) == 10:
        return 'day'
    elif len(value) <= 4:
        return 'year'
    else:
        return 'unknown'

# -----------------------------------------------------------------------------
def compute_totals(df, config):
    """
    Compute totals and percentages for each feature and data source.
    
    Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        config (dict): JSON configuration defining the features and data sources.
    
    Returns:
        pd.DataFrame: DataFrame with totals and percentages.
    """
    rows = []
    for feature in config["features"]:
        row = {"Feature": feature["label"]}
        column = feature["column"]
        
        dataSourceColumnName = config["dataSources"]["dataSourcesColumnName"]

        for source in config["dataSources"]["dataSourcesNames"]:
            source_df = df[df[dataSourceColumnName] == source]
            total_records = source_df.shape[0]
            count = source_df[column].notna().sum() if feature["percentage"] else total_records
            
            # Calculate percentage if required
            if feature["percentage"]:
                percentage = (count / total_records * 100) if total_records > 0 else 0
                row[source] = {"count": count, "percentage": percentage, "total": total_records}
            else:
                row[source] = {"count": count}
        
        rows.append(row)
    
    # Convert rows into a DataFrame
    total_df = pd.DataFrame(rows)
    return total_df

# -----------------------------------------------------------------------------
def render_markdown_table(current_df, previous_df=None):
    """
    Render the markdown table and compare with a previous version (optional).
    
    Parameters:
        current_df (pd.DataFrame): The current totals DataFrame.
        previous_df (pd.DataFrame): (Optional) A previous version of the totals DataFrame for comparison.
    """
    # Collect table rows
    table_rows = []
    headers = ["Feature"] + list(current_df.columns[1:])
    table_rows.append("| " + " | ".join(headers) + " |")
    table_rows.append("|" + " --- |" * len(headers))
    
    for _, row in current_df.iterrows():
        row_cells = [row["Feature"]]
        
        for source in row.index[1:]:
            current_value = row[source]
            
            # Extract current count and percentage
            current_count = current_value["count"]
            current_percentage = current_value.get("percentage", None)
            
            # Default display for count and percentage
            display_value = f"{current_count:,}"
            if current_percentage is not None:
                display_value += f" (<strong>{current_percentage:.0f}%</strong>)"
            
            # Compare with previous version if available
            if previous_df is not None and source in previous_df.columns:
                previous_row = previous_df[previous_df["Feature"] == row["Feature"]]
                if not previous_row.empty:
                    previous_value = previous_row[source].values[0]
                    previous_count = previous_value["count"]
                    previous_total = previous_value.get("total", 0)
                    
                    # Only calculate percentage difference if required
                    if "percentage" in current_value:
                        previous_percentage = (previous_count / previous_total * 100) if previous_total > 0 else 0
                        difference = current_percentage - previous_percentage
                        
                        #print(f'previous_total: {previous_total}, previous_percentage: {previous_percentage}, current_count {current_value.get("total", 0)} current_percentage: {current_percentage} difference is {difference}')
                        #print()
                        # Color coding for differences
                        if difference > 0.1:
                            diff_text = f" <span style='color:green;'>+{difference:.0f}%</span>"
                        elif abs(difference) < 0:
                            diff_text = f" <span style='color:red;'>{difference:.0f}%</span>"
                        else:
                            diff_text = ""
                        
                        # Combine current value with difference
                        display_value = (
    f"{current_count:,} (<strong>{current_percentage:.0f}%</strong>{diff_text})"
)
                        #display_value = f"{current_count:,} ({current_percentage:.0f}%)**{diff_text}**"
            
            row_cells.append(display_value)
        
        table_rows.append("| " + " | ".join(row_cells) + " |")
    
    # Combine rows into a markdown string
    markdown_table = "\n".join(table_rows)
    
    # Display the markdown table in the Jupyter notebook
    display(Markdown(markdown_table))

# -----------------------------------------------------------------------------
def applyEDTFStandardization(df, type_column, type_list, date_column, new_columns, func, dateMapping, monthMapping):

    # Create the mask to filter the rows
    mask = df[type_column].isin(type_list)

    # Apply the function where the mask is True
    df[new_columns] = df[mask].apply(
        lambda row: func(row.name, str(row[date_column]), dateMapping, monthMapping),
        axis=1, 
        result_type='expand'
    )

    # Set NaN for rows where the mask is False
    df.loc[~mask, new_columns] = np.nan

    return df

# -----------------------------------------------------------------------------
# mark rows where we have a 100$d date but no 046 date based on the certainty of the 100$d date
# e.g. "fl. 1610" or "16.." are uncertain dates, whereas "1620" is certain
def markDates100dType(df, dateColumn, valueColumn, otherValueColumn, problemColumn, typeColumn):
    uncertainDateMask = df[valueColumn].str.contains(r'[a-zA-Z.-]|[\?]|[\[]', regex=True, na=False)
    # 7,380 certain
    # 
    # if 046 date is already set, the 100$d date is irrelevant
    df.loc[
        (df[dateColumn].notna()), # there is a 046 date
        typeColumn
    ] = 'irrelevant'
    
    # sure no date, because also the other component does not have a date
    # no 046 date, no date in value column and no date in other value column
    # thus e.g. no birth date and no death date
    df.loc[
        (df[dateColumn].isna()) # no 046 date
        & (df[valueColumn].isna()) # no 100$d date component (e.g. birth)
        & (df[otherValueColumn].isna()), # no 100$d other component (e.g. death)
        typeColumn
    ] = 'noDate'
    
    # probably no date, because no split problem
    # no 046 date, no date in value column and no date in other value column
    # thus e.g. no birth date and no death date
    df.loc[
        (df[dateColumn].isna()) # no 046 date
        & (df[valueColumn].isna()) # no 100$d date component (e.g. birth)
        & (df[problemColumn].isna()), # no obvious problem while splitting 100$d
        typeColumn
    ] = 'noDate'
    
    # no 046 date, but a problem
    df.loc[
        (df[dateColumn].isna()) # no 046 date
        & (df[problemColumn].notna()),
        typeColumn
    ] = 'splitProblem'

    # no 046 date, but a problem
    # not specifically a problem while splitting, but something is wrong with the date
    df.loc[
        (df[dateColumn].isna()) # no 046 date
        & (df[valueColumn].notna())
        & (df[valueColumn].str.len() < 4),
        typeColumn
    ] = 'splitProblem'
 
    
    # no 046 date, but we have a 100$d date which is uncertain
    df.loc[
        (df[dateColumn].isna()) # no 046 date
        & (df[valueColumn].notna()) # a 100$d date component
        & (df[problemColumn].isna()) # no obvious problem while splitting 100$d
        & (uncertainDateMask), # uncertain date
        typeColumn
    ] = 'uncertain'
    
    # no 046 date, but we have a 100$d date which is certain
    df.loc[
        (df[dateColumn].isna()) # no 046 date
        & (df[valueColumn].notna()) # a 100$d date component
        & (df[valueColumn].str.len() >=4) # a reasonable certain date value
        & (df[problemColumn].isna()) # no obvious problem while splitting 100$d
        & (~uncertainDateMask), # certain date
        typeColumn
    ] = 'certain'

# -----------------------------------------------------------------------------
def extractCompleteName(value):
    lastNames = set()
    firstNames = set()
    if isinstance(value, list):
        for v in value:
            lastNames.update([n for n in v['lastName'].split(';')])
            firstNames.update([n for n in v['firstName'].split(';')])
        return ';'.join(lastNames) + ', ' + ';'.join(firstNames)
    else:
        return np.nan

# -----------------------------------------------------------------------------
def replacePlaceholderValues(df, column, regexList):
    for regex in regexList:
        df[column] = df[column].str.strip().replace(regex, np.nan, regex=True)   



# -----------------------------------------------------------------------------
def getTop10Place(df, variable):
    # Filter for birthPlace
    relevantData = df[df['variableType'] == variable]

    # Group by dataSource and placename, and calculate counts
    top_places = (relevantData.groupby(['dataSource', 'placename'])
                       .size()
                       .reset_index(name='count')
                       .sort_values(['dataSource', 'count'], ascending=[True, False]))

    # Get the top 10 placenames per dataSource
    top_10_places = top_places.groupby('dataSource').head(10)
    return top_10_places

# -----------------------------------------------------------------------------
def plotRadarChart(statsDataframes, variables):

  # Prepare data for radar chart (total counts, not percentages)
  data = {}

  for source in ['KBR', 'KIK-IRPA', 'KMSKB', 'KMKG']:
      # Gather the total count for each variable in each data source by summing up the values for the variable column
      data[source] = [df.loc[source, var] for var, df in zip(variables, statsDataframes)]

  # Calculate the global maximum value across all variables (not per variable)
  max_value_all_sources = max(max(df.loc[source, var] for source in ['KBR', 'KIK-IRPA', 'KMSKB', 'KMKG']) 
                              for var, df in zip(variables, statsDataframes))

  # Close the circle for radar chart (repeat the first value)
  categories = variables + [variables[0]]
  angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=True)

  # Create radar chart
  fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

  colors = ['blue', 'orange', 'green', 'red']
  for i, (source, values) in enumerate(data.items()):
      values = values + [values[0]]  # Close the circle
      ax.fill(angles, values, color=colors[i], alpha=0.25, label=source)
      ax.plot(angles, values, color=colors[i], linewidth=2)

  # Add labels and legend for the radar chart
  ax.set_xticks(angles[:-1])  # Exclude the last angle since it's just a repeat of the first
  ax.set_xticklabels(variables, fontsize=12)

  # Adjust y-axis dynamically (showing total counts)
  ax.set_yscale('log')  # Set logarithmic scale for the y-axis
  ax.set_yticks([10, 100, 1000, 10000, 100000])  # Logarithmic ticks
  ax.set_yticklabels([str(i) for i in [10, 100, 1000, 10000, 100000]], color="gray", size=10)

  # Set y-axis limit based on the global max value
  ax.set_ylim(1, max_value_all_sources)  # Set a lower limit to avoid log(0)

  ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=10)

  # Show the plot
  plt.tight_layout()

  fig.savefig('metadata-radar-chart-total-logarithmic.pdf', format='pdf', dpi=300)
  fig.savefig('metadata-radar-chart-total-logarithmic.png', format='png', dpi=300)
  plt.show()

# -----------------------------------------------------------------------------
def plotRadarChartPercentage(statsDataframes, variables):

  # Prepare data for radar chart
  data = {}
  for source in ['KBR', 'KIK-IRPA', 'KMSKB', 'KMKG']:
      data[source] = [df.loc[source, 'percentage'] for df in statsDataframes]

  # Close the circle for radar chart
  categories = variables + [variables[0]]
  angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=True)

  # Create radar chart
  fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

  colors = ['blue', 'orange', 'green', 'red']
  for i, (source, values) in enumerate(data.items()):
      values = values + [values[0]]  # Close the circle
      ax.fill(angles, values, color=colors[i], alpha=0.25, label=source)
      ax.plot(angles, values, color=colors[i], linewidth=2)

  # Add labels and legend for the radar chart
  ax.set_xticks(angles[:-1])
  ax.set_xticklabels(variables, fontsize=12)
  ax.set_yticks([20, 40, 60, 80, 100])
  ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], color="gray", size=10)
  ax.set_ylim(0, 100)  # Set y-axis limits from 0 to 100
  ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=10)


  # Show the plot
  plt.tight_layout()

  fig.savefig('metadata-radar-chart-percentage.pdf', format='pdf', dpi=300)
  fig.savefig('metadata-radar-chart-percentage.png', format='png', dpi=300)
  plt.show()


# -----------------------------------------------------------------------------
def plotBarChart(df, x_column, group_column=None, title="Bar Chart", log_scale=False, save_path_pdf=None, save_path_png=None):
    """
    Plots a bar chart with optional grouping and logarithmic y-axis scaling.

    Parameters:
        df (pd.DataFrame): The input dataframe containing the data.
        x_column (str): The column to be used for the x-axis (categorical variable).
        group_column (str, optional): The column used for grouping bars (categorical variable).
        title (str, optional): The title of the chart. Default is "Bar Chart".
        log_scale (bool, optional): Whether to use a logarithmic scale for the y-axis. Default is False.
        save_path_pdf (str, optional): Path to save the figure as a PDF.
        save_path_png (str, optional): Path to save the figure as a PNG.

    Returns:
        None
    """
    plt.figure(figsize=(12, 6))

    if group_column:
        # Grouped bar chart
        grouped_counts = df.groupby([x_column, group_column]).size().reset_index(name='count')

        # Compute total counts per x_column to sort from highest to lowest
        total_counts = grouped_counts.groupby(x_column)['count'].sum().reset_index()
        total_counts = total_counts.sort_values(by='count', ascending=False)

        # Reorder the original data
        grouped_counts[x_column] = pd.Categorical(grouped_counts[x_column], categories=total_counts[x_column], ordered=True)

        ax = sns.barplot(data=grouped_counts, x=x_column, y='count', hue=group_column, palette='Set2')
    else:
        # Regular bar chart
        counts = df[x_column].value_counts(dropna=False).reset_index()
        counts.columns = [x_column, 'count']
        counts = counts.sort_values(by='count', ascending=False)  # Sort by count

        ax = sns.barplot(data=counts, x=x_column, y='count')

    if log_scale:
      ax.set_yscale('log')

    # Add labels on top of bars (ensure small values are visible)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            x = p.get_x() + p.get_width() / 2
            text = f'{int(height)}'
            # Adjust label position depending on whether log scale is used
            y_offset = max(2, height * 0.02) if not log_scale else height * 0.2
            ax.annotate(text, (x, height + y_offset), ha='center', va='bottom', fontsize=14)

    # Formatting
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(x_column, fontsize=14)
    ax.set_ylabel('Count (log scale)' if log_scale else 'Count', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)

    # Adjust y-axis limits
    max_count = max(grouped_counts["count"]) if group_column else max(counts["count"])
    plt.ylim(1 if log_scale else 0, max(ax.get_ylim()[1], 1.1 * max_count))

    if group_column:
        plt.legend(title=group_column, fontsize=18)

    # Save plot if paths are provided
    fig = ax.get_figure()
    if save_path_pdf:
        fig.savefig(save_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
    if save_path_png:
        fig.savefig(save_path_png, format='png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()


# -----------------------------------------------------------------------------
def plotDecade(decade_counts, title_prefix, min_decade, max_decade=2100):

  plt.figure(figsize=(14, 8))
  beginCentury = str(min_decade)[:2]
  endCentury = str(max_decade)[:2]

  if max_decade < 2100:
    filtered_decade_counts = decade_counts[(decade_counts['decade'] >= min_decade) & (decade_counts['decade'] <= max_decade)]
    plt.title(f'Number of {title_prefix} Dates Across Decades {int(beginCentury)+1}th century to {int(endCentury)+1}th century by DataSource', fontsize=20)
  else:
    filtered_decade_counts = decade_counts[(decade_counts['decade'] >= min_decade)]
    plt.title(f'Number of {title_prefix} Dates Across Decades {int(beginCentury)+1}th century Onward) by DataSource', fontsize=20)

  # Bar plot with filtered data
  ax = sns.barplot(data=filtered_decade_counts, x='decade', y='count', hue='dataSource', palette='Set2')

  # Customize the plot
  plt.xlabel('Decade', fontsize=18)
  plt.ylabel('Count of Birth Dates', fontsize=18)
  plt.yticks(fontsize=18)
  plt.xticks(rotation=45, fontsize=18)
  plt.legend(title='Data Source', fontsize=18)

  for container in ax.containers:
    ax.bar_label(container, fmt='%d', label_type="edge", padding=3, fontsize=10, rotation=90)

  plt.tight_layout()

  plt.savefig(f'{title_prefix}-dates-decades-from-{min_decade}-to-{max_decade}.pdf', format='pdf', dpi=300, bbox_inches='tight')
  plt.savefig(f'{title_prefix}-dates-decades-from-{min_decade}-to-{max_decade}.png', format='png', dpi=300, bbox_inches='tight')

  plt.show()


# -----------------------------------------------------------------------------
def plotDatePatternsPercentages(percentage_data, title_prefix, is_percentage=False, save_path_pdf=None, save_path_png=None):
    # Define colors for each bar chart
    colors = ['blue', 'orange', 'green', 'red']
    
    # Create subplots with increased figure size
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)  # Increased figsize for better spacing
    axes = axes.flatten()  # Flatten to easily iterate through subplots
    
    # Plot each dataSource in a separate subplot
    for i, dataSource in enumerate(percentage_data.columns):
        ax = axes[i]
        bars = ax.bar(percentage_data.index, percentage_data[dataSource], color=colors[i], label=dataSource)
        ax.set_title(f'{title_prefix}: DataSource {dataSource}', fontsize=32, pad=20)
        ax.set_xlabel('Rule', fontsize=18)
        ax.set_ylabel('Percentage (%)' if is_percentage else 'Number', fontsize=18)
        ax.set_xticks(range(len(percentage_data.index)))
        ax.set_xticklabels(percentage_data.index, fontsize=18, rotation=45, ha='right')  # Slightly smaller labels
        ax.set_yscale('log')  # Logarithmic scale for y-axis
        ax.legend(fontsize=12)
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only add label if height > 0
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.1,  # Offset slightly above the bar
                    f'{height:.2f}%' if is_percentage else f'{height}',
                    ha='center',
                    va='bottom',
                    fontsize=14  # Slightly smaller font for data labels
                )
    
    # Remove empty subplot (if fewer than 4 dataSources)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    # Adjust layout and increase vertical space between rows
    fig.subplots_adjust(hspace=1.2, wspace=0.3)  # Increased hspace for more vertical space
    
    # Save the figure if paths are provided
    if save_path_pdf:
        fig.savefig(save_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
    
    if save_path_png:
        fig.savefig(save_path_png, format='png', dpi=300, bbox_inches='tight')

# -----------------------------------------------------------------------------
def plotDatePatternsPercentagesBAK(percentage_data, title_prefix, is_percentage=False, save_path_pdf=None, save_path_png=None):

  # Define colors for each bar chart
  colors = ['blue', 'orange', 'green', 'red']

  # Create subplots
  fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)  # 2x2 grid of subplots
  axes = axes.flatten()  # Flatten to easily iterate through subplots

  # Plot each dataSource in a separate subplot
  for i, dataSource in enumerate(percentage_data.columns):
      ax = axes[i]
      bars = ax.bar(percentage_data.index, percentage_data[dataSource], color=colors[i], label=dataSource)
      ax.set_title(f'{title_prefix}: DataSource {dataSource}', fontsize=14)
      ax.set_xlabel('Rule', fontsize=18)
      ax.set_ylabel('Percentage (%)' if is_percentage else 'Number', fontsize=18)
      ax.set_xticks(range(len(percentage_data.index)))
      ax.set_xticklabels(percentage_data.index, fontsize=18, rotation=45, ha='right')
      ax.set_yscale('log')  # Logarithmic scale for y-axis
      ax.legend(fontsize=18)
      
      # Add data labels
      for bar in bars:
          height = bar.get_height()
          ax.text(
              bar.get_x() + bar.get_width() / 2,
              height + 0.1,  # Offset slightly above the bar
              f'{height:.2f}%' if is_percentage else f'{height}',
              ha='center',
              va='bottom',
              fontsize=14
          )

  # Remove empty subplot (if fewer than 4 dataSources)
  for j in range(i + 1, len(axes)):
      fig.delaxes(axes[j])

  # Adjust layout
  plt.tight_layout()

  if save_path_pdf:
    fig.savefig(save_path_pdf, format='pdf', dpi=300)

  if save_path_png:
    fig.savefig(save_path_png, format='png', dpi=300)


# -----------------------------------------------------------------------------
if __name__ == "__main__":
  import doctest
  doctest.testmod()

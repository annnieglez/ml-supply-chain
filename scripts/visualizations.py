'''In this script, we will create visualizations for the data analysis (eda.ipynb)'''

# Standard Libraries
import os

# Data Handling & Computation
import pandas as pd
import numpy as np

# Visualization Libraries
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

# ==============================
# Directory Setup
# ==============================

# Define the directory name for saving images
OUTPUT_DIR = "../images"

# Check if the directory exists, if not, create it
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==============================
# Plot Styling & Customization
# ==============================

# Set a Minimalist Style
sns.set_style("whitegrid")

# Customize Matplotlib settings for a modern look
mpl.rcParams.update({
    'axes.edgecolor': 'grey',       
    'axes.labelcolor': 'black',     
    'xtick.color': 'black',         
    'ytick.color': 'black',         
    'text.color': 'black'           
})

# General color palette for plots
custom_colors = ["#8F2C78", "#1F4E79"]

# Colors for late and not late orders
non_risk_color = "#8F2C78" # Purple
risk_color = "#1F4E79" # Blue

# Define a custom colormap from light to dark shades of purple
custom_cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "custom_purple", ["#F5A7C4", "#8F2C78", "#5C0E2F"]
)

# ==============================
# Font Configuration
# ==============================

# Path to the custom font file
FONT_PATH = '../scripts/fonts/Montserrat-Regular.ttf'

# Add the font to matplotlib's font manager
font_manager.fontManager.addfont(FONT_PATH)

# Set the font family to Montserrat
plt.rcParams['font.family'] = 'Montserrat'

# ==============================
# Custom Axis Formatter Function
# ==============================

def currency_formatter(x, pos):
    '''
    Custom formatter function to display axis values,
    formatted as currency with comma separators.

    Parameters:
        - x (float): The numerical value to format.
        - pos (int): The tick position (required for matplotlib formatters).

    Returns:
        - str: Formatted string representation of the value.
    '''
    return f'${x:,.2f}'

def remove_axes():
    '''
    Removes the axes from the current plot by hiding the spines.

    Usage:
        Call this function after plotting your data to remove the axes from the figure.
    '''

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

# ==============================
# Visualization Functions
# ==============================

def plot_histograms(dataframe):
    '''
    Create histograms for all numerical columns in the dataframe. Returns a grid of subplots.
    '''

    # Create a copy of the dataframe
    df = dataframe.copy()
    df = df.loc[:, ~df.columns.str.contains('id', case=False)]
    df = df.loc[:, ~df.columns.str.contains('date', case=False)]
    
    # Get the number of columns
    num_columns = len(df.columns)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=9, ncols=3, figsize=(14, 20))
    axes = axes.flatten()
    if num_columns == 1:
        axes = [axes]  

    # Cumstoms bins for each column from the categorical.csv file
    bins = [6, 4, 50, 50, 2, 20, 50, 50, 50, 10, 50, 5, 50, 50, 100, 6, 4, 24, 31, 12, 7, 4, 24, 31, 12, 7]

    # Loop through each column and plot the histogram
    for i, column in enumerate(df.columns):
        sns.histplot(df[column], bins=bins[i],  kde=False, color='#6EE5D9', alpha=1.0, ax=axes[i])
        axes[i].set_xlabel(f"{column.replace('_', ' ').title()}", fontsize=12)
        axes[i].set_ylabel('Frequency', fontsize=12)
        axes[i].grid(axis='y', linestyle='--', alpha=0.3)
        axes[i].grid(axis='x', visible=False)

    # Hide any unused axes   
    for j in range(num_columns, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Histograms of Numerical Features", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    plt.savefig(os.path.join(OUTPUT_DIR, "Histograms_of_Numerical_Features.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True) 
    
def plot_histograms_delivery_risk_split(dataframe):
    '''
    Create histograms for all numerical columns in the dataframe, with late_delivery_risk as hue.
     Returns a grid of subplots.
    '''

    # Create a copy of the dataframe
    df = dataframe.copy()
    df = df.loc[:, ~df.columns.str.contains('id', case=False)]
    df = df.loc[:, ~df.columns.str.contains('date', case=False)]

    late_delivery_risk = df['late_delivery_risk']
    df = df.drop(columns=['late_delivery_risk']) 
    
    # Get the number of columns
    num_columns = len(df.columns)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=9, ncols=3, figsize=(14, 20))
    axes = axes.flatten()
    if num_columns == 1:
        axes = [axes]  

    bins = [6, 4, 50, 50, 20, 50, 50, 50, 10, 50, 5, 50, 50, 100, 6, 4, 24, 31, 12, 7, 4, 24, 31, 12, 7]

    # Loop through each column and plot the histogram
    for i, column in enumerate(df.columns):
        sns.histplot(df, x=column, hue=late_delivery_risk, bins=bins[i],
                         kde=False, palette=custom_colors, alpha=0.7, ax=axes[i], legend=False)
        #axes[i].set_title(f"Histogram of {column.replace('_', ' ').title()}", fontsize=14)
        axes[i].set_xlabel(f"{column.replace('_', ' ').title()}", fontsize=12)
        axes[i].set_ylabel('Frequency', fontsize=12)
        axes[i].grid(axis='y', linestyle='--', alpha=0.3)
        axes[i].grid(axis='x', visible=False)
        #axes[i].legend(title="Late Delivery Risk", labels=["Risk", "No Risk"])

    # Hide any unused axes   
    for j in range(num_columns, len(axes)):
        axes[j].axis('off')

    handles, labels = axes[0].get_legend_handles_labels()  # Get the legend from the first plot
    fig.legend(handles, labels, title="Late Delivery Risk", labels=["Risk", "No Risk"], loc='upper right', bbox_to_anchor=(0.9, 1), ncol=2)

    plt.suptitle("Histograms of Numerical Features by Delivery Risk", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    plt.savefig(os.path.join(OUTPUT_DIR, "Histograms_of_Numerical_Features_by_Delivery_Risk.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True) 
    
def calculate_late_percentage(data_frame):
    '''
    Calculate the percentage of late deliveries in the dataset.

    Parameters:
        - data_frame: Pandas DataFrame containing the dataset.

    Returns:
        - df: DataFrame with absolute and relative frequencies of late deliveries.
        - df_total: DataFrame with total counts and percentages.    
    '''
    
    # Compute the absolute and relative frequencies 
    late_count = data_frame['late_delivery_risk'].value_counts()
    late_count_n = data_frame['late_delivery_risk'].value_counts(normalize=True).round(4) * 100

    # Combine absolute and relative frequencies into a single DataFrame
    df = pd.concat([late_count, late_count_n], axis=1)
    df.columns = ["Absolute frequency", "Relative frequency"]

    # Set the 'is_fraud' values (0, 1) as the index of the table
    df.index = df.index.map({0: 'Non-Risk', 1: 'Risk'})  

    # Create a row for the total counts and percentages and append it to the DataFrame
    total_absolute = late_count.sum()
    total_relative = late_count_n.sum()
    total_row = pd.DataFrame({
        'Absolute frequency': [total_absolute],
        'Relative frequency': [total_relative]
    }, index=['Total']) 

    # Concatenate the totals row to the existing DataFrame
    df_total = pd.concat([df,total_row])
    
    # Donut chart
    fig_do, ax_do = plt.subplots(figsize=(7, 5))
    
    # Plot the Donut chart
    wedges, texts, autotexts = ax_do.pie(df['Absolute frequency'], 
                                        labels=df['Absolute frequency'].index, 
                                        autopct='%1.2f%%', 
                                        startangle=90, 
                                        colors=["#1F4E79", "#8F2C78"], 
                                        pctdistance=0.6, 
                                        explode=(0, 0.005), 
                                        wedgeprops={'width': 0.3})
    
    ax_do.set_title('Late Non-Risk vs. Risk Delivery Relative Frequency', fontsize=14, fontweight='regular', color='black')
    ax_do.set_ylabel('')

    for i, (label, text) in enumerate(zip(texts, autotexts)):
        if i == 0:  
            label.set_position((-0.85, -0.9))  
            text.set_position((-0.95, -0.75))  
            label.set_color('#1F4E79')
            text.set_color('#1F4E79') 
            label.set_fontweight('black') 
            label.set_fontsize(12)
        else:  
            label.set_position((0.5, 1.1))   
            text.set_position((0.75, 0.95))
            label.set_color('#8F2C78')
            text.set_color('#8F2C78')  
            label.set_fontweight('black') 
            label.set_fontsize(12)

    fig_do.savefig(os.path.join(OUTPUT_DIR, f"Donut_plot_relative_frequency.png"), 
                    bbox_inches='tight', 
                    facecolor='none', 
                    transparent=True)

    return (df, df_total)
    
def plot_box_plot(dataframe):
    '''
    Create box plots for all numerical columns in the dataframe. Returns a grid of subplots.
    '''

    # Create a copy of the dataframe
    df = dataframe.copy()
    df = df.loc[:, ~df.columns.str.contains('id', case=False)]
    df = df.loc[:, ~df.columns.str.contains('date', case=False)]
    
    # Get the number of columns
    num_columns = len(df.columns)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=9, ncols=3, figsize=(14, 20))
    axes = axes.flatten()
    if num_columns == 1:
        axes = [axes]  

    # Loop through each column and plot the histogram
    for i, column in enumerate(df.columns):
        sns.boxplot(y=df[column],
                legend=False, 
                color="#6EE5D9",
                showfliers= True,
                ax=axes[i])
        axes[i].set_ylabel(f"{column.replace('_', ' ').title()}", fontsize=12)
        axes[i].grid(axis='y', linestyle='--', alpha=0.3)
        axes[i].grid(axis='x', visible=False)
        axes[i].tick_params(axis='x', which='both', bottom=False, top=False)

    # Hide any unused axes   
    for j in range(num_columns, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Box Plots Distributions", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    plt.savefig(os.path.join(OUTPUT_DIR, "Box_Plots_Distributions.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True) 
    
def plot_side_by_side_box_plots(dataframe):
    '''
    Create histograms for all numerical columns in the dataframe, with late_delivery_risk as hue.
    Returns a grid of subplots.
    '''

    # Create a copy of the dataframe
    df = dataframe.copy()
    df = df.loc[:, ~df.columns.str.contains('id', case=False)]
    df = df.loc[:, ~df.columns.str.contains('date', case=False)]

    late_delivery_risk = df['late_delivery_risk']
    df = df.drop(columns=['late_delivery_risk']) 
    
    # Get the number of columns
    num_columns = len(df.columns)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=9, ncols=3, figsize=(14, 20))
    axes = axes.flatten()
    if num_columns == 1:
        axes = [axes]  


    # Loop through each column and plot the histogram
    for i, column in enumerate(df.columns):       
        sns.boxplot(df, y=column,
                legend=False, 
                palette=custom_colors,
                hue=late_delivery_risk,
                showfliers= True,
                ax=axes[i])
        axes[i].set_ylabel(f"{column.replace('_', ' ').title()}", fontsize=12)
        axes[i].grid(axis='y', linestyle='--', alpha=0.3)
        axes[i].grid(axis='x', visible=False)
        axes[i].tick_params(axis='x', which='both', bottom=False, top=False)
        #axes[i].legend(title="Late Delivery Risk", labels=["Risk", "No Risk"])

    # Hide any unused axes   
    for j in range(num_columns, len(axes)):
        axes[j].axis('off')
        
    handles, labels = axes[0].get_legend_handles_labels()  # Get the legend from the first plot
    fig.legend(handles, labels, title="Late Delivery Risk", labels=["No Risk", "Risk"], loc='upper right', bbox_to_anchor=(0.9, 1), ncol=2)

    plt.suptitle("Box Plot of Numerical Features by Delivery Risk", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    plt.savefig(os.path.join(OUTPUT_DIR, "Box_Plot_of_Numerical_Features_by_Delivery_Risk.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True)

def corr_heatmap(dataframe):
    '''
    Create a correlation heatmap for the numerical columns in the dataframe.
    Returns the correlation values with the target variable.
    '''

    df = dataframe.copy()
    df = df.loc[:, ~df.columns.str.contains('id', case=False)]
    df = df.loc[:, ~df.columns.str.contains('date', case=False)]

    # Moving price columns as last
    df = df[[col for col in df.columns if col != 'late_delivery_risk'] + ['late_delivery_risk']]

    # Correlation matrix calculation and heatmap
    correlation_matrix = df.corr()

    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    plt.figure(figsize=(20, 15))
    sns.heatmap(correlation_matrix, 
                    cmap="magma_r", 
                    linewidths=0.5, 
                    annot=True, 
                    fmt=".1f",
                    xticklabels=[col.replace('_', ' ').title() for col in df.columns],
                    yticklabels=[col.replace('_', ' ').title() for col in df.columns], 
                    mask=mask
                    )
    plt.title("Correlation Heatmap", fontsize=14)
    plt.xticks(rotation=90)
    plt.gca().tick_params(colors='black', labelsize=10, labelcolor='black', which='both', width=2)
   # plt.gca().tick_params(axis='both', which='major', labelsize=10, labelcolor='white', labelrotation=0, length=6, width=2)

    plt.savefig(os.path.join(OUTPUT_DIR, "Correlation_Heatmap.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True) 

    # Looking for correlations with the target for printing
    correlation_with_price = df.corrwith(df["late_delivery_risk"]).sort_values(ascending=False)
    correlation_with_price = correlation_with_price.round(5)

    return correlation_with_price

def aggregate_data_by_day(data_frame, feature='cumulative_profit'):
    '''
    Aggregates the data by day for a specified feature.
    
    Parameters:
        - data_frame (pd.DataFrame): The dataset with datetime and the feature to be aggregated.
        - feature (str): The name of the column to aggregate (default is 'cumulative_profit').
    
    Returns:
        - pd.DataFrame: Aggregated dataset with the feature summed up per day.
    '''

    df_copy = data_frame.copy()
    df_copy['order_date'] = df_copy['order_date'].dt.date
    
    # Ensure the feature column exists in the DataFrame
    if feature not in df_copy.columns and feature != 'sales_count':
        raise ValueError(f"Feature '{feature}' not found in the dataset.")
    
    # Selecting the 'order_date' and the specified feature to aggregate
    if feature in df_copy.columns:
        df_new_copy = df_copy[['order_date', feature]]
        
        # Group by date and sum the specified feature per day
        aggregated_df = df_new_copy.groupby('order_date').agg({feature: 'sum'}).reset_index()
    elif feature == 'sales_count':
        df_new_copy = df_copy[['order_date', 'sales']]
        
        # Group by date and sum the specified feature per day
        aggregated_df = df_new_copy.groupby('order_date').agg({'sales': 'count'}).reset_index()
    
    return aggregated_df

def plot_feature_trends_per_day(data_frame, feature='sales_count'):
    '''
    Plots the specified feature trends per day.
    '''

    df = data_frame.copy()

    # Aggregate the data by day first
    df_new = aggregate_data_by_day(df, feature)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if feature in df.columns:
        # Plotting the specified feature
        ax.plot(df_new['order_date'], 
                df_new[feature], 
                label=f'{feature.replace("_"," ").title()}', 
                marker='o', 
                color='#6EE5D9')
    elif feature == 'sales_count':
        # Plotting the specified feature
        ax.plot(df_new['order_date'], 
                df_new['sales'], 
                label=f'{feature.replace("_"," ").title()}', 
                marker='o', 
                color='#6EE5D9')
    
    # Formatting the plot
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(f'{feature.replace("_"," ").title()} per Day', fontsize=12)
    ax.set_title(f'{feature.replace("_"," ").title()} per Day Over Time', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)

    # Currency formatter function
    plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, f'{feature.replace("_"," ").title()}_per_Day.png'), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True) 

    plt.show()

def plot_customer_metrics(dataframe, customer_col='customer_id', sales_col='benefit_per_order', date_col='order_date'):
    '''
    Create histograms for customer metrics: total orders and total sales.
    '''
    
    df = dataframe.copy()
    
    # Ensure date column is in datetime format
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Compute metrics
    order_counts = df[customer_col].value_counts()  # Total orders per customer
    total_sales = df.groupby(customer_col)[sales_col].sum()  # Total benefit per customer
    tenure = df.groupby(customer_col)[date_col].agg(['min', 'max'])  # First and last purchase dates
    tenure['tenure_days'] = (df[date_col].max() - tenure['min']).dt.days  # Compute tenure in days

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    # Plot 1: Frequency of Total Orders per Customer
    sns.histplot(order_counts, bins=30, kde=True, ax=axes[0], color="#6EE5D9")
    axes[0].set_title("Distribution of Total Orders per Customer", fontsize=14)
    axes[0].set_xlabel("Total Orders", fontsize=12)
    axes[0].set_ylabel("Number of Customers", fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Plot 2: Total Sales per Customer
    sns.histplot(total_sales, bins=30, kde=True, ax=axes[1], color="#F39C12")
    axes[1].set_title("Distribution of Total Profits per Customer", fontsize=14)
    axes[1].set_xlabel("Total Profits", fontsize=12)
    axes[1].set_ylabel("Number of Customers", fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].xaxis.set_major_formatter(FuncFormatter(currency_formatter))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Customers_Metric.png'), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True) 
    plt.show()

def bar_plot(dataframe):
    '''
    Create bar plots for all categorical columns in the dataframe. Returns a grid of subplots.
    '''

    # Create a copy of the dataframe
    df = dataframe.copy()
    df = df.loc[:, ~df.columns.str.contains('id', case=False)]
    df = df.loc[:, ~df.columns.str.contains('date', case=False)]
    
    # Get the number of columns
    num_columns = len(df.columns)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 20))
    axes = axes.flatten()
    if num_columns == 1:
        axes = [axes]  

    # Loop through each column and plot the bar plots
    for i, column in enumerate(df.columns):

        counts = df[column].value_counts()

        sns.barplot(x=counts.index, y=counts.values, color='#6EE5D9', alpha=1.0, ax=axes[i])
        #axes[i].set_title(f"Histogram of {column.replace('_', ' ').title()}", fontsize=14)
        axes[i].set_xlabel(f"{column.replace('_', ' ').title()}", fontsize=12)
        axes[i].set_ylabel('Frequency', fontsize=12)
        axes[i].grid(axis='y', linestyle='--', alpha=0.3)
        axes[i].grid(axis='x', visible=False)

    # Hide any unused axes   
    for j in range(num_columns, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Bar Plot of Categorical Features", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    plt.savefig(os.path.join(OUTPUT_DIR, "Bar_Plot_of_Categorical_Features.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True) 
    
def plot_large_categorical(data_frame, category):
    '''
    Create a bar plot for a large categorical feature in the dataframe.
    '''
    
    # Count occurrences 
    counts = data_frame[category].value_counts().reset_index()
    counts.columns = [category, 'count'] 

    # Bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(y = category, hue=category, x='count', data=counts.head(14), palette='rocket', legend = False)
    #plt.title(f' {category.replace("_", " ").title()} vs. Number of Fraudulent Transactions')
    plt.xlabel('Number of Orders', fontsize=12)
    plt.ylabel(f'{category.replace("_", " ").title()}', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    # Saving the plot
    plt.savefig(os.path.join(OUTPUT_DIR, f"Box_plot_{category}.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True)
    
def plot_late_vs_nonlate_means(data_frame, category):
    '''
    Create a bar plot comparing the mean transaction profit for late and non-late deliveries by category.
    '''

    # Count fraud transactions per category and select the top 14
    top_fraud_categories = (
        data_frame[data_frame['late_delivery_risk'] == 1].groupby(category)['late_delivery_risk']
        .count().nlargest(14).index
    )

    # Filter data to include only the top fraud categories
    filtered_data = data_frame[data_frame[category].isin(top_fraud_categories)]

    # Group by category and fraud status, then calculate the mean transaction amount
    category_means = filtered_data.groupby([category, 'late_delivery_risk'])['benefit_per_order'].mean().reset_index()

    # Pivot the data for side-by-side comparison
    category_pivot = category_means.pivot(index=category, columns='late_delivery_risk', values='benefit_per_order').reset_index()
    category_pivot.columns = [category, 'No-Risk', 'Risk']
    category_pivot = category_pivot.sort_values(by='Risk', ascending=False)
    melted_data = category_pivot.melt(id_vars=category, var_name='Late Risk', value_name='Mean Profit')

    # Bar chart
    plt.figure(figsize=(10, 5))
    sns.barplot(x=category, y='Mean Profit', hue='Late Risk', data=melted_data, 
                palette=custom_colors)

    plt.title(f'Mean Transaction Profit: Late Delivery No-Risk vs.Risk by {category.replace("_"," ").title()}', fontsize=16)
    plt.xlabel(f'{category.replace("_"," ").title()}', fontsize=12)
    plt.ylabel('Mean Profit Amount', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(fontsize=12, loc= 'upper left', bbox_to_anchor=(1.01, 1))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Saving the plot
    plt.savefig(os.path.join(OUTPUT_DIR, f"Mean_Profit_by_{category}.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True)

def plot_feature_by_late_risk(data, feature):
    '''
    Create a bar plot comparing the number of orders by feature and late delivery risk.
    '''

    # Group by 'late_delivery_risk' and calculate the mean of the feature
    grouped_data = data.groupby(['late_delivery_risk', feature]).size().reset_index(name='count')

    # Plotting the bar plot
    plt.figure(figsize=(8, 6))
    sns.barplot(x=feature, y='count', data=grouped_data, hue= 'late_delivery_risk', palette=custom_colors)
    
    plt.title(f'Number of Order by {feature.replace("_", " ").title()} by Late Delivery Risk', fontsize=14)
    plt.xlabel(f'{feature.replace("_", " ").title()}', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, ["No Risk", "Risk"], title="Late Delivery Risk", loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=12)

    
    plt.savefig(os.path.join(OUTPUT_DIR, f"Number_of_Order_by_{feature.replace("_", " ").title()}_by_Late_Delivery_Risk.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True)
    plt.show()


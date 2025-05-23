'''This file groups functions for data cleaning, such as 
    formatting columns to a consistent format.'''

import pandas as pd

def convert_to_datetime(data_frame, columns):
    '''
    Converts specified columns to datetime format.
    
    Parameters:
        - data_frame (pd.DataFrame): The input DataFrame.
        - columns (str or list): Column name or list of column names to convert to datetime.
    
    Returns:
        - pd.DataFrame: The DataFrame with specified columns converted to datetime.
    '''

    # If a single column is provided, convert it to a list
    if isinstance(columns, str):
        columns = [columns]
    
    # Loop through each specified column and convert to datetime
    for col in columns:
        if not pd.api.types.is_datetime64_any_dtype(data_frame[col]):
            data_frame[col] = pd.to_datetime(data_frame[col], errors='coerce')
    
    return data_frame

def convert_to_str(data_frame, columns):
    '''
    Convert column to string
    Parameters:
        - data_frame (pd.DataFrame): The input DataFrame whose columns need to be formatted.

    Returns:
        - pd.DataFrame: DataFrame with input columns converted to str.

    '''

    # If a single column is provided, convert it to a list
    if isinstance(columns, str):
        columns = [columns]

    for col in columns:
        data_frame[col] = data_frame[col].astype(object)

    return data_frame

def convert_to_int(data_frame, columns):
    '''
    Convert column to int
    Parameters:
        - data_frame (pd.DataFrame): The input DataFrame whose columns need to be formatted.

    Returns:
        - pd.DataFrame: DataFrame with input columns converted to str.

    '''

    # If a single column is provided, convert it to a list
    if isinstance(columns, str):
        columns = [columns]

    for col in columns:
        data_frame[col] = data_frame[col].astype(int)

    return data_frame

def drop_rows_with_nan(data_frame, columns):
    """
    Drops rows from the DataFrame where the specified column(s) have NaN values.

    Parameters:
        - data_frame (pd.DataFrame): The input DataFrame.
        - columns (str or list): Column name or list of column names to check for NaN values.

    Returns:
        - pd.DataFrame: A new DataFrame with the rows removed.
    """
    # Ensure columns is a list
    if isinstance(columns, str):
        columns = [columns]

    data_frame = data_frame.dropna(subset=columns)

    return data_frame

def clean_for_database(data_frame):
    '''
    A function to call functions for data cleaning so the data can 
    be saved in a dataframe.
    
    Parameters:
        - data_frame (pd.DataFrame): The input DataFrame to clean.
    
    Returns:
        - pd.DataFrame: The cleaned DataFrame.
    '''
    # Convert columns to datetime
    data_frame = convert_to_datetime(data_frame,['order date (DateOrders)', 'shipping date (DateOrders)'])
    # Convert columns to str
    data_frame = convert_to_str(data_frame,['Product Description', 'Order Zipcode'])
    # Drop rows with NaN values in 'Customer Zipcode'
    data_frame = drop_rows_with_nan(data_frame, 'Customer Zipcode')

    return data_frame

def drop_col(data_frame, columns):
    '''
    Drops specified columns from a DataFrame.
    
    Parameters:
        - data_frame (pd.DataFrame): The input DataFrame from which columns will be dropped.
        - columns (list or str): A list of column names or a single column name to be dropped.
    
    Returns:
        - pd.DataFrame: The DataFrame with the specified columns removed.
    '''

    # Check for columns that do not exist in the DataFrame
    missing_cols = [col for col in columns if col not in data_frame.columns]

    # If there are missing columns, print a message and exclude them from the drop list
    if missing_cols:
        print(f"Warning: The following columns were not found and will be skipped: {', '.join(missing_cols)}")
        columns = [col for col in columns if col in data_frame.columns]  # Keep only existing columns
    
    # Drop the existing columns
    data_frame = data_frame.drop(columns, axis=1)

    return data_frame

def snake(data_frame):
    '''
    Converts column names to snake_case (lowercase with underscores).
    
    Parameters:
        - data_frame (pd.DataFrame): The input DataFrame whose columns need to be formatted.

    Returns:
        - pd.DataFrame: DataFrame with column names in snake_case.
    '''

    data_frame.columns = [column.lower().replace(" ", "_").replace(")", "").replace("(", "") for column in data_frame.columns]

    return data_frame

def column_name(data_frame, columns, word_to_remove):
    '''
    Formats columns name.

    Parameters:
        - data_frame (pd.DataFrame): The input DataFrame.
        - columns (list): List of column names to modify.
        - word_to_remove (str): The word to remove from the column names.

    Returns:
        - pd.DataFrame: The DataFrame with the updated column name.
    '''

    for column in columns:
        # If the column exists in the DataFrame, remove the word from the column name
        if column in data_frame.columns:
            new_column = column.replace(word_to_remove, '')
            data_frame = data_frame.rename(columns={column: new_column})

    return data_frame

def clean(data_frame):
    '''
    A function to call functions for data cleaning.
    
    Parameters:
        - data_frame (pd.DataFrame): The input DataFrame to clean.
    
    Returns:
        - pd.DataFrame: The cleaned DataFrame.
    '''
    # Convert to snake case
    data_frame = snake(data_frame)
    # Drop columns
    data_frame = drop_col(data_frame, ['product_description', 'product_image', 'product_status', 'order_zipcode', 'customer_lname', 'customer_fname', 'customer_email', 'customer_password', 'customer_street'])
    # Convert to datetime
    data_frame = convert_to_datetime(data_frame, ['order_date_dateorders', 'shipping_date_dateorders'])
    # Converting to int
    data_frame = convert_to_int(data_frame, 'customer_zipcode')
    # Editing column names
    data_frame = column_name(data_frame, ['order_date_dateorders', 'shipping_date_dateorders'], '_dateorders')


    return data_frame

def preprocess_datetime(data_frame, columns):
    '''
    Extracts time-based features from 'trans_date_trans_time'.
    
    Parameters:
        - data_frame (pd.DataFrame): The dataset containing the transaction date/time column.
    
    Returns:
        - pd.DataFrame: The modified dataframe with new time-based columns.
    '''

    if isinstance(columns, str):
        columns = [columns]
    df = data_frame.copy()

    # Mapping days to numbers
    day_mapping = {
        'Monday': 1,
        'Tuesday': 2,
        'Wednesday': 3,
        'Thursday': 4,
        'Friday': 5,
        'Saturday': 6,
        'Sunday': 7
    }

    for column in columns:
        if 'order' in column.lower():
            word = 'order'
        else:
            word = 'shipping'

        df[f'year_{word}'] = df[column].dt.year
        df[f'hour_{word}'] = df[column].dt.hour
        df[f'day_{word}'] = df[column].dt.day
        df[f'month_{word}'] = df[column].dt.month
        df[f'day_of_week_{word}'] = df[column].dt.day_name()

        # Convert days to numbers
        df[f'day_of_week_{word}'] = df[f'day_of_week_{word}'].map(day_mapping)

    #df = df.drop(columns=['order_date', 'shipping_date'])

    return df

def cumulative_profit(dataframe):
    '''
    Calculates the cumulative profit for each row in the DataFrame.
    
    Parameters:
        - dataframe (pd.DataFrame): The input DataFrame containing the profit column.
    
    Returns:
        - pd.DataFrame: The DataFrame with an additional column for cumulative profit.
    '''
    df = dataframe.copy()
    df['cumulative_profit'] = df['benefit_per_order'].cumsum()

    return df

def shipping_delay(dataframe):
    '''
    Calculates the shipping delay for each order.
    
    Parameters:
        - dataframe (pd.DataFrame): The input DataFrame containing the order and shipping dates.
    
    Returns:
        - pd.DataFrame: The DataFrame with an additional column for shipping delay.
    '''
    df = dataframe.copy()
    df['shipping_delay'] = (df['days_for_shipping_real'] - df['days_for_shipment_scheduled'])

    return df
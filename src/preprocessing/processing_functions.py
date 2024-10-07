import pandas as pd
import re
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

def drop_id(df):
    """
    Drops columns from the DataFrame that contain certain identifying strings such as 'id' and 'site_id' 
    in their column headers. This is useful for removing columns related to geographical or identification data 
    that are not needed for analysis.

    Parameters:
    df (pd.DataFrame): The input DataFrame from which columns will be dropped.

    Returns:
    pd.DataFrame: The DataFrame with the specified columns removed.
    """
    # List of strings to identify columns to delete
    strings_to_delete = ['id', 'site_id']
    
    # Construct a refined regex pattern with word boundaries for 'id'
    # Assuming 'id' should be a standalone word or preceded/followed by an underscore
    # Adjust the pattern based on your specific column naming conventions
    refined_patterns = [
        r'\bid\b',           # Matches 'id' as a whole word
        r'_id\b',            # Matches strings ending with '_id'
        r'\bid_',            # Matches strings starting with 'id_'
        'site_id'
    ]
    pattern = '|'.join(refined_patterns)
    
    # Compile the regex pattern for better performance
    regex = re.compile(pattern, flags=re.IGNORECASE)
    
    # Identify columns that contain any of the specified patterns
    pidd_columns = [col for col in df.columns if regex.search(col)]
    
    # Count the number of columns to be dropped
    num_columns_to_drop = len(pidd_columns)
    
    # Debug: List the columns being removed
    if num_columns_to_drop > 0:
        print(f"Columns to be removed ({num_columns_to_drop}): {pidd_columns}")
    else:
        print("No columns matched the specified patterns to remove.")
    
    # Drop the identified columns
    df_dropped = df.drop(columns=pidd_columns)
    
    # Count the remaining columns
    num_columns_remaining = df_dropped.shape[1]
    
    # Print the summary statement
    print(f"Removed {num_columns_to_drop} column(s). {num_columns_remaining} column(s) remain.")
    
    return df_dropped

def drop_uvaerosollayerheight_columns(df):
    """
    Drops columns from the DataFrame that contain the word 'uvaerosollayerheight' in their column headers.
    This function is used to remove columns related to the 'uvaerosollayerheight' sensor, 
    which may have limited or irrelevant data for analysis.

    Parameters:
    df (pd.DataFrame): The input DataFrame from which columns will be dropped.

    Returns:
    pd.DataFrame: The DataFrame with the 'uvaerosollayerheight' columns removed.
    """
    # Define the pattern to identify columns to delete
    pattern = 'uvaerosollayerheight'
    
    # Identify columns that contain the specified pattern (case-insensitive)
    target_columns = df.columns[df.columns.str.contains(pattern, case=False, regex=True)]
    
    # Count the number of columns to drop
    num_columns_to_drop = len(target_columns)
    
    # Drop the identified columns without modifying the original DataFrame in place
    df_dropped = df.drop(columns=target_columns)
    
    # Count the remaining columns
    num_columns_remaining = df_dropped.shape[1]
    
    # Print the summary statement
    if num_columns_to_drop > 0:
        print(f"Removed {num_columns_to_drop} column(s) containing '{pattern}'. {num_columns_remaining} column(s) remain.")
    else:
        print(f"No columns containing '{pattern}' were found. {num_columns_remaining} column(s) remain.")
    
    return df_dropped

def convert_date(df, date_column='date', date_format='%Y-%m-%d'):
    """
    Converts a specified date column in the DataFrame to datetime format.
    Sorts the DataFrame by the date column after conversion.
    Prints informative statements about the conversion process.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing the date column.
    date_column : str, optional
        The name of the column to convert to datetime. Default is 'date'.
    date_format : str, optional
        The format of the date strings in the column. Default is '%Y-%m-%d'.

    Returns:
    -------
    pd.DataFrame
        The DataFrame with the date column converted to datetime and sorted.
    """
    if date_column not in df.columns:
        raise ValueError(f"The specified column '{date_column}' does not exist in the DataFrame.")

    # Count missing values before conversion
    missing_before = df[date_column].isnull().sum()
    print(f"Missing values in '{date_column}' before conversion: {missing_before}")

    try:
        # Convert the specified date column to datetime
        df[date_column] = pd.to_datetime(df[date_column], format=date_format, errors='coerce')

        # Count missing values after conversion (due to parsing errors)
        missing_after = df[date_column].isnull().sum()
        if missing_after > 0:
            print(f"Missing or invalid dates after conversion: {missing_after}")
        else:
            print(f"All dates in '{date_column}' successfully converted to datetime format.")

        return df

    except Exception as e:
        print(f"An error occurred while converting '{date_column}' to datetime: {e}")
        return df

def ordinal_encoding_dates(df):
    """
    Applies ordinal encoding to the 'date' column in df by converting the dates into ordinal values 
    (integer labels based on their order of appearance). The data is first sorted by the 'date' column 
    before encoding, and df is returned with reset indices.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing a 'date' column, which will be encoded.

    Returns:
    pd.DataFrame: The DataFrame with ordinal encoding applied to the 'date' column.
    """
    # Sort the DataFrame by the date column
    df_sorted = df.sort_values(by='date').reset_index(drop=True)
    print(f"DataFrame sorted by 'date' in ascending order.\n")

    return df_sorted

def impute_missing_values(df, random_state=42):
    """
    Handles missing values in the DataFrame by imputing numerical columns using IterativeImputer.
    Prints the number of missing values before and after imputation.
    
    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame with potential missing values.
    random_state : int, default=0
        Random state for the IterativeImputer to ensure reproducibility.
    
    Returns:
    -------
    pd.DataFrame
        The DataFrame with missing values imputed in numerical columns.
    """
    # Identify numerical columns
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Check if there are any missing values in numerical columns
    total_missing_before = df[numerical_columns].isnull().sum().sum()
    if total_missing_before == 0:
        print("No missing values detected in numerical columns. Imputation not required.\n")
        return df
    
    # Initialize IterativeImputer
    imputer = IterativeImputer(random_state=random_state)
    
    # Display the number of missing values before imputation
    print(f"Number of missing values in numerical columns before imputation: {total_missing_before}")
    
    # Fit the imputer on the data and transform it
    df_imputed = imputer.fit_transform(df[numerical_columns])
    
    # Convert the numpy array back to a DataFrame
    df_imputed = pd.DataFrame(df_imputed, columns=numerical_columns, index=df.index)
    
    # Replace the original numerical columns with the imputed data
    df_cleaned = df.copy()
    df_cleaned[numerical_columns] = df_imputed
    
    # Display the number of missing values after imputation
    total_missing_after = df_cleaned[numerical_columns].isnull().sum().sum()
    print(f"Number of missing values in numerical columns after imputation: {total_missing_after}\n")
    
    return df_cleaned

def extract_date_components(df):
    """
    The function extracts specific components such as the day and the day of the week from the 'date' column.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing a 'date' column.

    Returns:
    pd.DataFrame: The DataFrame with  new columns for 'day' and 'day_of_week' extracted from the 'date' column.
    """
    # Extract month, day, and day of the week from the 'date' column
    # df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek

    return df

def scale_numerical_features(df, exclude_columns=['date', 'target'], scaler=None, print_output=True):
    """
    Scales numerical features in the DataFrame using the specified scaler, excluding certain columns.
    
    Parameters:
    - df : pd.DataFrame
        The DataFrame containing the data to be scaled.
    - exclude_columns : list, optional
        List of non-numerical columns to exclude from scaling. Default is ['date', 'target'].
    - scaler : sklearn scaler object, optional
        Scaler to use for scaling numerical features. If None, StandardScaler is used.
    - print_output : bool, optional
        Whether to print the numerical features and the scaled DataFrame. Default is True.
    
    Returns:
    - df_scaled : pd.DataFrame
        The DataFrame with scaled numerical features.
    - numerical_features : list
        List of numerical feature names that were scaled.
    - scaler : sklearn scaler object
        The scaler that was used (useful for inverse_transform or future scaling).
    """
    if scaler is None:
        scaler = StandardScaler()
    
    # Identify numerical features
    numerical_features = [col for col in df.columns 
                          if col not in exclude_columns and df[col].dtype in ['int64', 'float64']]
    
    if print_output:
        print("\nNumerical Features to Scale:", numerical_features)
    
    # Initialize the scaler
    scaler = scaler
    
    # Fit the scaler to the numerical columns and transform
    df_scaled = df.copy()
    df_scaled[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    if print_output:
        print("\nDataFrame After Standard Scaling:")
        print(df_scaled)
    
    return df_scaled, numerical_features, scaler
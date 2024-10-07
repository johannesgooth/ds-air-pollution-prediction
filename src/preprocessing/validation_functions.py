def detect_outliers_iqr(df, column='pm2_5'):
    """
    Detects outliers in a specified column using the Interquartile Range (IQR) method.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column (str): The column for which to calculate the outliers (default is 'pm2_5').

    Returns:
    outliers_iqr (pd.DataFrame): A DataFrame containing the rows identified as outliers.
    iqr_info (dict): A dictionary containing Q1, Q3, IQR, and the upper bound used for outlier detection.
    """
    
    # Calculate Q1 and Q3
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    # Calculate IQR
    IQR = Q3 - Q1

    # Define upper bound
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers_iqr = df[df[column] > upper_bound]

    # Print summary
    print(f"Q1: {Q1}")
    print(f"Q3: {Q3}")
    print(f"IQR: {IQR}")
    print(f"Upper Bound: {upper_bound}")
    print(f"Number of outliers detected using IQR: {outliers_iqr.shape[0]}")

    # Return the outliers and IQR information
    iqr_info = {
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'Upper Bound': upper_bound
    }

    return upper_bound
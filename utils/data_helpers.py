"""
Helpers for processing and pre-processing a specific
banking dataset
"""
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score


def add_date_cols(df, start_year=2008, start_month=5):
    """
    Adds a "date" column assuming that the data is in order,
    we can trust the "day" column, and we know the year and
    month that it starts
    :param df: input dataframe
    :param start_year: the year of the first row of data
    :param start_month: the integer of the month of the first row of data
    :return: A dataframe with new columns:
                "year" (int) e.g. 2018
                "date" (date) e.g. 2018-05-12
                "day_of_week_name" (str) e.g. Monday/Tuesday
    """
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sept': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    # Map the month names to month numbers
    df['month_num'] = df['month'].map(month_map)
    years = []
    current_year = start_year
    current_month = start_month
    for month in df['month_num']:
        if month < current_month:
            current_year += 1
        years.append(current_year)
        current_month = month
    df['year'] = years
    df['date'] = pd.to_datetime(df[['year', 'month_num', 'day']].rename(columns={'month_num': 'month'}))
    df['day_of_week_name'] = df['date'].dt.day_name()
    return df


def extract_data(filepath, cat_cols):
    """
    Extract the data, add in date features, and

    :param filepath: the full filepath to the data source
    :return: an expanded dataframe with date columns and categorical features one-hot encoded
    """
    df = pd.read_csv(filepath, sep=';')
    #TODO: Exception checking:
    #       check that all the expected columns are there,
    #       throw errors if there are any new or missing
    #       throw error if the success column is not yes/no
    df = df.rename(columns={'y': 'success'})
    df = add_date_cols(df)

    # One-hot encode categorical variables
    cols_to_encode = cat_cols + ['day_of_week_name']
    df_encoded = pd.get_dummies(df[cols_to_encode], prefix_sep='_cat_', columns=cols_to_encode, drop_first=True)
    # Concatenate the original DataFrame with the dummy encoded DataFrame
    df_concatenated = pd.concat([df, df_encoded], axis=1)
    ##TODO throw nicer error if there are no data rows
    return df_concatenated


def check_model_quality(clf, X_test, y_test, clf_name):
    """
    Checks & prints out the accuracy, precision, recall, f1, and ROC
    of a specific classifier against a test set

    :param clf: a trained model
    :param X_test: the test input data
    :param y_test: the test output/success data
    :param clf_name: the name of the classifier, to be printed
    :return: None
    """
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='yes')
    recall = recall_score(y_test, y_pred, pos_label='yes')
    f1 = f1_score(y_test, y_pred, pos_label='yes')
    print(f'{clf_name} Accuracy: {accuracy:.2f}')
    print(f'{clf_name} Precision: {precision:.2f}')
    print(f'{clf_name} Recall: {recall:.2f}')
    print(f'{clf_name} F1-score: {f1:.2f}')

    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f'{clf_name} ROC AUC: {auc:.2f}')

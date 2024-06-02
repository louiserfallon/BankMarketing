"""
This script performs data preprocessing, quick plotting,
and rough classification modelling for the bank marketing dataset
in order to analyse the most important features
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree

from utils import plot_helpers as ph, data_helpers as dh


def main(filepath='data/bank/bank-full.csv', no_figs=False, no_train=False):
    """
    Reads in data from a filepath, expecting to be in the same format
    Generates and saves a series of exploratory charts
    Trains a few ML models and records the quality of them against a test set

    :param filepath: data location
    :param no_figs: flag to not run the "figure generation" section
    :param no_train: flag to not run the "ML training" section
    :return: None
    """
    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan',
                'contact', 'poutcome', 'year', 'month']
    # Do not include duration because if there is no call, there is no success
    added_cat_cols = ['day_of_week_name']
    # Do not include duration because if there is no call, there is no success
    num_cols = ['balance', 'campaign', 'pdays', 'previous']

    print(f'extracting from {filepath}')
    raw_df = dh.extract_data(filepath, cat_cols)

    dummy_cols = [x for x in raw_df if '_cat_' in x]
    print(f'read data with {raw_df.shape} shape')
    success_counts = raw_df['success'].value_counts()
    print(f'Total number of successes: {success_counts['yes']}')
    print(f'Total number of failures: {success_counts['no']}')
    print(f'Total success rate: {success_counts['yes'] / success_counts.sum():.2%}')

    raw_df['success_numeric'] = raw_df['success'].map({'yes': 1, 'no': 0})

    # Figure Generation
    if not no_figs:
        print('------Plotting all variables as categories to figures folder')
        for col in cat_cols + added_cat_cols:
            ph.export_cat_chart(raw_df, col)

        for col in ['campaign', 'previous']:
            raw_df[f'{col}_cat'] = raw_df[col].clip(upper=10)
            ph.export_cat_chart(raw_df, col=f'{col}_cat', order_asc=True)

        ph.export_cat_chart(raw_df, col='pdays', order_asc=True,
                            cuts=[-np.inf, -2, -1, 0, 50, 125, 180, 250, 365, 500, np.inf])
        ph.export_num_chart(raw_df, col='pdays')
        print('------Plotted all variables')

        print('------Plotting time series to figures folder')
        ph.export_time_chart(raw_df)
        print('------Plotted time series')

        print('------Training a simple decision tree for illustrative purposes')
        dt_clf = DecisionTreeClassifier(max_depth=5)
        illustrative_cols = [x for x in dummy_cols + num_cols
                             if ('year' not in x
                                 and 'month' not in x
                                 and 'week' not in x)]
        dt_clf.fit(raw_df[illustrative_cols], raw_df['success'])
        plt.figure(figsize=(15, 10))
        tree.plot_tree(dt_clf, filled=True,
                       feature_names=illustrative_cols,
                       class_names=['No', 'Yes'],
                       proportion=True, rounded=True)
        ph.save_to_figs(plt, col='all', chart_name='decis_tree_prop')

    # ML Training
    if not no_train:
        # Not using a time series (temporal) test/train split, since this
        # analysis is more descriptive to see if patterns generalise than it is predictive
        X_train, X_test, y_train, y_test = train_test_split(raw_df[dummy_cols + num_cols],
                                                            raw_df['success'],
                                                            test_size=0.1, random_state=8)

        # filter for clients who were not previously successful
        # to specifically check precision and recall on this segment
        new_user_filter = raw_df[raw_df['poutcome'] != 'success'].index

        print('------Training a more complex decision tree')
        big_dt_clf = DecisionTreeClassifier(max_depth=10, min_samples_leaf=10)
        big_dt_clf.fit(X_train, y_train)
        dh.check_model_quality(big_dt_clf, X_test, y_test, new_user_filter, clf_name='big_decis_tree')
        print('------Feature Importances')
        # Print out significant features (rough check)
        for importance, feature_name in sorted(zip(big_dt_clf.feature_importances_, X_train.columns), reverse=True):
            if importance > 0.01:
                print(f'{feature_name}: {importance:.4f}')
        dh.check_model_quality(big_dt_clf, X_test, y_test, new_user_filter, clf_name='decis tree')

        print(f'------Training a logistic regression')
        linear_clf = LogisticRegression(penalty='l1', solver='liblinear', C=0.5)
        linear_clf.fit(X_train, y_train)
        print("Coefficients:")
        # Print out significant coefficients (rough check
        for feature, coef in zip(X_train.columns, linear_clf.coef_[0]):
            if abs(coef) > 0.01:
                print(f"{feature}: {coef:.4f}")
        print('------Checking model quality')
        dh.check_model_quality(linear_clf, X_test, y_test, new_user_filter, clf_name='linear_lasso')

        print(f'------Training a GBDT')
        gbdt_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_depth=5)
        gbdt_clf.fit(X_train, y_train)
        print('------Feature Importances')
        # Print out significant features (rough check)
        for importance, feature_name in sorted(zip(gbdt_clf.feature_importances_, X_train.columns), reverse=True):
            if importance > 0.01:
                print(f'{feature_name}: {importance:.4f}')
        dh.check_model_quality(gbdt_clf, X_test, y_test, new_user_filter, clf_name='gbdt')


if __name__ == "__main__":
    # Take arguments for the main command line argument
    # to turn off either major section of the main code (figures or training)
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-figs", action="store_true", help="Skip plotting categorical columns")
    parser.add_argument("--no-train", action="store_true", help="Skip training quick ML models")
    args = parser.parse_args()
    main(no_figs=args.no_figs, no_train=args.no_train)

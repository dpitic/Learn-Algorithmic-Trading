"""Turtle Trading Strategy.
This is a more advanced trading strategy where a long (buy) signal is created
when the price reaches the highest price for the last number of days specified
by the window size.
"""
import pandas as pd

import algolib.data as data


def main():
    # Prevent truncating display of DataFrame
    pd.set_option('display.width', None)
    goog_data = data.get_google_data('data/goog_data.pkl',
                                     start_date='2001-01-01',
                                     end_date='2018-01-01')


if __name__ == '__main__':
    main()

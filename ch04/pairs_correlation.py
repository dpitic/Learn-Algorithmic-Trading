"""Pair Correlation Trading.
Pair trading mean reversion is based on the correlation between two instruments.
If a pair of stocks already has a high correlation, at some point, the
correlation is diminished, it will come back to the original level (correlation
mean value). If the stock with the lower price drops, we can long this stock
and short the other stock of this pair.
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import algolib.data as data
import algolib.signals as signals


def main():
    # Prevent truncating display of DataFrame
    pd.set_option('display.width', None)

    # Financial instrument symbols to load and correlate
    symbol_ids = ['SPY', 'AAPL', 'ADBE', 'LUV', 'MSFT', 'SKYW', 'QCOM',
                  'HPQ', 'JNPR', 'AMD', 'IBM']
    financial_data = data.load_financial_data(symbol_ids)
    print('Financial data:')
    print(financial_data)

    pvalues, pairs = \
        signals.find_cointegrated_pairs(financial_data['Adj Close'])
    print(pairs)

    sns.heatmap(pvalues, xticklabels=symbol_ids,
                yticklabels=symbol_ids, cmap='RdYlGn_r',
                mask=(pvalues >= 0.98))
    plt.show()


if __name__ == '__main__':
    main()

"""Trading simulation module used to build the trading system and the external
objects, and run the trading simulation loop.
"""
from collections import deque

from ch07.liquidity_provider import LiquidityProvider
from ch07.market_simulator import MarketSimulator
from ch07.order_book import OrderBook
from ch07.order_manager import OrderManager
from ch07.trading_strategy import TradingStrategy


def main():
    # One-way message channel between liquidity provider and order book
    # through the liquidity provider gateway, for liquidity provider to send
    # messages to the order book (order book gw_2_ob).
    gw_2_ob = deque()
    # One-way message channel between order book and trading strategy for
    # order book to send messages to the trading strategy.
    ob_2_ts = deque()
    # Two-way message channel between trading strategy and order manager.
    # This channel is used by the trading strategy to send messages to the
    # order manager.
    ts_2_om = deque()
    # Two-way message channel between order manager and trading strategy.
    # This channel is used by the order manager to send messages to the
    # trading strategy.
    om_2_ts = deque()
    # Two-way message channel between order manager and market simulator
    # through the market gateway. This channel is used by the order manager
    # to send messages to the market (gateway).
    om_2_gw = deque()
    # Two-way message channel between market and order manager through the
    # market gateway. This channel is used by the market to send messages to
    # the order manager through the market gateway.
    gw_2_om = deque()

    # Trading system objects
    order_book = OrderBook(gw_2_ob, ob_2_ts)
    # Initialise trading strategy with $10,000
    trading_strategy = TradingStrategy(10000, ob_2_ts, ts_2_om, om_2_ts)
    order_manager = OrderManager(ts_2_om, om_2_ts, om_2_gw, gw_2_om)

    # External components
    liquidity_provider = LiquidityProvider(gw_2_ob, random_seed=0)
    market_simulator = MarketSimulator(om_2_gw, gw_2_om)

    # Get data from source (not implemented yet)
    liquidity_provider.read_tick_data_from_data_source()
    while len(gw_2_ob) > 0:
        order_book.handle_order_from_gateway()
        trading_strategy.handle_input_from_ob()
        order_manager.handle_trading_strategy_messages()
        market_simulator.handle_order_from_order_manager_gateway()
        order_manager.handle_input_from_market()
        trading_strategy.handle_response_from_om()
        liquidity_provider.read_tick_data_from_data_source()


if __name__ == '__main__':
    main()

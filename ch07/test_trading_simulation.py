import unittest
from collections import deque

from ch07.liquidity_provider import LiquidityProvider
from ch07.market_simulator import MarketSimulator
from ch07.order_book import OrderBook
from ch07.order_manager import OrderManager
from ch07.trading_strategy import TradingStrategy


class TestTradingSimulation(unittest.TestCase):

    def setUp(self) -> None:
        # One-way message channel between liquidity provider and order book
        # through the liquidity provider gateway, for liquidity provider to send
        # messages to the order book (order book gw_2_ob).
        self.gw_2_ob = deque()
        # One-way message channel between order book and trading strategy for
        # order book to send messages to the trading strategy.
        self.ob_2_ts = deque()
        # Two-way message channel between trading strategy and order manager.
        # This channel is used by the trading strategy to send messages to the
        # order manager.
        self.ts_2_om = deque()
        # Two-way message channel between order manager and trading strategy.
        # This channel is used by the order manager to send messages to the
        # trading strategy.
        self.om_2_ts = deque()
        # Two-way message channel between order manager and market simulator
        # through the market gateway. This channel is used by the order manager
        # to send messages to the market (gateway).
        self.om_2_gw = deque()
        # Two-way message channel between market and order manager through the
        # market gateway. This channel is used by the market to send messages to
        # the order manager through the market gateway.
        self.gw_2_om = deque()

        # Trading system objects
        self.order_book = OrderBook(self.gw_2_ob, self.ob_2_ts)
        # Initialise trading strategy with $10,000
        self.trading_strategy = TradingStrategy(10000, self.ob_2_ts,
                                                self.ts_2_om, self.om_2_ts)
        self.order_manager = OrderManager(self.ts_2_om, self.om_2_ts,
                                          self.om_2_gw, self.gw_2_om)

        # External components
        self.liquidity_provider = LiquidityProvider(self.gw_2_ob, random_seed=0)
        self.market_simulator = MarketSimulator(self.om_2_gw, self.gw_2_om)

    def test_add_liquidity(self):
        """Verify 2 orders are created to arbitrage the 2 liquidities.

        Verify adding two liquidities having bid higher than offer creates
        two orders to arbitrage the two liquidities.
        """
        # New order with buy/bid price higher than sell/offer price
        order_1 = {
            'id': 1,
            'price': 219,
            'quantity': 10,
            'side': 'buy',
            'action': 'create'
        }
        # Make liquidity provider manually send this order
        manual_order = self.liquidity_provider.send_manual_order(order_1.copy())
        self.assertEqual(len(self.gw_2_ob), 1,
                         'Liquidity provider gateway should have message '
                         'for order 1.')
        # Order book processes the message from the liquidity provider
        book_event = self.order_book.handle_gateway_message()
        self.assertEqual(len(self.gw_2_ob), 0,
                         'Order book should have removed book event from '
                         'liquidity provider gateway message channel.')
        self.assertEqual(len(self.ob_2_ts), 1,
                         'Order book should have sent trading strategy the '
                         'book event message for order 1.')
        # Trading strategy processes book event message from order book
        self.trading_strategy.handle_order_book_message()
        self.assertEqual(len(self.ob_2_ts), 0,
                         'Trading strategy should have removed book event '
                         'message from message channel from order book.')
        self.assertEqual(len(self.ts_2_om), 0,
                         'No trading signal so trading strategy should not '
                         'have sent any message to order manager.')

        # New order with buy/bid price higher than sell/offer price
        order_2 = {
            'id': 2,
            'price': 218,
            'quantity': 10,
            'side': 'sell',
            'action': 'create'
        }
        # Make liquidity provider manually send this order
        manual_order = self.liquidity_provider.send_manual_order(order_2.copy())
        self.assertEqual(len(self.gw_2_ob), 1,
                         'Liquidity provider gateway should have message for '
                         'order 2.')
        # Order book processes the message from the liquidity provider
        book_event = self.order_book.handle_gateway_message()
        self.assertEqual(len(self.gw_2_ob), 0,
                         'Order book should have removed book event from '
                         'liquidity provider gateway message channel.')
        self.assertEqual(len(self.ob_2_ts), 1,
                         'Order book should have sent trading strategy message '
                         'for order 2.')
        # Trading strategy processes message from order book
        self.trading_strategy.handle_order_book_message()
        self.assertEqual(len(self.ts_2_om), 2,
                         'Trading strategy should have sent 2 messages to order'
                         'manager.')
        # Order manager processes messages from trading strategy
        self.order_manager.handle_trading_strategy_message()
        self.assertEqual(len(self.ts_2_om), 1,
                         'Order manager should have removed 1 message from '
                         'trading strategy.')
        self.assertEqual(len(self.om_2_gw), 1,
                         'Order manager should have sent 1 message to market '
                         'simulator through market gateway.')
        self.order_manager.handle_trading_strategy_message()
        self.assertEqual(len(self.ts_2_om), 0,
                         'Order manager should have removed the last message '
                         'from the trading strategy.')
        self.assertEqual(len(self.om_2_gw), 2,
                         'Order manager should have sent 2 messages to the '
                         'market simulator through the gateway.')
        # Market simulator processes orders from order manager
        self.market_simulator.handle_order_manager_message()
        self.assertEqual(len(self.gw_2_om), 2,  # book was 1
                         'Market simulator should have sent 1 message to the '
                         'order manager through the market gateway.')
        self.market_simulator.handle_order_manager_message()
        self.assertEqual(len(self.gw_2_om), 4,  # book was 2
                         'Market simulator should have sent 2 messages to the '
                         'order manager through the market gateway.')
        # Order manager processes messages from market simulator
        self.order_manager.handle_market_message()
        self.order_manager.handle_market_message()
        self.assertEqual(len(self.om_2_ts), 2,
                         'Order manager should have sent 2 messages to the '
                         'trading strategy.')
        # Trading strategy processes messages from the order manager
        self.trading_strategy.handle_order_manager_message()
        self.assertEqual(self.trading_strategy.pnl, 0,
                         'Profit and loss should be $0.')
        # Market simulator process all orders
        self.market_simulator.process_orders()
        self.assertEqual(len(self.gw_2_om), 2,
                         'Market simulator should have sent 2 messages to the '
                         'order manager.')
        # Order manager processes messages from the market simulator
        self.order_manager.handle_market_message()
        self.order_manager.handle_market_message()
        self.assertEqual(len(self.om_2_ts), 3,
                         'Order manager should have sent 3 messages to the '
                         'trading strategy.')
        # Trading strategy processes message from order manager
        self.trading_strategy.handle_order_manager_message()
        self.assertEqual(len(self.om_2_ts), 2,
                         'Trading strategy should have removed 1 message from '
                         'the message channel from the order manager.')
        self.trading_strategy.handle_order_manager_message()
        self.assertEqual(len(self.om_2_ts), 1,
                         'Trading strategy should have removed 1 message from '
                         'the message channel from the order manager.')
        self.trading_strategy.handle_order_manager_message()
        self.assertEqual(len(self.om_2_ts), 0,
                         'Trading strategy should have removed all messages '
                         'from the message channel from the order manager.')
        self.assertEqual(self.trading_strategy.pnl, 10,
                         'Trading strategy profit should be 10.')
        # All message channels should be empty
        self.assertEqual(len(self.gw_2_ob), 0,
                         'All message channels should be empty.')
        self.assertEqual(len(self.ob_2_ts), 0,
                         'All message channels should be empty.')
        self.assertEqual(len(self.ts_2_om), 0,
                         'All message channels should be empty.')
        self.assertEqual(len(self.om_2_ts), 0,
                         'All message channels should be empty.')
        self.assertEqual(len(self.om_2_gw), 0,
                         'All message channels should be empty.')
        self.assertEqual(len(self.gw_2_om), 0,
                         'All message channels should be empty.')


if __name__ == '__main__':
    unittest.main()

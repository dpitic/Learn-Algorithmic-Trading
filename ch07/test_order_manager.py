import unittest

from ch07.order_manager import OrderManager


class TestOrderManager(unittest.TestCase):
    def setUp(self) -> None:
        # No message channels configured; operate order manager as simulation
        self.order_manager = OrderManager()

    def test_receive_order_from_trading_strategy(self):
        """"Verify whether an order from the trading strategy is correctly
        processed by the order manager.
        """
        # Mock order from trading strategy
        order_1 = {
            'id': 10,
            'price': 219,
            'quantity': 10,
            'side': 'buy'
        }
        # Trading strategy creates two orders in the message channel between
        # the trading strategy and the order manager, therefore verify the
        # order manager can handle 2 orders.
        self.order_manager.handle_trading_strategy_order(order_1)
        self.assertEqual(len(self.order_manager.orders), 1)
        self.order_manager.handle_trading_strategy_order(order_1)
        self.assertEqual(len(self.order_manager.orders), 2)
        self.assertEqual(self.order_manager.orders[0]['id'], 1)
        self.assertEqual(self.order_manager.orders[1]['id'], 2)

    def test_receive_invalid_order_from_trading_strategy(self):
        """Ensure invalid order messages are rejected by the order manager."""
        order_1 = {
            'id': 10,
            'price': -219,  # invalid price; price should be > 0
            'quantity': 10,
            'side': 'buy'
        }
        self.order_manager.handle_trading_strategy_order(order_1)
        self.assertEqual(len(self.order_manager.orders), 0)

        order_1 = {
            'id': 10,
            'price': 219,
            'quantity': -10,  # invalid quantity; quantity should be > 0
            'side': 'buy'
        }
        self.order_manager.handle_trading_strategy_order(order_1)
        self.assertEqual(len(self.order_manager.orders), 0)

    def test_receive_filled_order_from_market_gateway(self):
        """Verify processing filled orders from the market."""
        # Get orders from the trading strategy to populate order list
        self.test_receive_order_from_trading_strategy()
        # Market response order execution message
        order_execution_1 = {
            'id': 2,
            'price': 219,
            'quantity': 10,
            'side': 'buy',
            'status': 'filled'
        }
        self.order_manager.handle_market_order(order_execution_1)
        self.assertEqual(len(self.order_manager.orders), 1)

    def test_receive_acknowledged_order_from_market_gateway(self):
        """Verify processing acknowledged orders from the market."""
        # Get orders from the trading strategy to populate order list
        self.test_receive_order_from_trading_strategy()
        # Market response for acknowledged order
        order_execution_1 = {
            'id': 2,
            'price': 219,
            'quantity': 10,
            'side': 'buy',
            'status': 'acknowledged'
        }
        self.order_manager.handle_market_order(order_execution_1)
        self.assertEqual(len(self.order_manager.orders), 2)
        self.assertEqual(self.order_manager.orders[1]['status'], 'acknowledged')


if __name__ == '__main__':
    unittest.main()

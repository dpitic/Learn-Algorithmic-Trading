import unittest

from ch07.market_simulator import MarketSimulator


class TestMarketSimulator(unittest.TestCase):

    def setUp(self) -> None:
        self.market_simulator = MarketSimulator()

    def test_accept_new_order(self):
        """Validate new orders are accepted by the market."""
        order_1 = {
            'id': 10,
            'price': 219,
            'quantity': 10,
            'side': 'buy',
            'action': 'create'
        }
        self.market_simulator.handle_order(order_1)
        # Market accepts all new orders
        self.assertEqual(len(self.market_simulator.orders), 1)
        self.assertEqual(self.market_simulator.orders[0]['status'], 'accepted')

    def test_reject_amend_unknown_market_order(self):
        """Validate market does not accept order amendments for orders it does
        not know about.
        """
        order_1 = {
            'id': 10,
            'price': 219,
            'quantity': 10,
            'side': 'buy',
            'action': 'amend'
        }
        self.market_simulator.handle_order(order_1)
        # Market should not accept amendment to orders it doesn't know about
        self.assertEqual(len(self.market_simulator.orders), 0)


if __name__ == '__main__':
    unittest.main()

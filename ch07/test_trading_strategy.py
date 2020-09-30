import unittest

from ch07.trading_strategy import TradingStrategy


class TestTradingStrategy(unittest.TestCase):
    def setUp(self) -> None:
        self.trading_strategy = TradingStrategy(cash=10000)

    def test_receive_top_of_book(self):
        # Test event message from the exchange (through the order book)
        book_event = {
            'bid_price': 12,
            'bid_quantity': 100,
            'offer_price': 11,
            'offer_quantity': 150
        }
        self.trading_strategy.handle_book_event(book_event)
        self.assertEqual(len(self.trading_strategy.orders), 2)
        self.assertEqual(self.trading_strategy.orders[0]['price'], 12)
        self.assertEqual(self.trading_strategy.orders[0]['quantity'], 100)
        self.assertEqual(self.trading_strategy.orders[0]['side'], 'sell')
        self.assertEqual(
            self.trading_strategy.orders[0]['action'], 'no_action')
        self.assertEqual(self.trading_strategy.orders[1]['price'], 11)
        self.assertEqual(self.trading_strategy.orders[1]['quantity'], 100)
        self.assertEqual(self.trading_strategy.orders[1]['side'], 'buy')
        self.assertEqual(
            self.trading_strategy.orders[1]['action'], 'no_action')

    def test_rejected_order(self):
        # Get book event message and handle the event
        self.test_receive_top_of_book()
        # Create market response indicating a rejection of order with id=1
        order_execution = {
            'id': 1,
            'price': 12,
            'quantity': 100,
            'side': 'sell',
            'status': 'rejected'
        }
        self.trading_strategy.handle_message(order_execution)
        # Ensure the trading strategy removes the rejected order
        self.assertEqual(self.trading_strategy.orders[0]['price'], 11)
        self.assertEqual(self.trading_strategy.orders[0]['quantity'], 100)
        self.assertEqual(self.trading_strategy.orders[0]['side'], 'buy')
        self.assertEqual(self.trading_strategy.orders[0]['status'], 'new')

    def test_filled_order(self):
        # Get book event message and handle the event
        self.test_receive_top_of_book()
        # Create market response indicating filled order for order id=1
        order_execution = {
            'id': 1,
            'price': 12,
            'quantity': 100,
            'side': 'sell',
            'status': 'filled'
        }
        self.trading_strategy.handle_message(order_execution)
        self.assertEqual(len(self.trading_strategy.orders), 1)

        # Create market response indicating filled order for order id=2
        order_execution = {
            'id': 2,
            'price': 11,
            'quantity': 100,
            'side': 'buy',
            'status': 'filled'
        }
        self.trading_strategy.handle_message(order_execution)
        self.assertEqual(self.trading_strategy.position, 0)
        self.assertEqual(self.trading_strategy.cash, 10100)
        self.assertEqual(self.trading_strategy.pnl, 100)


if __name__ == '__main__':
    unittest.main()

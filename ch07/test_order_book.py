import unittest

from ch07.order_book import OrderBook


class TestOrderBook(unittest.TestCase):

    def setUp(self) -> None:
        self.order_book = OrderBook()

    def test_handle_create_new_order(self):
        """Verify the 'create' action creates and adds a new order."""
        # Verify create new buy orders
        order_1 = {
            'id': 1,
            'price': 219,
            'quantity': 10,
            'side': 'buy',
            'action': 'create'
        }

        # handle_order() method processes all actions
        book_event = self.order_book.process_order(order_1)
        self.assertEqual(book_event['bid_price'], 219)
        self.assertEqual(book_event['bid_quantity'], 10)
        self.assertEqual(book_event['offer_price'], -1)
        self.assertEqual(book_event['offer_quantity'], -1)
        order_2 = order_1.copy()
        order_2['id'] = 2
        order_2['price'] = 220
        book_event = self.order_book.process_order(order_2)
        self.assertEqual(book_event['bid_price'], 220)
        self.assertEqual(book_event['bid_quantity'], 10)
        self.assertEqual(book_event['offer_price'], -1)
        self.assertEqual(book_event['offer_quantity'], -1)
        order_3 = order_1.copy()
        order_3['id'] = 3
        order_3['price'] = 223
        book_event = self.order_book.process_order(order_3)
        self.assertEqual(book_event['bid_price'], 223)
        self.assertEqual(book_event['bid_quantity'], 10)
        self.assertEqual(book_event['offer_price'], -1)
        self.assertEqual(book_event['offer_quantity'], -1)
        # Verify create new sell orders
        order_4 = order_1.copy()
        order_4['id'] = 4
        order_4['price'] = 220
        order_4['side'] = 'sell'
        book_event = self.order_book.process_order(order_4)
        self.assertEqual(book_event['bid_price'], 223)
        self.assertEqual(book_event['bid_quantity'], 10)
        self.assertEqual(book_event['offer_price'], 220)
        self.assertEqual(book_event['offer_quantity'], 10)
        order_5 = order_4.copy()
        order_5['id'] = 5
        order_5['price'] = 223
        book_event = self.order_book.process_order(order_5)
        self.assertIsNone(book_event, 'Top of book has not changed.')
        order_6 = order_4.copy()
        order_6['id'] = 6
        order_6['price'] = 221
        book_event = self.order_book.process_order(order_6)
        self.assertIsNone(book_event, 'Top of book has not changed.')

        self.assertEqual(self.order_book.bid_list[0]['id'], 3)
        self.assertEqual(self.order_book.bid_list[1]['id'], 2)
        self.assertEqual(self.order_book.bid_list[2]['id'], 1)
        self.assertEqual(self.order_book.offer_list[0]['id'], 4)
        self.assertEqual(self.order_book.offer_list[1]['id'], 6)
        self.assertEqual(self.order_book.offer_list[2]['id'], 5)

    def test_handle_amend_order(self):
        """Verify 'amend' action updates an existing order."""
        # Populate order book
        self.test_handle_create_new_order()
        # Amend order id=1
        order_1 = {
            'id': 1,
            'quantity': 5,
            'action': 'amend'
        }
        book_event = self.order_book.process_order(order_1)
        self.assertIsNone(book_event, 'Top of book has not changed')
        self.assertEqual(self.order_book.bid_list[2]['id'], 1)
        self.assertEqual(self.order_book.bid_list[2]['quantity'], 5)

    def test_handle_cancel_order(self):
        """Verify 'cancel' action deletes an existing order."""
        # Populate order book
        self.test_handle_create_new_order()
        # Cancel order id=1
        order_1 = {
            'id': 1,
            'action': 'cancel'
        }
        self.assertEqual(len(self.order_book.bid_list), 3)
        book_event = self.order_book.process_order(order_1)
        self.assertIsNone(book_event, 'Top of book has not changed')
        self.assertEqual(len(self.order_book.bid_list), 2)

    def test_generate_top_of_book_event(self):
        """Validate top of book event messages."""
        # New buy order
        order_1 = {
            'id': 1,
            'price': 219,
            'quantity': 10,
            'side': 'buy',
            'action': 'create'
        }
        book_event = self.order_book.process_order(order_1)
        expected_book_event = {
            'bid_price': 219,
            'bid_quantity': 10,
            'offer_price': -1,
            'offer_quantity': -1
        }
        self.assertEqual(book_event, expected_book_event)
        # New sell order
        order_2 = order_1.copy()
        order_2['id'] = 2
        order_2['price'] = 220
        order_2['side'] = 'sell'
        book_event = self.order_book.process_order(order_2)
        expected_book_event = {
            'bid_price': 219,
            'bid_quantity': 10,
            'offer_price': 220,
            'offer_quantity': 10
        }
        self.assertEqual(book_event, expected_book_event)


if __name__ == '__main__':
    unittest.main()

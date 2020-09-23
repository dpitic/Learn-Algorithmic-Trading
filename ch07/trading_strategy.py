"""Trading strategy based on top of the book changes."""


class TradingStrategy:
    """This class is the trading strategy based on top of the book changes. It
    creates an order when the top of the book is crossed i.e. when there is
    a potential arbitrate situation. When the bid value is higher than the ask
    value, it can send an order to buy and sell at the same time and make money
    out of those two transactions. The class is divided into two parts:
      * Signal - Handles the trading signal. A signal will be triggered when the
                 top of the book is crossed.
      * Execution - Handles the execution of orders. It is responsible for
                    managing the order lifecycle.
    """

    def __init__(self, cash, ob_2_ts, ts_2_om, om_2_ts):
        self.orders = []
        self.order_id = 0
        self.position = 0
        self.pnl = 0
        self.cash = cash
        self.current_bid = 0
        self.current_offer = 0
        self.ob_2_ts = ob_2_ts
        self.ts_2_om = ts_2_om
        self.om_2_ts = om_2_ts

    def create_orders(self, book_event, quantity):
        """Create two orders simultaneously for an arbitrage situation."""
        self.order_id += 1
        order = {
            'id': self.order_id,
            'price': book_event['bid_price'],
            'quantity': quantity,
            'side': 'sell',
            'action': 'to_be_sent'
        }
        self.orders.append(order.copy())

        self.order_id += 1
        order = {
            'id': self.order_id,
            'price': book_event['offer_price'],
            'quantity': quantity,
            'side': 'buy',
            'action': 'to_be_sent'
        }
        self.orders.append(order.copy())

    def signal(self, book_event):
        """Return True if the bid price is higher than the ask price."""
        if book_event is not None:
            if book_event['bid_price'] > book_event['offer_price']:
                if book_event['bid_price'] > 0 and \
                        book_event['offer_price'] > 0:
                    return True
                else:
                    return False
        else:
            return False

    def execution(self):
        """Manage processing orders for the whole order lifecycle. When an order
        is created, its status is 'new'. Once the order has been sent to the
        market, the market will respond by acknowledging the order or reject
        the order. If the order is rejected, it is removed from the list of
        outstanding orders.
        """
        orders_to_be_removed = []
        for index, order in enumerate(self.orders):
            if order['action'] == 'to_be_sent':
                order['status'] = 'new'
                order['action'] = 'no_action'
                if self.ts_2_om is None:
                    print('Simulation mode')
                else:
                    self.ts_2_om.append(order.copy())
            if order['status'] == 'rejected':
                orders_to_be_removed.append(index)
            if order['status'] == 'filled':
                orders_to_be_removed.append(index)
                position = order['quantity'] if order['side'] == 'buy' \
                    else -order['quantity']
                self.position += position
                self.pnl -= position * order['price']
                self.cash -= position * order['price']
        for order_index in sorted(orders_to_be_removed, reverse=True):
            del self.orders[order_index]

    def handle_input_from_bb(self, book_event=None):
        """Check whether there are book events in the deque ob_2_ts."""
        if self.ob_2_ts is None:
            print('Simulation mode')
            self.handle_book_event(book_event)
        else:
            if len(self.ob_2_ts) > 0:
                self.handle_book_event(self.ob_2_ts.popleft())

    def handle_book_event(self, book_event):
        """Check whether there is a signal to send an order."""
        if book_event is not None:
            self.current_bid = book_event['bid_price']
            self.current_offer = book_event['offer_price']

        # Check whether there is a signal to send an order
        if self.signal(book_event):
            self.create_orders(book_event, min(book_event['bid_quantity'],
                                               book_event['offer_quantity']))
        self.execution()

    def get_order(self, order_id):
        """Return order and order index."""
        index = 0
        for order in self.orders:
            if order['id'] == order_id:
                return order, index
            index += 1
        return None, None

    def handle_response_from_om(self):
        """Collect information from the order manager (collect information from
        the market)."""
        if self.om_2_ts is not None:
            self.handle_market_response(self.om_2_ts.popleft())
        else:
            print('Simulation mode')

    def handle_market_response(self, order_execution):
        """Process order from order manager."""
        order, index = self.get_order(order_execution['id'])
        if order is None:
            print('Error: order not found')
            return
        order['status'] = order_execution['status']
        self.execution()

    def get_pnl(self):
        """Return profit and loss."""
        return self.pnl + \
               self.position * (self.current_bid + self.current_offer) / 2

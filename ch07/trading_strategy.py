"""Trading strategy based on top of the book changes."""


def signal(book_event):
    """Return True if the bid price is higher than the ask price.

    This method inspects a book event to determine whether a trading signal
    has been found. A trading signal occurs if the bid price is less than
    the offer price in the book event and the message is valid, i.e. both
    the bid price and the offer price are greater than zero. That way a
    profit can be made by buying a the lower bid price and selling at the
    higher offer price.
    :param book_event: Book event message from the exchange.
    :return: True if the bid price is higher than the ask price, otherwise
        False.
    """
    if book_event is not None:  # message validation
        # Check if profit can be made (i.e. trading signal)
        if book_event['bid_price'] > book_event['offer_price']:
            # Validate message from exchange (order book)
            if book_event['bid_price'] > 0 and \
                    book_event['offer_price'] > 0:
                return True
            else:
                return False
    else:
        return False


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

    def __init__(self, cash, ob_2_ts=None, ts_2_om=None, om_2_ts=None):
        """Initialise a TradingStrategy object.
        :param cash: Initial amount of money for this trading strategy.
        :param ob_2_ts: Order book to trading strategy message channel,
            default=None places the object in simulation mode.
        :param ts_2_om: Trading strategy to order manager message channel,
            default=Nome places the object in simulation mode.
        """
        self.orders = []
        self.order_id = 0
        self.position = 0
        self._pnl = 0
        self.cash = cash
        self.current_bid = 0
        self.current_offer = 0
        self.ob_2_ts = ob_2_ts
        self.ts_2_om = ts_2_om
        self.om_2_ts = om_2_ts

    def create_order(self, book_event, quantity):
        """Create two orders simultaneously for an arbitrage situation.

        This method creates a sell order and a buy order for the specified
        quantity. The sell price is the bid price in the book event, and the buy
        price is the offer price in the book event. The orders are added to the
        list of orders this trading strategy object manages.
        """
        self.order_id += 1
        sell_order = {
            'id': self.order_id,
            'price': book_event['bid_price'],
            'quantity': quantity,
            'side': 'sell',
            'action': 'to_be_sent'
        }
        self.orders.append(sell_order.copy())

        self.order_id += 1
        buy_order = {
            'id': self.order_id,
            'price': book_event['offer_price'],
            'quantity': quantity,
            'side': 'buy',
            'action': 'to_be_sent'
        }
        self.orders.append(buy_order.copy())

    def execution(self):
        """Manage processing orders for the whole order lifecycle. When an order
        is created, its status is 'new'. Once the order has been sent to the
        market, the market will respond by acknowledging the order or reject
        the order. If the order is rejected or filled, it is removed from the 
        list of outstanding orders.
        """
        orders_to_be_removed = []  # index of rejected orders
        for index, order in enumerate(self.orders):
            if order['action'] == 'to_be_sent':
                order['status'] = 'new'
                order['action'] = 'no_action'
                if self.ts_2_om is None:
                    print('Simulation mode')
                else:
                    # Send a copy of the order to the order manager
                    self.ts_2_om.append(order.copy())
            elif order['status'] == 'rejected':
                # Add order index to the rejection list to be removed
                orders_to_be_removed.append(index)
            elif order['status'] == 'filled':
                # Order is filled; end of order lifecycle
                orders_to_be_removed.append(index)
                position = order['quantity'] if order['side'] == 'buy' \
                    else -order['quantity']
                self.position += position
                self._pnl -= position * order['price']
                self.cash -= position * order['price']
        # Remove rejected and filled orders
        for order_index in sorted(orders_to_be_removed, reverse=True):
            del self.orders[order_index]

    def handle_input_from_ob(self, book_event=None):
        """Check whether there are book events in the deque ob_2_ts.

        This method handles input messages from the order book. If the message
        channel between the order book and the trading strategy is not 
        configured, a book event message can be passed in for testing purposes.
        :param book_event: Book event message from the order book, default=None
            in which case the trading strategy is operating in simulation mode.
        """
        if self.ob_2_ts is None:
            print('Simulation mode')
            self.handle_book_event(book_event)
        else:
            if len(self.ob_2_ts) > 0:
                self.handle_book_event(self.ob_2_ts.popleft())

    def handle_book_event(self, book_event):
        """Check whether there is a signal in the book event to send an order.

        This method implements the logic to detect trading signals from book 
        event messages obtained from the order book. If a trading signal is
        found, it creates an order based on the book event for the minimum
        quantity of bid quantity and offer quantity.
        :param book_event: Book event message from the order book.
        """
        if book_event is not None:
            self.current_bid = book_event['bid_price']
            self.current_offer = book_event['offer_price']

        # Check whether there is a signal to send an order
        if signal(book_event):
            self.create_order(book_event, min(book_event['bid_quantity'],
                                              book_event['offer_quantity']))
        self.execution()

    def get_order(self, order_id):
        """Return reference to the order and the order index.

        This method looks up the orders in the trading strategy and returns the
        order and its index, if found.
        :param order_id: Order id to look up in the order this trading strategy
            manages.
        :return: Tuple of order and index, if the order id is found, otherwise
            None.
        """
        for index, order in enumerate(self.orders):
            if order['id'] == order_id:
                return order, index
        return None, None

    def handle_response_from_om(self):
        """Collect information from the order manager (collect information from
        the market)."""
        if self.om_2_ts is not None:
            self.handle_market_response(self.om_2_ts.popleft())
        else:
            print('Simulation mode')

    def handle_market_response(self, order_execution):
        """Process order from order manager.

        This method implements the logic to handle order execution messages
        from the market (obtained through the order manager). It looks up the
        order corresponding to the order execution message order id, and updates
        its status to align it with the status in the order execution message.
        It then delegates to the execution module of the trading strategy to
        manage the order lifecycle. If the order is not found, the message is 
        dropped.
        :param order_execution: Order execution message obtained from the order
            manager.
        """
        # Look up order corresponding to the order id in the message
        order, index = self.get_order(order_execution['id'])
        # Check if there is a corresponding order
        if order is None:
            # Corresponding order not found; drop message
            print('Error: order not found')
        else:
            # Order found; update order status to status of message
            order['status'] = order_execution['status']
            # Process orders managed by this trading strategy
            self.execution()

    @property
    def pnl(self):
        """Return profit and loss.

        This method calculates the profit and loss based on the current profit
        and loss, position and the current bid and current offer.
        :return: Profit and loss.
        """
        return self._pnl + \
               self.position * (self.current_bid + self.current_offer) / 2

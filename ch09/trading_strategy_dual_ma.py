"""Trading strategy based on top of the book changes."""
from collections import deque


def average(lst):
    """Return the average of a list."""
    return sum(lst) / len(lst)


class TradingStrategyDualMA:
    """The dual moving average trading strategy places a buy order when the
    short moving average crosses the long moving average in an upward direction
    and places a sell order when the cross happens on the other side. The class
    is divided into two parts:
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
        :param om_2_ts: Order manager to trading strategy message channel,
            default=None places the object in simulation mode.
        """
        self.orders = []
        self.order_id = 0

        self.position = 0
        self._pnl = 0
        self.cash = cash

        self.paper_position = 0
        self.paper_pnl = 0
        self.paper_cash = cash

        self.current_bid = 0
        self.current_offer = 0
        self.ob_2_ts = ob_2_ts
        self.ts_2_om = ts_2_om
        self.om_2_ts = om_2_ts

        self.long_signal = False
        self.total = 0
        self.holdings = 0
        self.small_window = deque()
        self.large_window = deque()
        self.list_position = []
        self.list_cash = []
        self.list_holding = []
        self.list_total = []

        self.list_paper_position = []
        self.list_paper_cash = []
        self.list_paper_holdings = []
        self.list_paper_total = []

    def create_metrics(self, price_update, small_window_limit=50,
                       large_window_limit=100):
        """Return whether the price is tradable.

        This method calculates the long moving average and the short moving
        average. When the short window moving average is higher than the long
        window moving average, it generates a long signal.

        :param price_update: Equity price from ticker data.
        :param int small_window_limit: Short moving average window, default=50.
        :param int large_window_limit: Long moving average window, default=100.
        :return: True if the price update is tradable, otherwise False.
        """
        self.small_window.append(price_update)
        self.large_window.append(price_update)
        if len(self.small_window) > small_window_limit:
            self.small_window.popleft()
        if len(self.large_window) > large_window_limit:
            self.large_window.popleft()
        if len(self.small_window) == small_window_limit:
            if average(self.small_window) > average(self.large_window):
                self.long_signal = True
            else:
                self.long_signal = False
            return True
        return False

    def trade(self, book_event):
        """Buy, sell or hold.

        This method places orders. A buy order will be placed when there is a
        short position or no position. A sell order will be placed when there is
        a long position or no position. It keeps track of the position, the
        holdings, and the profit, along with the paper trading quantities.

        :param book_event: Price update event.
        """
        if self.long_signal and self.paper_position <= 0:
            self.create_order(book_event, book_event['bid_quantity'], 'buy')
            self.paper_position += book_event['bid_quantity']
            self.paper_cash -= book_event['bid_quantity'] * \
                               book_event['bid_price']
        elif self.paper_position > 0 and not self.long_signal:
            self.create_order(book_event, book_event['bid_quantity'], 'sell')
            self.paper_position -= book_event['bid_quantity']
            self.paper_cash -= \
                -book_event['bid_quantity'] * book_event['bid_price']

        self.paper_holdings = self.paper_position * book_event['bid_price']
        self.paper_total = self.paper_holdings + self.paper_cash
        print(f'Total={self.total}, holding={self.holdings}, cash={self.cash}')

        self.list_paper_position.append(self.paper_position)
        self.list_paper_cash.append(self.paper_cash)
        self.list_paper_holdings.append(self.paper_holdings)
        self.list_paper_total.append(self.paper_holdings + self.paper_cash)

        self.list_position.append(self.position)
        self.holdings = self.position * book_event['bid_price']
        self.list_holding.append(self.holdings)
        self.list_cash.append(self.cash)
        self.list_total.append(self.holdings + self.cash)

    def create_order(self, book_event, quantity, side):
        """Create two orders simultaneously for an arbitrage situation.

        This method creates a sell order and a buy order for the specified
        quantity. The sell price is the bid price in the book event, and the buy
        price is the offer price in the book event. The orders are added to the
        list of orders this trading strategy object manages.
        """
        self.order_id += 1
        order = {
            'id': self.order_id,
            'price': book_event['bid_price'],
            'quantity': quantity,
            'side': side,
            'action': 'to_be_sent'
        }
        self.orders.append(order.copy())

    def signal(self, book_event):
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
            if book_event['bid_quantity'] != -1 and \
                    book_event['offer_quantity'] != -1:
                self.create_metrics(book_event['bid_price'])
                self.trade(book_event)

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
            elif order['status'] == 'rejected' or \
                    order['status'] == 'cancelled':
                # Add order index to the list to be removed
                orders_to_be_removed.append(index)
            elif order['status'] == 'filled':
                # Order is filled; end of order lifecycle
                orders_to_be_removed.append(index)
                position = order['quantity'] if order['side'] == 'buy' \
                    else -order['quantity']
                self.position += position
                self.holdings = self.position * order['price']
                self._pnl -= position * order['price']
                self.cash -= position * order['price']
        # Remove rejected and filled orders
        for order_index in sorted(orders_to_be_removed, reverse=True):
            del self.orders[order_index]

    def handle_order_book_message(self, book_event=None):
        """Handle book event messages from the order book.

        This method handles book event messages from the order book by removing
        and processing a book event message if the message channel is configured
        and if there are any messages. If the message channel between the order
        book and the trading strategy is not configured, a book event message
        can be passed in for testing purposes.
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
        # if self.signal(book_event):
        #     self.create_order(book_event, min(book_event['bid_quantity'],
        #                                       book_event['offer_quantity']))
        self.signal(book_event)
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

    def handle_order_manager_message(self):
        """Collect information from the order manager (collect information from
        the market)."""
        if self.om_2_ts is None:
            print('Simulation mode')
        elif len(self.om_2_ts) > 0:
            self.handle_message(self.om_2_ts.popleft())

    def handle_message(self, order_execution):
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

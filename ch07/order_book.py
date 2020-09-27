"""Order book module."""


def create_book_event(bid, offer):
    """Return a book event consisting of the bid and offer prices & quantities.

    Either the bid or offer are None, the corresponding price and quantity are
    set to -1.
    :return: Book event containing bid and offer prices and quantities, or -1
        values if the bid or offer is None.
    """
    book_event = {
        'bid_price': bid['price'] if bid is not None else -1,
        'bid_quantity': bid['quantity'] if bid is not None else -1,
        'offer_price': offer['price'] if offer is not None else -1,
        'offer_quantity': offer['quantity'] if offer is not None else -1
    }
    return book_event


class OrderBook:
    """This class implements a limit order book. Its responsibility is to gather
    all of the orders and sort them in a way that facilitates the work of the
    trading strategy. The order book is used by exchanges to maintain sell and
    buy orders. Trading systems need to get the order book of the exchanges
    they trade on to know which prices are the best or to have a view on the
    market. Trading strategies need to make decisions very rapidly to buy, sell
    or hold stocks. Since the order book provides the required information to
    the trading strategies to make their decisions, it needs to be fast.

    An order book is a book for the orders coming from the buyers and a book for
    the orders from sellers. The highest bid and the lowest offer prices will
    have priority. In situations where there is more than one bid with the same
    price competing for the best price, the time stamp will be used to sort
    which one should be sold i.e. the bid order with the earliest timestamp will
    be executed first.

    The order book will provide the following operations to support the life
    cycle of the orders:
      * create: Create and insert an order into the order book. This operation
                needs to be fast so the algorithm and data structure chosen for
                this operation are critical, because the book of bids and offers
                needs to be sorted at all times.
      * amend:  Look up an existing order in the order book using the order id
                and amend the order with the updated information.
      * cancel: Remove an order from the order book using the order id.

    The order book collects orders from the liquidity provider (exchange) and
    sorts the orders and creates book events. The book events implemented in
    this trading system will be generated each time there is a change on the top
    of the book i.e. any changes in teh first level of the book will create a
    book event.
    """

    def __init__(self, gw_2_ob=None, ob_2_ts=None):
        self.bid_list = []  # buy orders
        self.offer_list = []  # sell orders
        self.gw_2_ob = gw_2_ob
        self.ob_2_ts = ob_2_ts
        self.current_bid = None
        self.current_offer = None

    def generate_top_of_book_event(self):
        """Return a book event message when the top of the book has changed.

        When the price or the quantity for the best bid or offer has changed, a
        book event will be generated, otherwise return None.
        :return: Book event when top of book has changed, otherwise None.
        """
        tob_changed = False
        # Process bid (buy) list
        current_bid_list = self.bid_list
        # Check if the bid top of book has changed
        if len(current_bid_list) == 0:
            # No bids in the order book; reset current bid TOB pointer
            if self.current_bid is not None:
                # Current bid TOB pointer is pointing to an old bid
                tob_changed = True
                # Reset (clear) current bid top of book pointer
                self.current_bid = None
        else:
            # Bids exist in the order book; check if pointer points to TOB
            if self.current_bid != current_bid_list[0]:
                # There is a new offer at the top of book; TOB has changed
                tob_changed = True
                # Point to new top of book bid
                self.current_bid = current_bid_list[0]

        # Process offer (sell) list
        current_offer_list = self.offer_list
        # Check if the offer top of book has changed
        if len(current_offer_list) == 0:
            # No offers in the order book; reset current offer TOB pointer
            if self.current_offer is not None:
                # Current offer TOB pointer is pointing to an old offer
                tob_changed = True
                # Reset (clear) current offer top of book pointer
                self.current_offer = None
        else:
            # Offers exist in the order book; check if pointer points to TOB
            if self.current_offer != current_offer_list[0]:
                # There is a new offer at the top of book; TOB has changed
                tob_changed = True
                # Point to new top of book offer
                self.current_offer = current_offer_list[0]

        # Create new book event if top of book has changed
        book_event = None
        if tob_changed:
            book_event = create_book_event(self.current_bid, self.current_offer)
            # Send book event message to trading strategy if gateway configured
            if self.ob_2_ts is not None:
                self.ob_2_ts.append(book_event)
        return book_event

    def handle_order_from_gateway(self, mock_order=None):
        """Receive order messages from the liquidity provider.

        This method removes order messages from the liquidity provider sent
        on the message channel (gateway) and delegates processing of the order.
        If the gateway from the liquidity provider is not configured, the object
        operates in simulation mode and handles the manual order passed in.
        :param mock_order: Mock order message for testing, default=None.
        """
        order = mock_order
        if self.gw_2_ob is None:
            print('Simulation mode')
        elif len(self.gw_2_ob) > 0:
            # Message channel contains an order, remove message from channel
            order = self.gw_2_ob.popleft()
        self.handle_order(order)

    def handle_order(self, order):
        """Return book event after processing orders from liquidity provider.

        This method delegates processing based on the 'action' specified in the
        order. If the top of book has changed, it returns the book event,
        otherwise None.
        :param order: Order message from the liquidity provider.
        :return: Book event message if top of book has changed, otherwise None.
        """
        if order['action'] == 'create':
            self.create_new_order(order)
        elif order['action'] == 'amend':
            self.amend_order(order)
        elif order['action'] == 'cancel':
            self.cancel_order(order)
        else:
            print(f'Error: order action={order["action"]} not supported. '
                  f'Dropping order message with order id={order["id"]}.')
        return self.generate_top_of_book_event()

    def create_new_order(self, new_order):
        """Create a new order in the order book and sort by price."""
        if new_order['side'] == 'buy':
            self.bid_list.append(new_order)
            # Best buy orders from exchange have highest prices
            self.bid_list.sort(key=lambda x: x['price'], reverse=True)
        elif new_order['side'] == 'sell':
            self.offer_list.append(new_order)
            # Best sell orders from exchange have lowest prices
            self.offer_list.sort(key=lambda x: x['price'])

    def get_order_list(self, order):
        order_list = None
        # Find appropriate list based on order side
        if 'side' in order:
            if order['side'] == 'buy':
                order_list = self.bid_list
            elif order['side'] == 'sell':
                order_list = self.offer_list
            else:
                print(f'Unknown order side={order["side"]}. Dropping order '
                      f'id={order["id"]}.')
        else:
            # Find appropriate list based on order id
            for buy_order in self.bid_list:
                if buy_order['id'] == order['id']:
                    return self.bid_list
            for sell_order in self.offer_list:
                if sell_order['id'] == order['id']:
                    return self.offer_list
            # List containing order not found
            print(f'Order id={order["id"]} not found in order book.')
        return order_list

    def get_order(self, order, order_book_list=None):
        """Return order from order book.
        :param order: Order to find the the order book list.
        :param order_book_list: Order book list, default=None.
        """
        if order_book_list is None:
            order_list = self.get_order_list(order)
        else:
            order_list = order_book_list

        if order_list is not None:
            for book_order in order_list:
                if book_order['id'] == order['id']:
                    return book_order
            # Order not found in order book
            print(f'Order not found order id={order["id"]}.')
        return None

    def amend_order(self, amended_order):
        """Modify an existing order in the order book."""
        book_order = self.get_order(amended_order)
        if book_order is not None:
            if book_order['quantity'] > amended_order['quantity']:
                book_order['quantity'] = amended_order['quantity']
            else:
                print('Cannot amend order to increase quantity.')
        else:
            print(f'Cannot amend order: Order id={amended_order["id"]} '
                  f'not found')

    def cancel_order(self, cancelled_order):
        """Cancel an existing order in the order book."""
        order_book_list = self.get_order_list(cancelled_order)
        book_order = self.get_order(cancelled_order, order_book_list)
        if order_book_list is not None and book_order is not None:
            order_book_list.remove(book_order)
        else:
            print(f'Unable to cancel order id={cancelled_order["id"]}.')

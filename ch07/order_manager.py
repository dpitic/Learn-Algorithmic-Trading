"""Order manager module."""


def validate_order(order):
    """Return True if the order is valid.

    Valid orders have both 'quantity' and 'price' greater than zero.
    :param dict order: Order to validate.
    :return: True for valid orders, otherwise False.
    """
    if order['quantity'] < 0 or order['price'] < 0:
        return False
    else:
        return True


class OrderManager:
    """
    The order manager is responsible for gathering the orders from all of the
    trading strategies and to communicate this with the market. In the process
    of performing this main function, it checks the validity of the orders and
    acts as a safeguard against mistakes introduced in the trading strategies.
    This component is the interface between the trading strategies and the
    market.
    """

    def __init__(self, ts_2_om=None, om_2_ts=None,
                 om_2_gw=None, gw_2_om=None) -> None:
        """Initialise the order manager object. If the message channels are
        not configured, the object will operate in simulation mode, which is
        designed for unit testing.
        :param ts_2_om: Message channel between the trading strategy and the 
            order manager, default=None and operate in simulation mode.
        :param om_2_ts: Message channel between order manager and trading
            strategy, default=None and operate in simulation mode.
        :param om_2_gw: Message channel between order manager and (market)
            gateway, default=None and operate in simulation mode.
        :param gw_2_om: Message channel between (market) gateway and order
            manager, default=None and operate in simulation mode.
        """
        self.orders = []
        self.order_id = 0
        self.ts_2_om = ts_2_om
        self.om_2_ts = om_2_ts
        self.om_2_gw = om_2_gw
        self.gw_2_om = gw_2_om

    def create_new_order(self, order):
        """Return new order based on the order information passed in.

        Create a new order object using the attributes of the order passed in.
        :param dict order: Order on which to base the newly created order.
        :return: New order.
        """
        self.order_id += 1
        new_order = {
            'id': self.order_id,
            'price': order['price'],
            'quantity': order['quantity'],
            'side': order['side'],
            'status': 'new',
            'action': 'create'
        }
        return new_order

    def handle_trading_strategy_message(self):
        """Handle message from the trading strategy.

        This method checks whether the message channel from the trading strategy
        has been configured, and if not, it drops the message and operates in
        simulation mode. If the channel has been configured and contains a
        message from the trading strategy, it removes the message from the from
        the channel and delegates processing the message to another method.
        """
        # Check message channel exists and an order message on the channel
        if self.ts_2_om is None:
            print('Simulation mode')
        elif len(self.ts_2_om) > 0:
            self.handle_trading_strategy_order(self.ts_2_om.popleft())

    def handle_trading_strategy_order(self, order):
        """Process the order from the trading strategy.

        After validating the order from the trading strategy, store the new
        order in the order manager. If the channel between the order manager and
        the market gateway is configured, send a copy of the order to the
        gateway, otherwise operate in simulation mode and drop the message.
        :param dict order: Order message from the trading strategy.
        """
        # Validate order from trading strategy
        if validate_order(order):
            # Create a new order based on the order received from the trading
            # strategy, and add a copy of the new order to the list of orders
            # managed by the order manager.
            new_order = self.create_new_order(order)
            self.orders.append(new_order.copy())
            # Validate the message channel from the order manager to the market
            # gateway
            if self.om_2_gw is None:
                # Message channel from order manager to market gateway not
                # configured; operate in simulation mode and drop order message
                print('Simulation mode')
            else:
                # Send a copy of the new order from the trading strategy to the
                # market through the market gateway
                self.om_2_gw.append(new_order.copy())

    def get_order(self, order_id):
        """
        Return the order specified by the order id, if it exists in the order
        manager, otherwise None.
        :param int order_id: Order identifier.
        :return: Order corresponding to the order id, otherwise None if that 
            order cannot be found in the order manager.
        """
        # Find the order in the order manager with the specified order id
        for index, order in enumerate(self.orders):
            if order['id'] == order_id:
                return order
        # Order not found
        return None

    def remove_filled_orders(self):
        """Remove filled orders.

        This method removes filled orders from the list of orders maintained by
        this order manager object.
        """
        filled_order_indexes = []
        # Find filled orders
        # TODO: What about 'rejected' orders?
        for index, order in enumerate(self.orders):
            if order['status'] == 'filled':
                filled_order_indexes.append(index)
        # Remove filled orders from order manager
        for order_index in sorted(filled_order_indexes, reverse=True):
            del self.orders[order_index]

    def handle_market_message(self):
        """
        Handle order update messages from the market (through the market 
        gateway) and delegate processing of the order update messages. If the
        message channel from the gateway (market) to the order manager is not
        configured, operate in simulation mode and drop the message.
        """
        if self.gw_2_om is None:
            print('Simulation mode')
        elif len(self.gw_2_om) > 0:
            self.handle_market_order(self.gw_2_om.popleft())

    def handle_market_order(self, order_update):
        """
        Handle order update messages from the market (through the market
        gateway) and send updated orders to the trading strategy. If the order
        cannot be found in the order manager, log a warning message and drop the
        message. If the message channel to the trading strategy is not
        configured, operate in simulation mode and drop the message. This method
        also removes filled orders.
        :param dict order_update: Order update message from the market, through
            the market gateway.
        """
        # Look up the updated order id in the order manager
        order = self.get_order(order_update['id'])
        if order is not None:
            # Update the order status with the order update message status
            order['status'] = order_update['status']
            # Send a copy of the order to the trading strategy
            if self.om_2_ts is not None:
                self.om_2_ts.append(order.copy())
            else:
                # Simulation mode; drop order update message
                print('Simulation mode')
            # Clean up filled orders in the order manager
            self.remove_filled_orders()
        else:
            # Warning: order in order update message not found in order manager
            print('Warning: order not found')

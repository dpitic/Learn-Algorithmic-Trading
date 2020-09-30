"""Market simulator module."""
import random


class MarketSimulator:
    """This class implements the market behaviour and exchange trading rules. It
    implements the market assumptions such as the order rejection rate and which
    types of orders can be accepted. It implements the trading rules of the
    exchange. The exchange communicates with the order manager using two message
    channels to send and receive messages.
    """

    def __init__(self, om_2_gw=None, gw_2_om=None):
        """Initialise a market simulator object.

        If the message channels are not configured, the object operates in
        simulation mode.
        :param om_2_gw: Message channel between order manager and market
            (gateway), default=None which operates in simulation mode.
        :param gw_2_om: Message channel between market (gateway) and order
            manager, default=None which operates in simulation mode.
        """
        self.orders = []  # market orders
        self.om_2_gw = om_2_gw
        self.gw_2_om = gw_2_om

    def get_order(self, order_id):
        """Return market order and index in list of orders for the order id.

        This method looks up the order with the specified order id and returns
        the order and its index in the list of orders managed by the market
        simulator if found, otherwise None.
        :param int order_id: Order id for the order to retrieve from the market
            simulator.
        :return tuple: Order and index of the order if found, otherwise None.
        """
        for index, order in enumerate(self.orders):
            if order['id'] == order_id:
                return order, index
        # Order not found
        return None, None

    def process_orders(self, ratio=100):
        """Exchange order processing logic and send orders to order manager.

        This method implements the order processing logic of the exchange
        which either fills or cancels orders. If the message channel to the
        order manager gateway is configured, it sends a copy of the order to the
        order manager gateway, otherwise it operates in simulation mode and
        drops all 'filled' and 'cancelled' orders (all processed orders).
        :param int ratio: Order filling ratio, default=100.
        """
        orders_to_be_removed = []
        for index, order in enumerate(self.orders):
            if random.randrange(100) <= ratio:
                order['status'] = 'filled'
            else:
                order['status'] = 'cancelled'
            orders_to_be_removed.append(index)

            if self.gw_2_om is not None:
                self.gw_2_om.append(order.copy())
            else:
                print('Simulation mode')
        # Remove all processed orders (filled and cancelled orders)
        for order_index in sorted(orders_to_be_removed, reverse=True):
            del self.orders[order_index]

    def handle_order_manager_message(self):
        """Process order messages from the order manager (gateway).

        This method validates the message channel between the order manager and
        the market gateway and removes messages from the message channel if it
        contains any order messages. It delegates processing the order manager
        messages to another method.
        """
        # Check message channel is configured and contains a message
        if self.om_2_gw is None:
            print('Simulation mode')
        elif len(self.om_2_gw) > 0:
            # Extract message from order manager
            self.handle_order(self.om_2_gw.popleft())

    def handle_order(self, order_execution):
        """Process order manager order execution messages.

        This method processes order execution messages from the order manager
        and accepts any new orders ('action' == 'create'). If an order already 
        has the same order id, the order execution message will be dropped. If 
        the order manager cancels or amends an order, the order is automatically
        cancelled or amended.
        :param order_execution: Order execution message from the order manager
            (gateway).
        """
        # Get market order for the order execution message, if the order exits
        order, index = self.get_order(order_execution['id'])
        if order is None:
            # Market order does not exist (new order execution)
            if order_execution['action'] == 'create':
                # Accept all new orders from the order manager
                order_execution['status'] = 'accepted'
                self.orders.append(order_execution)

                # Acknowledge order has been accepted
                if self.gw_2_om is not None:
                    self.gw_2_om.append(order_execution.copy())
                    # TODO: this will never get called in simulation mode
                    self.process_orders()
                else:
                    print('Simulation mode')
                # return
            elif order_execution['action'] == 'cancel' or \
                    order_execution['action'] == 'amend':
                # Market order not found. Drop message
                print(f'Order id={order_execution["id"]} not found. '
                      f'Order rejected by market.')
                # Advise order manager the order was rejected
                # TODO: confirm the status should be set to 'rejected'
                order_execution['status'] = 'rejected'
                if self.gw_2_om is not None:
                    self.gw_2_om.append(order_execution.copy())
                else:
                    print('Simulation mode')
                # return
        elif order is not None:
            # Found market order
            if order_execution['action'] == 'create':
                # Duplicate order execution message. Reject order & drop message
                print(f'Duplicate order id={order_execution["id"]}. '
                      f'Order rejected by market.')
                # return
            elif order_execution['action'] == 'cancel':
                # Process cancelled order execution message from order manager
                order['status'] = 'cancelled'
                # Acknowledge order manager the order has been cancelled
                if self.gw_2_om is not None:
                    self.gw_2_om.append(order.copy())
                else:
                    print('Simulation mode')
                # Remove orders cancelled by order manager from market simulator
                print(
                    f'Order id={order["id"]} cancelled & removed from market')
                del self.orders[index]
            elif order_execution['action'] == 'amend':
                # Process order amendment order execution message from OM
                order['status'] = 'accepted'
                # Acknowledge order manager the order amendment was accepted
                if self.gw_2_om is not None:
                    self.gw_2_om.append(order.copy())
                else:
                    print('Simulation mode')
                print(f'Order id={order["id"]} amended on the market')

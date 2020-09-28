"""Liquidity provider is the exchange."""
import random


class LiquidityProvider:
    """The Liquidity Provider is the Exchange.

    The goal of this component is to generate liquidities (orders). This class
    randomly generates liquidities and sends price updates to the trading
    system.
    """

    def __init__(self, gateway=None, random_seed=None):
        """Create new LiquidityProvider object.
        :param gateway: Reference to the gateway object that provides a message
            channel between the liquidity provider and the order book,
            default=None.
        :param random_seed: Random generator seed, default=None.
        """
        self._orders = []
        self._order_id = 0
        if random_seed is not None:
            random.seed(random_seed)
        self._gateway = gateway

    @property
    def orders(self):
        """Return list of orders the liquidity provider (exchange) manages."""
        return self._orders

    @property
    def order_id(self):
        """Return current order ID."""
        return self._order_id

    @property
    def gateway(self):
        """Return the configured gateway for this liquidity provider."""
        return self._gateway

    def get_order(self, order_id):
        """Return the order and index for the specified order id, if it exists.
        :param order_id: Order id to retrieve.
        :return: Tuple of order (dictionary) and index in the list of orders if
            order is found, otherwise None.
        """
        for index, order in enumerate(self.orders):
            if order['id'] == order_id:
                return order, index
        # Order not found
        return None, None

    def send_manual_order(self, order):
        """Return manually sent order to liquidity provider.

        If the liquidity provider is not configured with a gateway, then it is
        operating in simulation mode and this method simply returns the order.
        If a gateway is configured, it sends a copy of the order to the gateway.
        This is a convenience function intended for testing to enable orders
        to be manually sent to the liquidity provider in simulation mode rather
        than the liquidity provider providing orders.
        :param order: Order (dictionary).
        :return: Order.
        """
        if self.gateway is None:
            print('Simulation mode')
        else:
            self.gateway.append(order.copy())
        return order

    def read_tick_data_from_data_source(self):
        self.generate_random_order()

    def generate_random_order(self):
        """Return randomly generated order or amended existing order.

        The liquidity provider generates random orders. If a gateway is not
        configured, the liquidity provider is operating in simulation mode and
        simply returns the random order it generated, otherwise it appends a
        copy of the random order to the gateway. An order is a dictionary data
        structure defined as:
        order = {
            'id': // unique identifier of the order
            'price': // price of the order
            'quantity': // quantity of the order
            'side': // side or type of order: buy or sell
            'action': // order action: create, update, or delete
        }
        :return: New random order or amended order for update or delete action.
        """
        order_id = random.randrange(self.order_id + 1)
        price = random.randrange(8, 12)
        quantity = random.randrange(1, 10) * 100
        side = random.sample(['buy', 'sell'], 1)[0]

        # Check if the order exists
        order, order_index = self.get_order(order_id)

        new_order = False
        if order is None:
            new_order = True
            action = 'create'
        else:
            action = random.sample(['amend', 'cancel'], 1)[0]

        random_order = {
            'id': order_id,
            'price': price,
            'quantity': quantity,
            'side': side,
            'action': action
        }

        # if new_order:
        #     self._order_id += 1
        #     self.orders.append(random_order)
        #
        # if self.gateway is None:
        #     print('Simulation mode')
        # else:
        #     self.gateway.append(random_order.copy())
        # return random_order

        # Create new order
        if new_order:
            self._order_id += 1
            self.orders.append(random_order.copy())
            if self.gateway is not None:
                self.gateway.append(random_order.copy())
            else:
                print('Simulation mode')
            return random_order

        # Update existing order state for amend or cancel action
        if order_index is not None:
            if action == 'amend':
                self.orders[order_index] = random_order.copy()
                if self.gateway is not None:
                    self.gateway[order_index] = random_order.copy()
                else:
                    print('Simulation mode')
                return random_order
            elif action == 'cancel':
                self.orders[order_index]['action'] = action
                if self.gateway is not None:
                    self.gateway[order_index]['action'] = action
                else:
                    print('Simulation mode')
                return self.orders[order_index]

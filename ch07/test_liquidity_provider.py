import unittest

from ch07.liquidity_provider import LiquidityProvider


class TestLiquidityProvider(unittest.TestCase):
    def setUp(self) -> None:
        # Liquidity provider without gateway (simulation mode object)
        self.liquidity_provider = LiquidityProvider(random_seed=0)
        # Liquidity provider with gateway
        self.gateway = []
        self.liquidity_provider_with_gateway = LiquidityProvider(self.gateway,
                                                                 random_seed=0)

    def test_generate_random_order(self):
        # Test in simulation mode (no gateway)
        order = self.liquidity_provider.generate_random_order()
        print(order)
        self.assertIsNone(self.liquidity_provider.gateway)
        self.assertEqual(self.liquidity_provider.orders[0]['id'], 0)
        self.assertEqual(self.liquidity_provider.orders[0]['price'], 11)
        self.assertEqual(self.liquidity_provider.orders[0]['quantity'], 100)
        self.assertEqual(self.liquidity_provider.orders[0]['side'], 'sell')
        self.assertEqual(self.liquidity_provider.orders[0]['action'], 'create')

        order = self.liquidity_provider.generate_random_order()
        print(order)
        self.assertIsNone(self.liquidity_provider.gateway)
        self.assertEqual(self.liquidity_provider.orders[1]['id'], 1)
        self.assertEqual(self.liquidity_provider.orders[1]['price'], 11)
        self.assertEqual(self.liquidity_provider.orders[1]['quantity'], 500)
        self.assertEqual(self.liquidity_provider.orders[1]['side'], 'sell')
        self.assertEqual(self.liquidity_provider.orders[1]['action'], 'create')

        # Test orders in the gateway are the same as the liquidity provider
        print('Test gateway')
        for i in range(5):
            order = self.liquidity_provider_with_gateway.generate_random_order()
            print(order)
        self.assertEqual(self.gateway,
                         self.liquidity_provider_with_gateway.orders)

    def test_get_order(self):
        self.liquidity_provider.generate_random_order()
        self.assertIsNone(self.liquidity_provider.gateway)
        order, idx = self.liquidity_provider.get_order(0)
        print(order)
        self.assertEqual(idx, 0)
        self.assertEqual(order['id'], 0)
        self.assertEqual(order['price'], 11)
        self.assertEqual(order['quantity'], 100)
        self.assertEqual(order['side'], 'sell')
        self.assertEqual(order['action'], 'create')

    def test_orders(self):
        for i in range(5):
            order = self.liquidity_provider.generate_random_order()
            print(order)
        self.assertIsNone(self.liquidity_provider.gateway)
        orders = self.liquidity_provider.orders
        expected_orders = [
            {
                'id': 0,
                'price': 8,
                'quantity': 600,
                'side': 'sell',
                'action': 'update'
            },
            {
                'id': 1,
                'price': 11,
                'quantity': 500,
                'side': 'sell',
                'action': 'delete'
            }
        ]
        self.assertEqual(orders, expected_orders)

    def test_gateway(self):
        for i in range(5):
            order = self.liquidity_provider_with_gateway.generate_random_order()
            print(order)
        gateway = self.liquidity_provider_with_gateway.gateway
        self.assertEqual(gateway, self.gateway)

    def test_send_manual_order(self):
        # Test in simulation mode (no gateway)
        expected_order = {
            'id': 0,
            'price': 8,
            'quantity': 600,
            'side': 'sell',
            'action': 'update'
        }
        order = self.liquidity_provider.send_manual_order(expected_order)
        self.assertIsNone(self.liquidity_provider.gateway)
        self.assertEqual(order, expected_order)
        self.assertIsNone(self.liquidity_provider.gateway)

        # Test with gateway
        print('Test with gateway')
        order = self.liquidity_provider_with_gateway.send_manual_order(
            expected_order)
        self.assertEqual(order, expected_order)
        gateway_order = self.liquidity_provider_with_gateway.gateway[0]
        self.assertEqual(gateway_order, expected_order)


if __name__ == '__main__':
    unittest.main()

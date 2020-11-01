"""Backtest algorithms."""
from collections import deque

from ch07.liquidity_provider import LiquidityProvider
from ch07.market_simulator import MarketSimulator
from ch07.order_book import OrderBook
from ch07.order_manager import OrderManager
from ch09.trading_strategy_dual_ma import TradingStrategyDualMA


def average(lst):
    """Return the average of a list."""
    return sum(lst) / len(lst)


class ForLoopBackTester:
    """Naive trading backtester which reads price updates line by line and 
    calculates more metrics out of those prices (such as moving averages at
    the close). It then makes a decison on the trading direction. The profit
    and loss is calculated and displayed at the end of the backtester. The
    design is very simple and can quickly discern whether a trading idea is
    feasible."""

    def __init__(self, cash=10000) -> None:
        self.small_window = deque()
        self.large_window = deque()
        self.list_position = []
        self.list_cash = []
        self.list_holdings = []
        self.list_total = []

        self.long_signal = False
        self.position = 0
        self.cash = cash
        self.total = 0
        self.holdings = 0

    def create_metrics(self, price_update, small_window_limit=50,
                       large_window_limit=100):
        """Return whether the price is tradable.

        This method creates metrics out of prices and determines whether the
        price update point is tradable or not.
        """
        self.small_window.append(price_update['price'])
        self.large_window.append(price_update['price'])
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

    def trade(self, price_update, num_shares=10):
        """Buy, sell or hold."""
        if self.long_signal and self.position <= 0:
            print(f'{price_update["date"]} send buy order for {num_shares} '
                  f'shares price={price_update["price"]}')
            self.position += num_shares
            self.cash -= num_shares * price_update['price']
        elif self.position > 0 and not self.long_signal:
            print(f'{price_update["date"]} send sell order for {num_shares} '
                  f'shares price={price_update["price"]}')
            self.position -= num_shares
            self.cash -= num_shares * price_update['price']

        self.holdings = self.position * price_update['price']
        self.total = self.holdings + self.cash
        print(f'{price_update["date"]}: total={self.total}, '
              f'holding={self.holdings}, cash={self.cash}')

        self.list_position.append(self.position)
        self.list_cash.append(self.cash)
        self.list_holdings.append(self.holdings)
        self.list_total.append(self.holdings + self.cash)


def call_if_not_empty(deq, fun):
    while len(deq) > 0:
        fun()


class EventDrivenBackTester:
    """An event driven backtester uses almost all of the components of the
    trading system, which makes it more realistic. It uses a main loop which
    calls all of the components one by one. The components read the input one
    after the other and will then generate events if needed. All of the events
    are inserted into a queue. The events encountered include:
      *  Tick events - when a new line of market data is read.
      *  Book events - when the top of the book is modified.
      *  Signal events - when it is possible to go long or short.
      *  Order events - when orders are sent to the market.
      *  Market response events - when the market response comes to the
           trading system.
    """

    def __init__(self) -> None:
        # One-way message channel between liquidity provider and order book
        # through the liquidity provider gateway, for liquidity provider to send
        # messages to the order book (order book gw_2_ob).
        self.gw_2_ob = deque()
        # One-way message channel between order book and trading strategy for
        # order book to send messages to the trading strategy.
        self.ob_2_ts = deque()
        # Two-way message channel between trading strategy and order manager.
        # This channel is used by the trading strategy to send messages to the
        # order manager.
        self.ts_2_om = deque()
        # Two-way message channel between order manager and trading strategy.
        # This channel is used by the order manager to send messages to the
        # trading strategy.
        self.om_2_ts = deque()
        # Two-way message channel between order manager and market simulator
        # through the market gateway. This channel is used by the order manager
        # to send messages to the market (gateway).
        self.om_2_gw = deque()
        # Two-way message channel between market and order manager through the
        # market gateway. This channel is used by the market to send messages to
        # the order manager through the market gateway.
        self.gw_2_om = deque()

        # Trading system objects
        self.order_book = OrderBook(self.gw_2_ob, self.ob_2_ts)
        # Initialise trading strategy with $10,000
        self.trading_strategy = TradingStrategyDualMA(10000, self.ob_2_ts,
                                                      self.ts_2_om,
                                                      self.om_2_ts)
        self.order_manager = OrderManager(self.ts_2_om, self.om_2_ts,
                                          self.om_2_gw, self.gw_2_om)

        # External components
        self.liquidity_provider = LiquidityProvider(
            self.gw_2_ob, random_seed=0)
        self.market_simulator = MarketSimulator(self.om_2_gw, self.gw_2_om)

    def process_events(self):
        """Main event processing loop.

        This method processes the events in the system by calling all of the
        components in the trading system."""
        while len(self.gw_2_ob) > 0:
            call_if_not_empty(self.gw_2_ob,
                              self.order_book.handle_gateway_message)
            call_if_not_empty(self.ob_2_ts,
                              self.trading_strategy.handle_order_book_message)
            call_if_not_empty(self.ts_2_om,
                              self.order_manager.handle_trading_strategy_message)
            call_if_not_empty(self.om_2_gw,
                              self.market_simulator.handle_order_manager_message)
            call_if_not_empty(self.gw_2_om,
                              self.order_manager.handle_market_message)
            call_if_not_empty(self.om_2_ts,
                              self.trading_strategy.handle_order_manager_message)

    def process_price_data(self, price, quantity=1000):
        """Process price information from tick data.

        This method creates two orders, a buy order and a sell order, both with
        the same specified quantity. It sends both orders to the order book and
        then runs the main event processing loop. Once the loop is finished
        processing the orders, it cancels both orders and sends the order
        updates to the order book."""
        order_bid = {
            'id': 1,
            'price': price,
            'quantity': quantity,
            'side': 'buy',
            'action': 'create'
        }
        order_ask = {
            'id': 1,
            'price': price,
            'quantity': quantity,
            'side': 'sell',
            'action': 'create'
        }
        self.gw_2_ob.append(order_ask)
        self.gw_2_ob.append(order_bid)
        self.process_events()
        order_ask['action'] = 'cancel'
        order_bid['action'] = 'cancel'
        self.gw_2_ob.append(order_ask)
        self.gw_2_ob.append(order_bid)

"""Simulated clock used in backtesting."""
from datetime import datetime


class SimulatedClock:
    """Simulated clock used for backtesting. Every component in the trading
    system uses this class to obtain the time. The simulated time used for
    backtesting is the timestamp on the order, which is obtained by processing
    the order. If not operating in simulation mode, the class provides the
    real time.
    """

    def __init__(self, simulated=False):
        self.simulated = simulated
        self.simulated_time = None

    def process_order(self, order):
        """Obtain the timestamp from the order.

        This method processes the order to obtain the timestamp to use as the
        simulated clock time.
        """
        self.simulated_time = datetime.strptime(order['timestamp'],
                                                '%Y-%m-%d %H:%M:%S.%f')

    def get_time(self):
        """Return the time.

        If operating in simulation mode, return the timestamp from the order.
        If operating in real mode, return the real clock time.
        """
        if not self.simulated:
            return datetime.now()
        else:
            return self.simulated_time


def main():
    realtime = SimulatedClock()
    print(f'Real time = {realtime.get_time()}')
    simulated_time = SimulatedClock(simulated=True)
    simulated_time.process_order(
        {'id': 1,
         'timestamp': '2018-06-29 08:15:27.243860'
         }
    )
    print(f'Simulated time = {simulated_time.get_time()}')


if __name__ == '__main__':
    main()

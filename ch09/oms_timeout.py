"""Order management system with timeout."""
import threading
from datetime import timedelta
from time import sleep

from simulated_clock import SimulatedClock


class TimeOut(threading.Thread):
    """A timer class that runs on a separate thread and invokes a callback after
    the specified time interval has elapsed.
    """

    def __init__(self, sim_clock, time_to_stop, fun):
        super().__init__()
        self.time_to_stop = time_to_stop
        self.sim_clock = sim_clock
        self.callback = fun
        self.disabled = False

    def run(self):
        """Run the timer for the configured interval and invoke the callback."""
        while not self.disabled and \
                self.sim_clock.get_time() < self.time_to_stop:
            sleep(1)
        if not self.disabled:
            self.callback()


class OMS:
    """Order manager system class with timeout. This class only implements a
    subset of an order manager class just to demonstrate the order manager
    sending an order with timeout.
    """

    def __init__(self, sim_clock):
        self.sim_clock = sim_clock
        self.five_sec_order_timeout_management = \
            TimeOut(sim_clock, sim_clock.get_time() + timedelta(0, 5),
                    self.on_timeout)

    def send_order(self):
        """Send order with timeout handling."""
        self.five_sec_order_timeout_management.disabled = False
        self.five_sec_order_timeout_management.start()
        print('send order')

    def handle_market_message(self):
        """Handle order update message from market."""
        self.five_sec_order_timeout_management.disabled = True

    def on_timeout(self):
        """Callback function on timeout when order not acknowledged."""
        print('Order Timeout Please Take Action')


def main():
    print('Case 1: real time')
    sim_clock = SimulatedClock()
    oms = OMS(sim_clock)
    oms.send_order()
    for i in range(10):
        print(f'Do something else: {i}')
        sleep(1)

    print('\nCase 2: simulated clock')
    sim_clock = SimulatedClock(simulated=True)
    sim_clock.process_order(
        {'id': 1,
         'timestamp': '2018-06-29 08:15:27.243860'}
    )
    oms = OMS(sim_clock)
    oms.send_order()
    sim_clock.process_order(
        {'id': 1,
         'timestamp': '2018-06-29 08:21:27.243860'}
    )


if __name__ == '__main__':
    main()

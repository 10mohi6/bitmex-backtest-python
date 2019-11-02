from bitmex_backtest import Backtest
import pytest
import time


@pytest.fixture(scope="module", autouse=True)
def scope_module():
    yield Backtest(test=True)


@pytest.fixture(scope="function", autouse=True)
def bt(scope_module):
    time.sleep(1)
    yield scope_module


@pytest.mark.parametrize(
    "resolution,exp",
    [
        ("1", 1 * 60),
        ("5", 5 * 60),
        ("60", 60 * 60),
        ("1D", 1 * 60 * 60 * 24),
        ("1W", 1 * 60 * 60 * 24 * 7),
        ("1M", 1 * 60 * 60 * 24 * 7 * 4),
    ],
)
def test_resolution_to_seconds(bt, resolution, exp):
    actual = bt._resolution_to_seconds(resolution)
    expected = exp
    assert expected == actual


def test_candles(bt):
    actual = bt.candles("XBTUSD")
    expected = 500
    assert expected == len(actual)


def test_candles_count(bt):
    params = {"resolution": "5", "count": 1000}
    actual = bt.candles("XBTUSD", params)
    expected = 1000
    assert expected == len(actual)


def test_run_basic(bt):
    bt.candles("XBTUSD")
    fast_ma = bt.sma(period=5)
    slow_ma = bt.sma(period=25)
    bt.sell_exit = bt.buy_entry = (fast_ma > slow_ma) & (
        fast_ma.shift() <= slow_ma.shift()
    )
    bt.buy_exit = bt.sell_entry = (fast_ma < slow_ma) & (
        fast_ma.shift() >= slow_ma.shift()
    )
    actual = bt.run()
    expected = 11
    assert expected == len(actual)


def test_run_advanced(bt):
    filepath = "xbtusd-60.csv"
    if bt.exists(filepath):
        bt.read_csv(filepath)
    else:
        bt.candles("XBTUSD", {"resolution": "60", "count": "5000"})
        bt.to_csv(filepath)

    fast_ma = bt.sma(period=10)
    slow_ma = bt.sma(period=30)
    exit_ma = bt.sma(period=5)
    bt.buy_entry = (fast_ma > slow_ma) & (fast_ma.shift() <= slow_ma.shift())
    bt.sell_entry = (fast_ma < slow_ma) & (fast_ma.shift() >= slow_ma.shift())
    bt.buy_exit = (bt.C < exit_ma) & (bt.C.shift() >= exit_ma.shift())
    bt.sell_exit = (bt.C > exit_ma) & (bt.C.shift() <= exit_ma.shift())

    bt.quantity = 100
    bt.stop_loss = 200
    bt.take_profit = 1000

    actual = bt.run()
    expected = 11
    assert expected == len(actual)

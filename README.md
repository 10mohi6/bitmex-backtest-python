# bitmex-backtest

[![PyPI](https://img.shields.io/pypi/v/bitmex-backtest)](https://pypi.org/project/bitmex-backtest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/10mohi6/bitmex-backtest-python/branch/master/graph/badge.svg)](https://codecov.io/gh/10mohi6/bitmex-backtest-python)
[![Build Status](https://travis-ci.com/10mohi6/bitmex-backtest-python.svg?branch=master)](https://travis-ci.com/10mohi6/bitmex-backtest-python)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/bitmex-backtest)](https://pypi.org/project/bitmex-backtest/)
[![Downloads](https://pepy.tech/badge/bitmex-backtest)](https://pepy.tech/project/bitmex-backtest)

bitmex-backtest is a python library for backtest with bitmex fx trade rest api on Python 3.6 and above.


## Installation

    $ pip install bitmex-backtest

## Usage

### basic
```python
from bitmex_backtest import Backtest

bt = Backtest()
bt.candles("XBTUSD")
fast_ma = bt.sma(period=5)
slow_ma = bt.sma(period=25)
bt.sell_exit = bt.buy_entry = (fast_ma > slow_ma) & (fast_ma.shift() <= slow_ma.shift())
bt.buy_exit = bt.sell_entry = (fast_ma < slow_ma) & (fast_ma.shift() >= slow_ma.shift())
bt.run()
bt.plot()
```

### advanced
```python
from bitmex_backtest import Backtest

bt = Backtest(test=True)
filepath = "xbtusd-60.csv"
if bt.exists(filepath):
    bt.read_csv(filepath)
else:
    params = {
        "resolution": "60",  # 1 hour candlesticks (default=1) 1,3,5,15,30,60,120,180,240,360,720,1D,3D,1W,2W,1M
        "count": "5000" # 5000 candlesticks (default=500)
    }
    bt.candles("XBTUSD", params)
    bt.to_csv(filepath)

fast_ma = bt.sma(period=10)
slow_ma = bt.sma(period=30)
exit_ma = bt.sma(period=5)
bt.buy_entry = (fast_ma > slow_ma) & (fast_ma.shift() <= slow_ma.shift())
bt.sell_entry = (fast_ma < slow_ma) & (fast_ma.shift() >= slow_ma.shift())
bt.buy_exit = (bt.C < exit_ma) & (bt.C.shift() >= exit_ma.shift())
bt.sell_exit = (bt.C > exit_ma) & (bt.C.shift() <= exit_ma.shift())

bt.quantity = 100 # default=1
bt.stop_loss = 200 # stop loss (default=0)
bt.take_profit = 1000 # take profit (default=0)

print(bt.run())
bt.plot("backtest.png")
```

```python
total profit       -342200.000
total trades           162.000
win rate                32.716
profit factor            0.592
maximum drawdown    470950.000
recovery factor         -0.727
riskreward ratio         1.295
sharpe ratio            -0.127
average return         -20.325
stop loss               23.000
take profit              1.000
```
![advanced.png](https://raw.githubusercontent.com/10mohi6/bitmex-backtest-python/master/tests/advanced.png)


## Supported indicators
- Simple Moving Average 'sma'
- Exponential Moving Average 'ema'
- Moving Average Convergence Divergence 'macd'
- Relative Strenght Index 'rsi'
- Bollinger Bands 'bband'
- Stochastic Oscillator 'stoch'
- Market Momentum 'mom'


## Getting started

For help getting started with bitmex REST API, view our online [documentation](https://www.bitmex.com/app/restAPI).


## Contributing

1. Fork it
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create new Pull Request
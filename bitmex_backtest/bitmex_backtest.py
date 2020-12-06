import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Tuple, List, Any
from datetime import datetime, timezone


class Backtest(object):
    def __init__(self, *, test: bool = True) -> None:
        if test:
            self._base_url = "https://testnet.bitmex.com"
        else:
            self._base_url = "https://www.bitmex.com"
        self._quantity = 1
        self._take_profit = 0
        self._stop_loss = 0
        self._initial_deposit = 0

    def _resolution_to_seconds(self, resolution: str) -> int:
        if "D" in resolution:
            return int(resolution[0]) * 60 * 60 * 24
        elif "W" in resolution:
            return int(resolution[0]) * 60 * 60 * 24 * 7
        elif "M" in resolution:
            return int(resolution[0]) * 60 * 60 * 24 * 7 * 4
        else:
            return int(resolution) * 60

    def candles(self, symbol: str, params: Any = {}) -> pd.DataFrame:
        url = "{}/api/udf/history".format(self._base_url)
        params["symbol"] = symbol
        if "resolution" not in params:
            params["resolution"] = "1"
        _count = 499
        if "count" in params:
            _count = int(params["count"]) - 1
        if "to" not in params:
            params["to"] = datetime.now(timezone.utc).timestamp()
        if "from" not in params:
            params["from"] = (
                params["to"]
                - self._resolution_to_seconds(params["resolution"]) * _count
            )
        r = requests.get(url, params=params).json()
        self.df = pd.DataFrame.from_dict(
            {
                "T": pd.to_datetime(r["t"], unit="s"),
                "O": r["o"],
                "H": r["h"],
                "L": r["l"],
                "C": r["c"],
                "V": r["v"],
            },
        ).set_index("T")
        return self.df

    def exists(self, filepath: str) -> bool:
        return os.path.exists(filepath)

    def to_csv(self, filepath: str) -> str:
        return self.df.to_csv(filepath)

    def read_csv(self, filepath: str) -> pd.DataFrame:
        self.df = pd.read_csv(
            filepath, index_col=0, parse_dates=True, infer_datetime_format=True
        )
        return self.df

    def sma(self, *, period: int, price: str = "C") -> pd.DataFrame:
        return self.df[price].rolling(period).mean()

    def ema(self, *, period: int, price: str = "C") -> pd.DataFrame:
        return self.df[price].ewm(span=period).mean()

    def bband(
        self, *, period: int = 20, band: int = 2, price: str = "C"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        std = self.df[price].rolling(period).std()
        mean = self.df[price].rolling(period).mean()
        return mean - (std * band), mean + (std * band)

    def macd(
        self,
        *,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        price: str = "C"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        macd = (
            self.df[price].ewm(span=fast_period).mean()
            - self.df[price].ewm(span=slow_period).mean()
        )
        signal = macd.ewm(span=signal_period).mean()
        return macd, signal

    def stoch(
        self, *, k_period: int = 5, d_period: int = 3
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        k = (
            (self.df.C - self.df.L.rolling(k_period).min())
            / (self.df.H.rolling(k_period).max() - self.df.L.rolling(k_period).min())
            * 100
        )
        d = k.rolling(d_period).mean()
        return k, d

    def mom(self, *, period: int = 10, price: str = "C") -> pd.DataFrame:
        return self.df[price].diff(period)

    def rsi(self, *, period: int = 14, price: str = "C") -> pd.DataFrame:
        return 100 - 100 / (
            1
            - self.df[price].diff().clip(lower=0).rolling(period).mean()
            / self.df[price].diff().clip(upper=0).rolling(period).mean()
        )

    @property
    def buy_entry(self) -> List[bool]:
        return self._buy_entry

    @buy_entry.setter
    def buy_entry(self, buy_entry: pd.DataFrame) -> None:
        self._buy_entry = buy_entry.values

    @property
    def sell_entry(self) -> List[bool]:
        return self._sell_entry

    @sell_entry.setter
    def sell_entry(self, sell_entry: pd.DataFrame) -> None:
        self._sell_entry = sell_entry.values

    @property
    def buy_exit(self) -> List[bool]:
        return self._buy_exit

    @buy_exit.setter
    def buy_exit(self, buy_exit: pd.DataFrame) -> None:
        self._buy_exit = buy_exit.values

    @property
    def sell_exit(self) -> List[bool]:
        return self._sell_exit

    @sell_exit.setter
    def sell_exit(self, sell_exit: pd.DataFrame) -> None:
        self._sell_exit = sell_exit.values

    @property
    def quantity(self) -> int:
        return self._quantity

    @quantity.setter
    def quantity(self, quantity: int) -> None:
        self._quantity = quantity

    @property
    def take_profit(self) -> int:
        return self._take_profit

    @take_profit.setter
    def take_profit(self, take_profit: int) -> None:
        self._take_profit = take_profit

    @property
    def stop_loss(self) -> int:
        return self._stop_loss

    @stop_loss.setter
    def stop_loss(self, stop_loss: int) -> None:
        self._stop_loss = stop_loss

    @property
    def initial_deposit(self) -> int:
        return self._initial_deposit

    @initial_deposit.setter
    def initial_deposit(self, initial_deposit: int) -> None:
        self._initial_deposit = initial_deposit

    @property
    def C(self) -> pd.DataFrame:
        return self.df.C

    @property
    def O(self) -> pd.DataFrame:
        return self.df.O

    @property
    def H(self) -> pd.DataFrame:
        return self.df.H

    @property
    def L(self) -> pd.DataFrame:
        return self.df.L

    @property
    def V(self) -> pd.DataFrame:
        return self.df.V

    def run(self) -> pd.Series:
        o = self.df.O.values
        l = self.df.L.values
        h = self.df.H.values
        N = len(self.df)

        long_trade = np.zeros(N)
        short_trade = np.zeros(N)

        # buy entry
        buy_entry_s = np.hstack((False, self._buy_entry[:-1]))  # shift
        long_trade[buy_entry_s] = o[buy_entry_s]

        # buy exit
        buy_exit_s = np.hstack((False, self._buy_exit[:-2], True))  # shift
        long_trade[buy_exit_s] = -o[buy_exit_s]

        # sell entry
        sell_entry_s = np.hstack((False, self._sell_entry[:-1]))  # shift
        short_trade[sell_entry_s] = o[sell_entry_s]

        # sell exit
        sell_exit_s = np.hstack((False, self._sell_exit[:-2], True))  # shift
        short_trade[sell_exit_s] = -(o[sell_exit_s])

        long_pl = pd.Series(np.zeros(N))  # profit/loss of buy position
        short_pl = pd.Series(np.zeros(N))  # profit/loss of sell position
        buy_price = sell_price = 0
        long_rr = []  # long return rate
        short_rr = []  # short return rate
        stop_loss = take_profit = 0

        for i in range(1, N):
            # buy entry
            if long_trade[i] > 0:
                if buy_price == 0:
                    buy_price = long_trade[i]
                    short_trade[i] = -buy_price  # sell exit
                else:
                    long_trade[i] = 0

            # sell entry
            if short_trade[i] > 0:
                if sell_price == 0:
                    sell_price = short_trade[i]
                    long_trade[i] = -sell_price  # buy exit
                else:
                    short_trade[i] = 0

            # buy exit
            if long_trade[i] < 0:
                if buy_price != 0:
                    long_pl[i] = (
                        -(buy_price + long_trade[i]) * self._quantity
                    )  # profit/loss fixed
                    long_rr.append(
                        round(long_pl[i] / buy_price * 100, 2)
                    )  # long return rate
                    buy_price = 0
                else:
                    long_trade[i] = 0

            # sell exit
            if short_trade[i] < 0:
                if sell_price != 0:
                    short_pl[i] = (
                        sell_price + short_trade[i]
                    ) * self._quantity  # profit/loss fixed
                    short_rr.append(
                        round(short_pl[i] / sell_price * 100, 2)
                    )  # short return rate
                    sell_price = 0
                else:
                    short_trade[i] = 0

            # close buy position with stop loss
            if buy_price != 0 and self._stop_loss > 0:
                stop_price = buy_price - self._stop_loss
                if l[i] <= stop_price:
                    long_trade[i] = -stop_price
                    long_pl[i] = (
                        -(buy_price + long_trade[i]) * self._quantity
                    )  # profit/loss fixed
                    long_rr.append(
                        round(long_pl[i] / buy_price * 100, 2)
                    )  # long return rate
                    buy_price = 0
                    stop_loss += 1

            # close buy positon with take profit
            if buy_price != 0 and self._take_profit > 0:
                limit_price = buy_price + self._take_profit
                if h[i] >= limit_price:
                    long_trade[i] = -limit_price
                    long_pl[i] = (
                        -(buy_price + long_trade[i]) * self._quantity
                    )  # profit/loss fixed
                    long_rr.append(
                        round(long_pl[i] / buy_price * 100, 2)
                    )  # long return rate
                    buy_price = 0
                    take_profit += 1

            # close sell position with stop loss
            if sell_price != 0 and self._stop_loss > 0:
                stop_price = sell_price + self._stop_loss
                if h[i] >= stop_price:
                    short_trade[i] = -stop_price
                    short_pl[i] = (
                        sell_price + short_trade[i]
                    ) * self._quantity  # profit/loss fixed
                    short_rr.append(
                        round(short_pl[i] / sell_price * 100, 2)
                    )  # short return rate
                    sell_price = 0
                    stop_loss += 1

            # close sell position with take profit
            if sell_price != 0 and self._take_profit > 0:
                limit_price = sell_price - self._take_profit
                if l[i] <= limit_price:
                    short_trade[i] = -limit_price
                    short_pl[i] = (
                        sell_price + short_trade[i]
                    ) * self._quantity  # profit/loss fixed
                    short_rr.append(
                        round(short_pl[i] / sell_price * 100, 2)
                    )  # short return rate
                    sell_price = 0
                    take_profit += 1

        win_trades = np.count_nonzero(long_pl.clip(lower=0)) + np.count_nonzero(
            short_pl.clip(lower=0)
        )
        lose_trades = np.count_nonzero(long_pl.clip(upper=0)) + np.count_nonzero(
            short_pl.clip(upper=0)
        )
        trades = (np.count_nonzero(long_trade) // 2) + (
            np.count_nonzero(short_trade) // 2
        )
        gross_profit = long_pl.clip(lower=0).sum() + short_pl.clip(lower=0).sum()
        gross_loss = long_pl.clip(upper=0).sum() + short_pl.clip(upper=0).sum()
        profit_pl = gross_profit + gross_loss
        self.equity = (long_pl + short_pl).cumsum()
        mdd = (self.equity.cummax() - self.equity).max()
        self.return_rate = pd.Series(short_rr + long_rr)

        s = pd.Series(dtype="object")
        s.loc["total profit"] = round(profit_pl, 3)
        s.loc["total trades"] = trades
        s.loc["win rate"] = round(win_trades / trades * 100, 3)
        s.loc["profit factor"] = round(-gross_profit / gross_loss, 3)
        s.loc["maximum drawdown"] = round(mdd, 3)
        s.loc["recovery factor"] = round(profit_pl / mdd, 3)
        s.loc["riskreward ratio"] = round(
            -(gross_profit / win_trades) / (gross_loss / lose_trades), 3
        )
        s.loc["sharpe ratio"] = round(
            self.return_rate.mean() / self.return_rate.std(), 3
        )
        s.loc["average return"] = round(self.return_rate.mean(), 3)
        s.loc["stop loss"] = stop_loss
        s.loc["take profit"] = take_profit
        return s

    def plot(self, filepath: str = "") -> None:
        plt.subplot(2, 1, 1)
        plt.plot(self.equity + self._initial_deposit, label="equity")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.hist(self.return_rate, 50, rwidth=0.9)
        plt.axvline(
            sum(self.return_rate) / len(self.return_rate),
            color="orange",
            label="average return",
        )
        plt.legend()
        if filepath == "":
            plt.show()
        else:
            plt.savefig(filepath)

if __name__ == "__main__":
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

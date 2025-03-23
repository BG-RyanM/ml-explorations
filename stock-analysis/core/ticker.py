from typing import Optional, List, Union, Tuple
import asyncio
import pandas as pd
from pandas import DataFrame, read_pickle, DatetimeIndex
import yfinance as yf

from datetime import datetime, timedelta
from pytz import timezone
import logging

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


class Ticker(object):
    """
    Represents a stock ticker (e.g. GOOG) and its price data over a stretch of time.
    """

    def __init__(self, symbol: str, years: int = 10):
        self._symbol = symbol
        self._loaded = False
        self._data_frame: Optional[DataFrame] = None
        self._date_breaks: List[str] = []
        self._date_breaks_as_dt: List[datetime] = []

        current_dt = datetime.now()
        # HACK
        current_dt = current_dt - timedelta(days=1)
        est_tz = timezone("US/Eastern")
        est_dt = est_tz.localize(current_dt)
        self._current_date: datetime = est_dt
        self._start_date: datetime = est_dt - timedelta(days=(years * 365))

        self._lock = asyncio.Lock()

    async def load(self, force_web_scrape: bool = False):
        """
        Loads all the price data for the ticker, preferring data cached on disk over
        data obtained from the web. However, any price data not on disk will be grabbed
        from the web.
        :param force_web_scrape: if True, purge data cached on disk and grab it all from the web
        """
        if force_web_scrape:
            self._loaded = False
            self._data_frame = None
        if self._loaded:
            return

        async with self._lock:
            path = f"./data/price_{self._symbol}.zip"
            if not force_web_scrape:
                try:
                    self._data_frame = read_pickle(path)
                except FileNotFoundError:
                    pass

            def _modify_date(dt: datetime, inc: int = 0):
                # Return a date plus or minus the one specified by days specified.
                return dt + timedelta(days=inc)

            def _date_earlier_than(dt: datetime, dt_other: datetime):
                # Return True if dt is earlier than dt_other
                delta = (dt_other - dt).days
                return delta >= 0

            scrape_dates_before = None
            scrape_dates_after = None
            if self._data_frame is not None:
                # We loaded at least some data off disk
                _logger.info(f"got {self._symbol} from pickle")
                est_tz = timezone("US/Eastern")
                dates = self._data_frame.index
                first_dti: DatetimeIndex = dates[0]
                first_date = datetime(
                    year=first_dti.year, month=first_dti.month, day=first_dti.day
                )
                first_date = est_tz.localize(first_date)
                last_dti: DatetimeIndex = dates[-1]
                last_date = datetime(
                    year=last_dti.year, month=last_dti.month, day=last_dti.day
                )
                last_date = est_tz.localize(last_date)
                date_pair_before = (self._start_date, _modify_date(first_date, -1))
                if _date_earlier_than(date_pair_before[0], date_pair_before[1]):
                    scrape_dates_before = date_pair_before
                date_pair_after = (_modify_date(last_date, 1), self._current_date)
                if _date_earlier_than(date_pair_after[0], date_pair_after[1]):
                    scrape_dates_after = date_pair_after
            else:
                # No data was loaded from disk, must scrape everything
                scrape_dates_before = (self._start_date, self._current_date)

            def _get_scraped_dataframe(date_pair: Optional[tuple] = None):
                # performs web scraping for ticker symbol in specified data range, returns
                # pandas DataFrame
                if (date_pair[1] - date_pair[0]).days >= 0:
                    ticker_data = yf.Ticker(self._symbol)
                    start_date_str = (
                        f"{date_pair[0].year}-{date_pair[0].month}-{date_pair[0].day}"
                    )
                    end_date_str = (
                        f"{date_pair[1].year}-{date_pair[1].month}-{date_pair[1].day}"
                    )
                    _logger.info(
                        f"now scraping, start date = {start_date_str}, end date = {end_date_str}"
                    )
                    scraped_data_frame = ticker_data.history(
                        period="1d", start=start_date_str, end=end_date_str
                    )
                    _logger.info(
                        f"got {self._symbol} from scraping, start date = {start_date_str}, end date = {end_date_str}"
                    )
                    return scraped_data_frame
                return None

            # Do we need to scrape any data?
            scrape_before_df = None
            if scrape_dates_before is not None:
                scrape_before_df = _get_scraped_dataframe(scrape_dates_before)
            scrape_after_df = None
            if scrape_dates_after is not None:
                scrape_after_df = _get_scraped_dataframe(scrape_dates_after)

            # Incorporate scraped data into whatever data we'd gotten from disk
            if scrape_before_df is not None:
                # Was there disk data?
                if self._data_frame is not None:
                    self._data_frame = pd.concat([scrape_before_df, self._data_frame])
                else:
                    self._data_frame = scrape_before_df
            if scrape_after_df is not None:
                if self._data_frame is not None:
                    self._data_frame = pd.concat([self._data_frame, scrape_after_df])
                else:
                    self._data_frame = scrape_after_df

            # columns = self._data_frame.columns
            # print("columns are:", columns)
            # dates = self._data_frame.index

            self._loaded = True

        self._compute_date_breaks()

        if scrape_before_df is not None or scrape_before_df is not None:
            # We scraped some new data, so save the info out to disk
            await self.save()

    async def save(self):
        """Save ticker data to disk."""
        if not self._loaded:
            return
        # HACK: remove last two rows
        # dates = self._data_frame.index
        # self._data_frame.drop(labels=dates[-3:], inplace=True)

        async with self._lock:
            path = f"./data/price_{self._symbol}.zip"
            self._data_frame.to_pickle(path)

    @property
    def symbol(self) -> str:
        """
        Returns symbol name.
        """
        return self._symbol

    @property
    def dataframe(self) -> Optional[DataFrame]:
        """
        Returns Dataframe of ticker data. Each row contains Open, High, Low, Close, Volume, Dividends,
        Stock Splits.
        """
        return self._data_frame

    @property
    def date_breaks(self):
        return self._date_breaks

    @property
    def num_entries(self) -> int:
        return len(self._data_frame.index)

    def get_date(
        self, start_date: Optional[datetime] = None, days_offset: int = 0, as_str=False
    ) -> Union[datetime, str]:
        """
        Returns date some number of days earlier/later than specified date. If the found date
        is not a trading day, then return the first trading day that occurs after it.

        :param start_date: date to use. If not given, use today's date
        :param days_offset: if a negative number, step back in time; if positive, forward
        :param as_str: if True, return as human-readable string
        :return: requested date
        """
        ret_dt = (start_date if start_date else self._current_date) + timedelta(
            days=days_offset
        )
        ret_as_str = ret_dt.strftime("%Y-%m-%d")

        # We want the date to be one on which trading actually occurred
        valid_trading_day = False
        present_dts = [
            d.strftime("%Y-%m-%d") for d in pd.to_datetime(self._data_frame.index)
        ]
        original_ret_dt = ret_dt
        while not valid_trading_day:
            if ret_as_str in present_dts:
                valid_trading_day = True
            else:
                # Advance by one day
                ret_dt = ret_dt + timedelta(days=1)
                if ret_dt >= self._current_date:
                    # Just use the return dt we had before
                    valid_trading_day = True
                    ret_dt = original_ret_dt
                ret_as_str = ret_dt.strftime("%Y-%m-%d")

        # Do this to get rid of hours/minutes/seconds
        ret_dt = datetime.strptime(ret_as_str, "%Y-%m-%d")
        return ret_as_str if as_str else ret_dt

    def get_index_of_date(
        self, the_date: Union[datetime, str], forward_scan=False
    ) -> Optional[int]:
        """
        Given a date, get its index. If not a trading day, get index of nearest date that is.

        :param the_date: date
        :param forward_scan: if True, nearest trading day is the first that comes after specified
            date. If False, it's the nearest that comes BEFORE.
        :return: index or None, if can't be found
        """
        if isinstance(the_date, datetime):
            desired_dt = the_date
        else:
            desired_dt = datetime.strptime(the_date, "%Y-%m-%d")
        index = 0 if forward_scan else self.num_entries - 1
        while 0 <= index < self.num_entries:
            dt = self._data_frame.index[index]
            dt_as_str = dt.strftime("%Y-%m-%d")
            dt = datetime.strptime(dt_as_str, "%Y-%m-%d")
            if forward_scan and dt >= desired_dt:
                return index
            if not forward_scan and dt <= desired_dt:
                return index
            index += 1 if forward_scan else -1
        return None

    def get_date_of_index(self, index: int) -> Optional[str]:
        """
        Get date of entry at given index
        """
        return self._data_frame.index[index]

    def print_info(self):
        print(f"=================\nSymbol is: {self._symbol}")
        print("First five entries:\n--------")
        print(self._data_frame.head(5))
        print("Last five entries:\n--------")
        print(self._data_frame.tail(5))
        print("Most recent date breaks:\n--------")
        for i in range(3):
            print(self._date_breaks[-(i + 1)])

    def _compute_date_breaks(self):
        """
        The price data doesn't include every single calendar date (because of weekends and holidays),
        so we need to work out a list of what dates are missing.
        """
        df = self._data_frame
        # Removing all empty dates
        # Build complete timeline from start date to end date (every single day is present)
        all_dts = pd.date_range(start=df.index[0], end=df.index[-1])
        # retrieve the dates that ARE in the original dataset
        present_dts = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(df.index)]
        # define dates with missing values
        self._date_breaks = [
            d for d in all_dts.strftime("%Y-%m-%d").tolist() if not d in present_dts
        ]
        self._date_breaks_as_dt = [
            datetime.strptime(db, "%Y-%m-%d") for db in self._date_breaks
        ]


class TickerManager(object):
    """
    Manages Ticker objects
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if TickerManager._instance is None:
            TickerManager._instance = object.__new__(cls)
        return TickerManager._instance

    def __init__(self):
        if TickerManager._initialized:
            return
        else:
            TickerManager._initialized = True

        self._tickers = []  # list of Tickers
        self._ticker_map = {}  # map from ticker symbol (e.g. SPY) to Ticker

    async def load(self, symbol_list: List[str], years: Optional[int] = 10):
        """
        Loads ticker data into memory, creating Ticker objects.
        :param symbol_list: list of stock symbols
        :param years: number of years to cover for each ticker
        """
        for symbol in symbol_list:
            ticker = Ticker(symbol, years=years)
            await ticker.load()
            await asyncio.sleep(0.5)
            self._tickers.append(ticker)
            self._ticker_map[symbol] = ticker

    def get_ticker(self, symbol: str) -> Ticker:
        return self._ticker_map.get(symbol)

    def print_info(self):
        for ticker in self._tickers:
            ticker.print_info()

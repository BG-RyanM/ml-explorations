from typing import Optional, List, Union, Tuple, Dict, Any, Set
import asyncio
import pandas as pd
from pandas import DataFrame, read_pickle, DatetimeIndex

from core.ticker import Ticker
from core.indicator import Indicator, SMAIndicator, EMAIndicator, LongEntryIndicator


class Chart(object):

    """
    A Chart object contains the data needed to produce a chart and its various indicators (such as SMA),
    for graphical representation or for analysis by a deep learning system.

    An important feature of Chart is the ability to return adjusted data. That is, all price values
    (open, high, low, close, indicators with price values) are adjusted to be relative to a specific SMA.
    The new values are expressed as a multiple of that SMA on that day, with the SMA itself being changed
    to a value of 1.0. This makes it easier for a deep learning system to analyze a chart.

    A Chart also has a "safety" window. This is the part of a date range in which all signals are present
    and can be used. An easy example is a Simple Moving Average: the 200-day MA won't be generated until
    200 days from the start of data.
    """

    def __init__(self, ticker: Ticker):
        """
        Constructor
        :param ticker: a Ticker object that has been successfully loaded
        """
        self._ticker = ticker
        self._data_frame: Optional[DataFrame] = None

        # Maps indicator type, e.g. "SMA" to inner list, which contains Indicator objects
        self._indicators: Dict[str, List[Indicator]] = {}
        # Maps a specific indicator name, e.g. "SMA-30", to its Indicator
        self._indicators_by_name: Dict[str, Indicator] = {}

        self._left_safety_window = 0
        self._right_safety_window = 0

        self._build_local_dateframe()

    @property
    def symbol(self) -> str:
        """
        Returns symbol name.
        """
        return self._ticker.symbol

    @property
    def num_entries(self) -> int:
        return self._ticker.num_entries

    @property
    def dataframe(self) -> Optional[DataFrame]:
        return self._data_frame

    @property
    def ticker(self) -> Ticker:
        return self._ticker

    def get_indicator(self, type_name: str, parameters: Tuple) -> Optional[Indicator]:
        """
        Gets indicator that has been added to this chart, e.g. "SMA-30"
        :param type_name: type of indicator, e.g. "SMA"
        :param parameters: tuple of parameters, e.g. (30,)
        :return: name if indicator is present, otherwise None
        """
        if self._indicators.get(type_name) is None:
            return None
        for indicator in self._indicators[type_name]:
            if indicator.parameters == parameters:
                return indicator
        return None

    def get_indicator_by_name(self, name: str):
        """Get indicator by name, e.g. 'SMA-30'"""
        return self._indicators_by_name.get(name, None)

    def get_all_indicator_names(self) -> List[str]:
        """
        Returns the names of all indicators that have been added to the chart, e.g.
        "SMA-30", "EMA-20", etc.
        """
        out_list = []
        for type_name in self._indicators.keys():
            for indicator in self._indicators[type_name]:
                out_list.append(indicator.name)
        return out_list

    def get_indicators_by_chart_attachment(self, on_main_chart: bool):
        return [indicator for indicator in self._indicators_by_name.values() if indicator.on_main_chart == on_main_chart]

    def get_indicators(self):
        return [indicator for indicator in self._indicators_by_name.values()]

    def add_sma(self, period):
        """Helper function for adding SMA indicator"""
        self.add_indicator(SMAIndicator((period,)))

    def add_ema(self, period):
        """Helper function for adding EMA indicator"""
        self.add_indicator(EMAIndicator((period,)))

    def add_long_entry(self, target, max_loss, days_to_profit):
        """Helper function for adding LongEntry indicator"""
        self.add_indicator(LongEntryIndicator((target, max_loss, days_to_profit)))

    def get_plotable_dataframe(
        self, start_index: int, end_index: int, spacing: int = 1, adjust_to_sma: int = 0
    ) -> Tuple[pd.DataFrame, List[int], List[str]]:
        """
        Gets a dataframe that's convenient to plot with matplotlib. The indices of returned DF
        will just be numbers, rather than dates, which prevents weekends and holidays from showing
        up on the chart. Also generates labels to be plotted on x-axis.

        :param start_index: starting index within ticker's overall dataframe
        :param end_index: ending index within ticker's overall dataframe
        :param spacing: spacing between date labels on x-axis
        :param adjust_to_sma: see class-level docstring about adjustment
        :return: (plottable DataFrame, days for x ticks -- day 0 being leftmost, dates to put on x ticks)
        """

        date_labels = [
            d.strftime("%Y-%m-%d")
            for d in pd.to_datetime(self._data_frame.index[start_index:end_index])
        ]
        x_indices = [i for i in range(len(date_labels))]

        ret_df = self.get_ml_dataframe(start_index, end_index, adjust_to_sma)

        return ret_df, x_indices[::spacing], date_labels[::spacing]

    def get_ml_dataframe(self, start_index: int, end_index: int, adjust_to_sma: int):
        """
        Returns a dataframe that can be fed to PyTorch.

        :param start_index: starting index within ticker's overall dataframe
        :param end_index: ending index within ticker's overall dataframe
        :param adjust_to_sma: see class-level docstring about adjustment
        :return:
        """
        do_adjustment = self.get_indicator("SMA", (adjust_to_sma,)) is not None
        partial_df = self._data_frame.iloc[start_index:end_index]

        # Basically makes a copy of the local dataframe, but with the date indices replaced with sequential
        # integers.
        column_names = ["Open", "Close", "High", "Low", "Volume", "LongEntry"]
        for indicator_name in self.get_all_indicator_names():
            column_names.append(indicator_name)
        frame_dict = {}
        for col in column_names:
            series = partial_df[col]
            frame_dict[col] = list(series.array)

        if do_adjustment:
            self._adjust_frame_dict(frame_dict, adjust_to_sma)

        return pd.DataFrame(frame_dict)

    def get_usable_window_for_ml(self) -> Tuple[int, int]:
        """Returns left/right indices that can be used for ML, indicators considered"""
        return self._left_safety_window, self.num_entries - self._right_safety_window

    def add_indicator(
        self, indicator: Indicator
    ):
        """
        Adds an indicator to the chart.
        """
        indicator.compute(self._data_frame)

        if self._indicators.get(indicator.type_name) is None:
            self._indicators[indicator.type_name] = []

        if indicator.safety_window > self._left_safety_window:
            self._left_safety_window = indicator.safety_window

        # Is it already there? Add only if not.
        if self._indicators_by_name.get(indicator.name) is None:
            self._indicators[indicator.type_name].append(indicator)
            self._indicators_by_name[indicator.name] = indicator

    def _adjust_frame_dict(self, frame_dict: Dict, adjust_to_sma):
        """
        Given a frame dict, adjust contents of each column to be relative to a particular SMA.

        :param frame_dict: dict mapping column names to lists of values
        :param adjust_to_sma: SMA number, e.g. 30 for 30-day SMA
        :return:
        """
        adjust_to_sma_name = f"SMA-{adjust_to_sma}"

        adjust_column_names = ["Open", "Close", "High", "Low"]
        for indicator in self.get_indicators():
            if indicator.adjustable and indicator.name != adjust_to_sma_name:
                # print("**** adding indicator", indicator_name)
                adjust_column_names.append(indicator.name)

        adjustment_sma_list = frame_dict[adjust_to_sma_name]
        for col, col_data in frame_dict.items():
            if col in adjust_column_names:
                for i in range(len(col_data)):
                    col_data[i] = col_data[i] / adjustment_sma_list[i]
        for i in range(len(adjustment_sma_list)):
            adjustment_sma_list[i] = 1.0

    def _build_local_dateframe(self):
        """The chart has its own copy of the ticker's DataFrame. (More columns can be added.)"""
        self._data_frame = self._ticker.dataframe.copy(deep=True)


from typing import Optional, List, Union, Tuple, Dict, Any, Set
import asyncio
import pandas as pd
from pandas import DataFrame, read_pickle, DatetimeIndex

from core.ticker import Ticker


class Indicator(object):

    def __init__(self, type_name: str, parameters: Tuple, on_main_chart=True):
        """
        :param type_name: indicator type, e.g. "SMA"
        :param parameters: tuple of parameters for indicator, e.g. (30,)
        :param on_main_chart: True if indicator should be displayed on main chart
        """
        self._type_name = type_name
        self._parameters = parameters
        self._on_main_chart = on_main_chart

    @property
    def type_name(self):
        return self._type_name

    @property
    def parameters(self):
        return self._parameters

    @property
    def name(self):
        out_str = self._type_name
        for p in self._parameters:
            out_str += f"-{p}"
        return out_str


class Chart(object):

    """
    A Chart object contains the data needed to produce a chart and its various indicators (such as SMA),
    for graphical representation or for analysis by a deep learning system.

    An important feature of Chart is the ability to return adjusted data. That is, all price values
    (open, high, low, close, indicators with price values) are adjusted to be relative to a specific SMA.
    The new values are expressed as a multiple of that SMA on that day, with the SMA itself being changed
    to a value of 1.0. This makes it easier for a deep learning system to analyze a chart.
    """

    def __init__(self, ticker: Ticker):
        """
        Constructor
        :param ticker: a Ticker object that has been successfully loaded
        """
        self._ticker = ticker
        self._data_frame: Optional[DataFrame] = None

        # Maps indicator type, e.g. "SMA" to inner set, which contains Indicator objects
        self._indicators: Dict[str, Set[Indicator]] = {}

        self._build_local_dateframe()

    def _build_local_dateframe(self):
        """The chart has its own copy of the ticker's DataFrame. (More columns can be added.)"""
        self._data_frame = self._ticker.dataframe.copy(deep=True)

    @property
    def symbol(self) -> str:
        """
        Returns symbol name.
        """
        return self._ticker.symbol

    def _add_indicator(self, type_name: str, parameters: Tuple) -> str:
        """
        Adds an indicator to the chart (its values must next be added to local DataFrame by caller).

        :param type_name: name of indicator type, e.g. "SMA"
        :param parameters: tuple of parameters for this indicator
        :return: name of new indicator, e.g. "SMA-30"
        """
        if self._indicators.get(type_name) is None:
            self._indicators[type_name] = set()
        new_indicator = Indicator(type_name, parameters)
        self._indicators[type_name].add(new_indicator)
        return new_indicator.name

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

    def add_sma(self, window):
        """Adds a SMA indicator to chart."""
        sma_name = self._add_indicator("SMA", (window,))

        # Calculating simple moving average using .rolling(window).mean(),
        # with window size = 30
        self._data_frame[sma_name] = self._data_frame['Close'].rolling(window).mean()

        # Removing all the NULL values using dropna() method
        # self._data_frame.dropna(inplace=True)

    def get_plotable_dataframe(self, start_index: int, end_index: int, spacing: int = 1, adjust_to_sma: int = 0) -> \
            Tuple[pd.DataFrame, List[int], List[str]]:
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
        do_adjustment = self.get_indicator("SMA", (adjust_to_sma,)) is not None
        partial_df = self._data_frame.iloc[start_index:end_index]
        date_labels = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(self._data_frame.index[start_index:end_index])]
        x_indices = [i for i in range(len(date_labels))]

        # Basically makes a copy of the local dataframe, but with the date indices replaced with sequential
        # integers.
        column_names = ["Open", "Close", "High", "Low", "Volume"]
        for indicator_name in self.get_all_indicator_names():
            column_names.append(indicator_name)
        frame_dict = {}
        for col in column_names:
            series = partial_df[col]
            frame_dict[col] = list(series.array)

        if do_adjustment:
            self._adjust_frame_dict(frame_dict, adjust_to_sma)

        return pd.DataFrame(frame_dict), x_indices[::spacing], date_labels[::spacing]

    def _adjust_frame_dict(self, frame_dict: Dict, adjust_to_sma):
        adjust_to_sma_name = f"SMA-{adjust_to_sma}"

        adjust_column_names = ["Open", "Close", "High", "Low"]
        for indicator_name in self.get_all_indicator_names():
            if indicator_name != adjust_to_sma_name:
                #print("**** adding indicator", indicator_name)
                adjust_column_names.append(indicator_name)

        adjustment_sma_list = frame_dict[adjust_to_sma_name]
        for col, col_data in frame_dict.items():
            if col in adjust_column_names:
                for i in range(len(col_data)):
                    col_data[i] = col_data[i] / adjustment_sma_list[i]
        for i in range(len(adjustment_sma_list)):
            adjustment_sma_list[i] = 1.0



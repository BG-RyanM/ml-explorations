from typing import Dict
import matplotlib.pyplot as plt
from pandas import DataFrame

from core.chart import Chart, Indicator


class MPLHelper(object):
    """
    For plotting chart data using matplotlib. Operates on data taken from a Chart object and its
    indicators.
    """

    def __init__(self, chart: Chart):
        """All indicators must have been added to Chart by this point."""
        self._chart: Chart = chart

        self._indicator_colors = ["purple", "orange", "green", "blue", "red", "pink"]
        self._indicator_types = ["SMA", "EMA", "LongEntry"]
        # Maps indicator type to category
        self._indicator_categories: Dict[str, str] = {"SMA": "MA", "EMA": "MA"}
        # maps indicator category to next available color
        self._next_indicator_color: Dict[str, int] = {}
        categories = set(cat for cat in self._indicator_categories.values())
        for cat in categories:
            self._next_indicator_color[cat] = 0

    def show(self, start_index: int, end_index: int):
        self._show_impl(start_index, end_index, 0)

    def show_adjusted(self, start_index: int, end_index: int, adjust_to_sma: int):
        self._show_impl(start_index, end_index, adjust_to_sma)

    def _show_impl(self, start_index: int, end_index: int, adjust_to_sma: int):
        prices, x_indices, datelabels = self._chart.get_plotable_dataframe(
            start_index, end_index, spacing=14, adjust_to_sma=adjust_to_sma
        )
        # prices, x_indices, datelabels = my_chart.get_plotable_dataframe(-200, -1, spacing=14)

        extra_charts_indicators = self._chart.get_indicators_by_chart_attachment(False)
        # Create figure with multiple subplots, one for price data, one for volume, and all others for indicators
        # One row for price (and on-chart-indicators), one for volume, one for each "extra" indicator
        total_rows = 2 + len(extra_charts_indicators)
        height_ratios = [12] + [1 * len(extra_charts_indicators)] + [4]
        height_ratios = tuple(height_ratios)
        #   3, 1 = two rows, one column
        #   height_ratios=(12,1,4) means that upper subplot is 12 times taller than middle
        figure, axes_tups = plt.subplots(
            total_rows, 1, figsize=(12, 8), height_ratios=height_ratios
        )
        figure.suptitle(f"Data for {self._chart.symbol}")

        # Define width of candlestick elements, the candle and wick
        width_candle = 0.4
        width_wick = 0.05

        # Define up and down prices as two different DFs
        up = prices[prices.Close >= prices.Open]
        down = prices[prices.Close < prices.Open]

        # Define colors to use
        col1 = "green"
        col2 = "red"

        ########### Price subplot

        # Plot 'up' prices
        axes1 = axes_tups[0]
        axes1.bar(
            up.index,
            up.Close - up.Open,
            width_candle,
            bottom=up.Open,
            color=col1,
            label=self._chart.symbol,
        )
        axes1.bar(up.index, up.High - up.Close, width_wick, bottom=up.Close, color=col1)
        axes1.bar(up.index, up.Low - up.Open, width_wick, bottom=up.Open, color=col1)

        # Plot 'down' prices
        axes1.bar(
            down.index,
            down.Close - down.Open,
            width_candle,
            bottom=down.Open,
            color=col2,
        )
        axes1.bar(
            down.index, down.High - down.Open, width_wick, bottom=down.Open, color=col2
        )
        axes1.bar(
            down.index, down.Low - down.Close, width_wick, bottom=down.Close, color=col2
        )

        # Plot indicators that are on main chart

        main_chart_indicators = self._chart.get_indicators_by_chart_attachment(True)

        for type_name in self._indicator_types:
            self._next_indicator_color[type_name] = 0

        for indicator in main_chart_indicators:
            self._plot_indicator_main_chart(indicator.name, prices, axes1)

        # Put ticks but no labels at bottom
        axes1.set_xticks(x_indices, labels=["" for i in range(len(x_indices))])

        adjusted_str = " (adjusted)" if adjust_to_sma > 0 else ""
        axes1.set_title(f"Price Data for {self._chart.symbol}{adjusted_str}")
        axes1.legend()

        ########### Entry point subplot

        for idx, indicator in enumerate(extra_charts_indicators):
            axes_to_use = axes_tups[1 + idx]
            self._plot_indicator_extra(indicator.name, prices, x_indices, axes_to_use)


        ########### Volume subplot

        axes3 = axes_tups[-1]
        # Plot 'up' volume
        axes3.bar(up.index, up.Volume, width_candle, bottom=0, color=col1)

        # Plot 'down' volume
        axes3.bar(down.index, down.Volume, width_candle, bottom=0, color=col2)

        # Add x-axis ticks and labels with dates
        axes3.set_xticks(x_indices, labels=datelabels, rotation=45, ha="right")

        axes3.set_title(f"Volume for {self._chart.symbol}")

        plt.show()

    def _plot_indicator_main_chart(self, name: str, prices: DataFrame, axes):
        type_name = name.split("-")[0]
        if type_name not in self._indicator_types:
            return
        indicator_cat = self._indicator_categories[type_name]
        color_idx = self._next_indicator_color[indicator_cat]
        self._next_indicator_color[indicator_cat] = color_idx + 1
        color = self._indicator_colors[color_idx]

        if type_name in ["SMA", "EMA"]:
            sma_series = prices[name]
            sma_list = list(sma_series.array)
            axes.plot(sma_list, color=color, label=name)

    def _plot_indicator_extra(self, name: str, prices: DataFrame, x_indices, axes):
        type_name = name.split("-")[0]
        if type_name not in self._indicator_types:
            return

        width_candle = 0.4
        width_wick = 0.05

        entry_points = prices[prices.LongEntry > 0]
        non_entry_points = prices[prices.LongEntry <= 0]
        axes.bar(
            entry_points.index, entry_points.LongEntry, width_candle, bottom=0, color="blue"
        )
        axes.bar(non_entry_points.index, 0.0, width_candle, bottom=0, color="red")
        # Put ticks but no labels at bottom
        axes.set_xticks(x_indices, labels=["" for i in range(len(x_indices))])

        axes.set_title(f"Entry points for {self._chart.symbol}")

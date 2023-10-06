from typing import Dict
import matplotlib.pyplot as plt
from pandas import DataFrame

from core.chart import Chart, Indicator


class MPLHelper(object):
    def __init__(self, chart: Chart):
        """All indicators must have been added to Chart by this point."""
        self._chart: Chart = chart

        self._indicator_colors = ["purple", "orange", "green", "blue", "red", "pink"]
        self._indicator_types = ["SMA", "EMA"]
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
            -200, -1, spacing=14, adjust_to_sma=adjust_to_sma
        )
        # prices, x_indices, datelabels = my_chart.get_plotable_dataframe(-200, -1, spacing=14)

        # Create figure with three subplots, one for price data, one for entry points, and one for volume
        #   3, 1 = two rows, one column
        #   height_ratios=(12,1,4) means that upper subplot is 12 times taller than middle
        figure, (axes1, axes2, axes3) = plt.subplots(
            3, 1, figsize=(12, 8), height_ratios=(12, 1, 4)
        )
        figure.suptitle(f"Data for {self._chart.symbol}")

        # Define width of candlestick elements, the candle and wick
        width_candle = 0.4
        width_wick = 0.05

        # Define up and down prices as two different DFs
        up = prices[prices.Close >= prices.Open]
        down = prices[prices.Close < prices.Open]
        entry_points = prices[prices.Long > 0]
        non_entry_points = prices[prices.Long <= 0]

        # Define colors to use
        col1 = "green"
        col2 = "red"

        ########### Price subplot

        # Plot 'up' prices
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

        # Plot indicators

        for type_name in self._indicator_types:
            self._next_indicator_color[type_name] = 0

        for indicator_name in self._chart.get_all_indicator_names():
            self._plot_indicator(indicator_name, prices, axes1)

        # Put ticks but no labels at bottom
        axes1.set_xticks(x_indices, labels=["" for i in range(len(x_indices))])

        adjusted_str = " (adjusted)" if adjust_to_sma > 0 else ""
        axes1.set_title(f"Price Data for {self._chart.symbol}{adjusted_str}")
        axes1.legend()

        ########### Entry point subplot

        axes2.bar(
            entry_points.index, entry_points.Long, width_candle, bottom=0, color="blue"
        )
        axes2.bar(non_entry_points.index, 0.0, width_candle, bottom=0, color="red")
        # Put ticks but no labels at bottom
        axes2.set_xticks(x_indices, labels=["" for i in range(len(x_indices))])

        axes3.set_title(f"Entry points for {self._chart.symbol}")

        ########### Volume subplot

        # Plot 'up' volume
        axes3.bar(up.index, up.Volume, width_candle, bottom=0, color=col1)

        # Plot 'down' volume
        axes3.bar(down.index, down.Volume, width_candle, bottom=0, color=col2)

        # Add x-axis ticks and labels with dates
        axes3.set_xticks(x_indices, labels=datelabels, rotation=45, ha="right")

        axes3.set_title(f"Volume for {self._chart.symbol}")

        plt.show()

    def _plot_indicator(self, name: str, prices: DataFrame, axes):
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

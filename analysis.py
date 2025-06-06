from typing import TextIO, Dict, Any, List

import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt

# CONSTANT VARIABLES #############################################################################
START_DATE: int = 0
END_DATE: int = 100
NUMBER_OF_INSTRUMENTS: int = 50
RAW_PRICES_FILEPATH: str = "./prices.txt"

# CLASSES ########################################################################################
class MarketData:
    def __init__(self) -> None:
        # Initialise the market data dataframe
        self.market_data_df: DataFrame | None = None
        self.initialise_market_data_dataframe()

    """
    loads the prices.txt file and initialises the market data dataframe with the following 
    columns:

    - day: day number
    - instrument-no: the instrument number
    - open-price: the price at market open for that instrument on that day
    - return: the percentage change for that instrument on that day based on the previous day
    """
    def initialise_market_data_dataframe(self) -> DataFrame:
        # Read the raw prices dataset file
        raw_prices_df: DataFrame = pd.read_csv(RAW_PRICES_FILEPATH, sep=r"\s+", header=None)

        # Initialise a new empty market data dataframe
        market_data_df: DataFrame = pd.DataFrame(columns=["day", "instrument-no", "open-price",
            "change-pct"])

        for day in range(START_DATE, END_DATE):
            for instrument_no in range(0,NUMBER_OF_INSTRUMENTS):
                new_entry: List[int, int, float, float] = []
                new_entry.append(day)
                new_entry.append(instrument_no)
                new_entry.append(raw_prices_df.iloc[day, instrument_no])

                # If it has at least been 1 day, add returns
                if day > 0:
                    change_pct: float = raw_prices_df.iloc[day, instrument_no] / raw_prices_df.iloc[
                        day - 1, instrument_no]
                    new_entry.append(change_pct)
                else:
                    new_entry.append(0)

                # Add the new entry to the dataframe
                market_data_df.loc[day*NUMBER_OF_INSTRUMENTS + instrument_no] = new_entry

        self.market_data_df = market_data_df

    """
    plots an instrument's price data over a specified timeline 
    """
    def plot_instrument_price_data(self, instrument_no: int, start_day: int, end_day: int) -> None:
        # Filter rows matching the instrument and day range and sort by day
        instrument_data: DataFrame = self.market_data_df[
            (self.market_data_df["instrument-no"] == instrument_no) &
            (self.market_data_df["day"] >= start_day) & (self.market_data_df["day"] <= end_day)
        ].sort_values(by="day")

        days: Series = instrument_data["day"]
        prices: Series = instrument_data["open-price"]

        plt.figure(figsize=(10,5))
        plt.plot(days, prices, marker="o", linestyle="-")
        plt.xlabel("Day")
        plt.ylabel("Open Price")
        plt.title(f"Instrument {instrument_no}: Price from Day {start_day} to {end_day}")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()




marketData = MarketData()
marketData.plot_instrument_price_data(5, 0, 100)
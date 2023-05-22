# Stock Chart Analysis Experiments

## Overview

![](./images/TechnicalAnalysis.jpg)

Stock trading relies heavily on the art/science of technical analysis, which involves analyzing the historical price/volume data of a particular security to find patterns and trends in its movement. These provide clues about the level of interest that big funds are taking in the stock, as big funds tend to make moves a little at a time, like a large ship slowly turning into a harbor. The idea is, if you observe the ship, you can guess where it's probably going. And if you're in a small boat, maybe you can trail along behind, letting the big ship clear your way. Usually, that's a better outcome than being in the path of the ship.

With technical analysis, there's also a tail-wagging-the-dog effect at work. Since everyone in the world of stocks and bonds is using technical analysis to some extent, it becomes a self-reinforcing phenomena. If you're a fund manager, and you see a stock's price approaching an important trend line or moving average, you can bet that all other fund managers, aware of the same basic playbook, have the same fact in mind, and know that you probably do as well. Thus, it becomes a self-fulfilling prophecy. Well, not always, but it happens often enough to be a factor.

That's the theory of TA, anyway.

Ever since Deep Learning burst into public awareness, stock analysis has become a popular area of application for these techniques. Certainly, plenty of YouTube videos cover the topic to some extent. Unfortunately, many of these are made by people whose understanding of Deep Learning or stock trading (or both) is pretty limited, and some of the information is questionable -- though still propagated by the stock bro community.

## Purpose of This Project

This project contains a framework for collecting and visualizing stock data, then applying Deep Learning to it via PyTorch.

Folders:

| Folder      | Purpose                                                                                               |
|-------------|-------------------------------------------------------------------------------------------------------|
| `core/`     | Code for collecting and storing price/volume data, and for adding various technical indicators to it. |
| `ui/`       | Code for the visual display of stock charts.                                                          |
| `learning/` | To be added.                                                                                          |

Key Classes:

| Class       | Purpose                                                                                                                                                                                                |
|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Ticker`    | Holds raw stock data, scraped from the web and cached to disk.                                                                                                                                         |
| `Chart`     | For applying extra indicators (such as Simple Moving Average) to price data; these can be helpful for analysis. Also provides functions for returning data in a form that's digestible by an ML system. |
| `MPLHelper` | For plotting data in a `matplotlib` chart.                                                                                                             |

### Adjusted Data

In my opinion, raw price data isn't in an ideal format for DL analysis. Most (tradeable) stocks see their value increase exponentially over years, so a neural network that worked well on prices ranging from $10 to $100, would probably struggle with prices ranging from $500 to $1000, and vice-versa.

My special innovation is to adjust a chart from showing absolute price data to showing price data relative to a particular Simple Moving Average. That moving average becomes a horizontal line with a value of 1.0, and all price data is expressed as a multiplier.

![](./images/RegularPriceData.png)    
`Regular Price Data`

![](./images/AdjustedPriceData.png)    
`Adjusted Price Data`
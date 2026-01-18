# data input format

Price file is a csv with at least two columns "date" and "close" where date is in format YYY-MM-DD. This
can be collected from various places.

The surface is generated from a Kaggle sample option chains data file. The columns used are "quote_date",
"underlying_last", "dte", "strike" and "c_last". This can be downloaded from:

https://www.kaggle.com/datasets/kylegraupe/aapl-options-data-2016-2020

The data isn't necesssarily good quality. For example, many of the underlying prices might be incorrect.
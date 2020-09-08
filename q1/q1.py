import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

DATA_PATH = "2019 Winter Data Science Intern Challenge Data Set.csv"
data = pd.read_csv(DATA_PATH)

# create a dataframe for the two quantitative columns (not ID or categorical)
data_q = data[["order_amount", "total_items"]]

# take a look at the data
print(data.head(), end="\n\n")
print("Count:", data.iloc[:, 0].count(), end="\n\n")
print("Mean:", data_q.mean(), sep="\n", end="\n\n")
print("Median:", data_q.median(), sep="\n", end="\n\n")
print("Max:", data_q.max(), sep="\n", end="\n\n")
print("Skew:", data_q.skew(), sep="\n", end="\n\n")

# box-plot the quantitative data
data_q.plot(subplots=True, kind="box")
# scatter-plot the order totals vs. item counts
data_q.plot("order_amount", "total_items", kind="scatter")

# create a new column to determine product 
data_q = data_q.assign(cost_per_item=data_q["order_amount"] / data_q["total_items"])

data_q.drop_duplicates(subset=["order_amount", "cost_per_item"]).plot("order_amount", "cost_per_item", kind="scatter")

# check that product prices are truly unique (they are)
print(
    "Unique shop_id/cost_per_item combos:",
    pd.concat([data["shop_id"], data_q["cost_per_item"]], axis=1).drop_duplicates(subset=["shop_id"]).iloc[:, 0].count(),
    sep="\n", end="\n\n")

# remove order records > 3.5 standard deviations away from mean in either quantitative attribute
data_q = data_q[
                (pd.Series(data_q["cost_per_item"].pipe(stats.zscore)).abs() <= 3.5)
                & (pd.Series(data_q["total_items"].pipe(stats.zscore)).abs() <= 3.5)
            ].sort_values(by=["cost_per_item", "total_items"])

print("New count: ", data_q.iloc[:, 0].count(), end="\n\n")
print("New mean:", data_q.mean(), sep="\n", end="\n\n")

# now to compare with earlier:
# box-plot the quantitative data
data_q.plot(subplots=True, kind="box")
# scatter-plot the order totals vs. item counts
data_q.plot("order_amount", "total_items", kind="scatter")

plt.show()

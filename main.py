import pandas as pd
from MultiStats import MultiStats


def main():
   featsForStats = pd.read_csv('modified.txt')
   mainDataset = pd.read_csv('GapperStatsFinal2.csv')


   # add extra row to account for the column name being one of your required values
   new_row = pd.DataFrame({'1DayRangePct___17': '1DayRangePct___17'}, index=[0])
   df = pd.concat([new_row, featsForStats]).reset_index(drop=True)

   # Run analysis on whichever variable you'd like. I chose "DayRet"
   tmpStats = MultiStats(mainDataset, featsForStats, "DayRet")


   tmpStats.linRegression(0.1)
   tmpStats.rSquared(0.1)
   tmpStats.ScatterGraph()
if __name__ == "__main__":
  main()

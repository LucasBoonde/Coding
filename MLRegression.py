import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression

df = quandl.get("WIKI/GOOGL")
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume', ]]
df['HL_PCT'] = (df['Adj. High']-df['Adj. Close'])/df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open'] * 100.0

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

#Man kan bruge forecast_col til andet end bare stock prices. 
forecast_col = 'Adj. Close'
#fillna (fill no a number) - er bare fill dem som man ikke kunne få adgang til. Den bliver bare håndteret som en outlier så.
df.fillna(-99999, inplace=True)

#Predicter 10% ud i fremtiden af det data man har?
forecast_out = int(math.ceil(0.01*len(df)))

#Laver et shift på vores forecast_col, som gør at outputtet er vores forecast 10 dage ud i fremtiden.
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

#Features er X, og derfor fjerner man alt det der ikke er features, nemlig Labels.
X = np.array(df.drop(['label'],1))
y = np.array(df['label'])
# - Dette step kan nogle gange skippes
X = preprocessing.scale(X)
df.dropna(inplace=True)
y = np.array(df['label'])

print(len(X), len(y))

add1 = 10 + 1
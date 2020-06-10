import pymongo
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor


def get_data():
	mongo_client = pymongo.MongoClient(
		'mongodb://mlcandidates:crackthecode@100.2.158.147:27017/')
	finDb = mongo_client['findata']

	intradayCollection = finDb['intraday']
	# dailyCollection = finDb['day']

	# all_unique_intraday_symbols = intradayCollection.distinct('Symbol')

	msft_intraday_df = pd.DataFrame(list(intradayCollection.find({
		'Symbol': 'MSFT', 'close': {'$exists':True}})))

	return msft_intraday_df

def predicting_close(data):
	df = data[['volume', 'trending_score', 'sentiment_change',
	'volume_change', 'close']]

	X = df.drop(['close'], axis= 1)
	y = df['close']

	X_train, X_test, y_train, y_test = tts(X, y, test_size=0.33,
		random_state=42)

	dtr = DecisionTreeRegressor(max_depth=2, random_state=0)
	dtr.fit(X_train, y_train)
	y_pred = dtr.predict(X_test)
	mse = mean_squared_error(y_test, y_pred)

	return mse

if __name__ == "__main__":
	data = get_data()
	# regressor(data)
	print("Mean squared error: ", predicting_close(data))


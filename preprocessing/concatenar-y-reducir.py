import pandas as pd

def reducir(in_, out_):
	print(in_, '->', out_)
	df = pd.read_csv(in_, index_col=1)
	df = df.drop(df.columns[0], axis=1)
	#df = df.set_index('id')
	#df = df.drop(['conformation_descriptor_id','drug_conformation_id','software_id'], axis=1)
	df.to_csv(out_)
	del df

reducir('red1.csv', 'x1.csv')
reducir('red2.csv', 'x2.csv')
reducir('red3.csv', 'x3.csv')
reducir('red4.csv', 'x4.csv')
reducir('red5.csv', 'x5.csv')
reducir('red6.csv', 'x6.csv')

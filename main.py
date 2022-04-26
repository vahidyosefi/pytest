#
import pandas as pd
from parsivar import Normalizer
df = pd.read_csv('data.csv', index_col=False)

my_normalizer = Normalizer()

mylist = list(map(lambda x: [my_normalizer.normalize(x[0]), x[1]], zip(df['Text'], df['Suggestion']) ))

mylist1 = list(filter(lambda x: x[1]==1, mylist))
print(len(mylist1))
mylist2 = list(filter(lambda x: x[1]==2, mylist))
print(len(mylist2))
mylist3 = list(filter(lambda x: x[1]==3, mylist))
print(len(mylist3))

f_my_list = mylist1[:400]+mylist2[:400]+mylist3[:400]
from sklearn.model_selection import train_test_split
X,Y = zip(*f_my_list)
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2)


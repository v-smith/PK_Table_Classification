# imports
from sklearn.preprocessing import MultiLabelBinarizer

inp_labels = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [4, 2], [5, 5, 5], [2, 5]]
new_labels = [list(filter(lambda a: a != 5, x)) for x in inp_labels]
print(new_labels)

mlb = MultiLabelBinarizer()
all_transformed_labels = mlb.fit_transform(new_labels)
all_transformed_labels = list(all_transformed_labels)
print(all_transformed_labels)
a = 1

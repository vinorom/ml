import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn import neighbors, svm, ensemble

plt.style.use('ggplot')
# %matplotlib


print '======='
print 'ANALYZE'
print '======='

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data'
url = 'data/crx.data'

interactive_plot = False
enable_plot = False


def plot(msg, file_name):
    if interactive_plot:
        print msg
        plt.show()
    else:
        file_name = 'plot/%s' % file_name
        print '%s: %s' % (msg, file_name)
        plt.savefig(file_name)
    plt.close()


def scatter_plot(data, col1, col2):
    if not enable_plot:
        return

    plt.figure(figsize=(10, 6))

    plt.scatter(data[col1][data['class'] == '+'],
                data[col2][data['class'] == '+'],
                alpha=0.75,
                color='green',
                label = '+')

    plt.scatter(data[col1][data['class'] == '-'],
                data[col2][data['class'] == '-'],
                alpha=0.75,
                color='red',
                label='-')

    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.legend(loc='best')

    plot('scatter plot(%s, %s)' % (col1, col2), 'scatter_plot_%s_%s.png' % (col1, col2))


data = pd.read_csv(url, header=None, na_values='?')

print '---'
size = data.shape
print 'size:', size

print '---'
data.columns = ['A' + str(i) for i in range(1, size[1])] + ['class']
print 'data:\n', data.head(), '\n...'

print '---'
cat_cols = [c for c in data.columns if data[c].dtype.name == 'object']
num_cols = [c for c in data.columns if data[c].dtype.name != 'object']
print 'categorical columns:', cat_cols
print 'numerical columns:', num_cols

print '---'
print 'summary (cat columns):\n', data[cat_cols].describe()  # data.describe(include=[object])
print 'summary (num columns):\n', data[num_cols].describe()  # data.describe()

print '---'
print 'correlation matrix:\n', data.corr()

print '---'
if enable_plot:
    scatter_matrix(data, alpha=0.05, figsize=(10, 10))
    plot('scatter matrix', 'scatter_matrix.png')

print '---'
cols = set()
for col1 in num_cols:
    for col2 in num_cols:
        if col1 != col2 and col2 not in cols:
            scatter_plot(data, col1, col2)
    cols.add(col1)


print '======='
print 'PREPARE'
print '======='

# scikit-learn cannot process data with missing values, drop or fill such items

print '---'
print 'missing values:\n', data.count(axis=0)

# drop rows with missing values
# data = data.dropna(axis=0)

# fill missing values in numerical columns with median of corresponding column
data = data.fillna(data.median(axis=0), axis=0)

# fill missing values in categorical columns with top value of corresponding columns
cat_data_describe = data[cat_cols].describe()
for c in cat_cols:
    data[c] = data[c].fillna(cat_data_describe[c]['top'])

print '---'
print 'missing values:\n', data.count(axis=0)

print '---'
print 'summary (cat columns):\n', data[cat_cols].describe()  # data.describe(include=[object])
print 'summary (num columns):\n', data[num_cols].describe()  # data.describe()

# scikit-learn cannot process categorical features, translate them to numerical ones

print '---'
cat_bin_cols = [c for c in cat_cols if cat_data_describe[c]['unique'] == 2]
cat_nonbin_cols = [c for c in cat_cols if cat_data_describe[c]['unique'] > 2]
print 'categorical binary columns:', cat_bin_cols
print 'categorical non-binary columns:', cat_nonbin_cols

# translate categorical binary feature into numerical: 0, 1
for c in cat_bin_cols:
    top = cat_data_describe[c]['top']
    top_items = data[c] == top
    data.at[top_items, c] = 0
    data.at[np.logical_not(top_items), c] = 1

print '---'
print 'summary (cat bin columns):\n', data[cat_bin_cols].describe()

# translate categorical non-binary feature into multiple binary features with value 0 and 1
data_cat_nonbin = pd.get_dummies(data[cat_nonbin_cols])

print '---'
print 'new cat nonbin columns:\n', data_cat_nonbin.columns

# normalize numerical data
data_num = data[num_cols]
data_num = (data_num - data_num.mean()) / data_num.std()

print '---'
print 'summary (normalized num columns):\n', data_num.describe()

# concatenate all columns again
data = pd.concat((data_num, data[cat_bin_cols], data_cat_nonbin), axis=1)
data = pd.DataFrame(data, dtype=float)

print '---'
print 'size:', data.shape
print '---'
print 'columns:\n', data.columns
print '---'
print 'summary:\n', data.describe()

X = data.drop(('class'), axis=1)
y = data['class']
feature_names = X.columns

print '---'
print 'feature names:\n', feature_names
print '---'
print 'input size:', X.shape
print 'output size:', y.shape


print '====='
print 'LEARN'
print '====='

# split input and output
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)

print '---'
print 'train size:', X_train.shape
print 'test size:', X_test.shape


def find_optimal_params(algo, param_grid):
    grid = GridSearchCV(algo, param_grid=param_grid)
    grid.fit(X_train, y_train)

    err_best_cv = 1 - grid.best_score_
    return err_best_cv, grid.best_params_


def classify(algo):
    algo.fit(X_train, y_train)

    y_train_predict = algo.predict(X_train)
    y_test_predict = algo.predict(X_test)

    err_train = np.mean(y_train != y_train_predict)
    err_test = np.mean(y_test != y_test_predict)

    return err_train, err_test


def classify_with_optimal_params(algo_name, algo, param_grid=None):
    print '---'

    if param_grid:
        cv_err, best_params = find_optimal_params(algo, param_grid)

        print algo_name, 'CV error:', cv_err
        print algo_name, 'best params: ', best_params

        algo.set_params(**best_params)

    err_train, err_test = classify(algo)

    print algo_name, 'train error:', err_train
    print algo_name, 'test error:', err_test



# measure error of classification for different algorithms

# classify_with_optimal_params('kNN', neighbors.KNeighborsClassifier(),
#                              {'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
#                               'weights': ['uniform', 'distance'],
#                               'n_neighbors': [1, 3, 5, 7, 8, 9, 11]})
#
# classify_with_optimal_params('SVC (rbf)', svm.SVC(kernel='rbf'),
#                              {'C': np.logspace(-3, 3, num=7),
#                               'gamma': np.logspace(-5, 2, num=8)})
#
# classify_with_optimal_params('SVC (linear)', svm.SVC(kernel='linear'),
#                              {'C': np.logspace(-3, 3, num=7)})
#
# classify_with_optimal_params('SVC (poly)', svm.SVC(kernel='poly'),
#                              {'C': np.logspace(-5, 2, num=8),
#                               'gamma': np.logspace(-5, 2, num=8),
#                               'degree': [2, 3, 4]})
#
# classify_with_optimal_params('GBT', ensemble.GradientBoostingClassifier(random_state=11, n_estimators=100),
#                              {'loss': ['deviance', 'exponential'],
#                               'max_features': ['auto', 'sqrt', 'log2'],
#                               'max_depth': [3, 7, 10, 15, 20]})

# best one:
# Random Forest CV error: 0.130434782609
# Random Forest best params:  {'n_estimators': 100, 'max_depth': 10}
# Random Forest train error: 0.0124223602484
# Random Forest test error: 0.0869565217391
rf = ensemble.RandomForestClassifier(random_state=11, n_estimators=100)
classify_with_optimal_params('Random Forest', rf,
                             {'max_depth': [3, 7, 10, 15, 20]})

# feature importance

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print '---'
print 'feature importance:'
for f, idx in enumerate(indices):
    print('{:2d}. feature {:5s} ({:.4f})'.format(f + 1, feature_names[idx], importances[idx]))

print '---'
if enable_plot:
    d_first = 20
    plt.figure(figsize=(8, 8))
    plt.title('Feature importance')
    plt.bar(range(d_first), importances[indices[:d_first]], align='center')
    plt.xticks(range(d_first), np.array(feature_names)[indices[:d_first]], rotation=90)
    plt.xlim([-1, d_first]);

    plot('feature importance', 'feature_importance.png')

# best features

best_features = indices[:8]
best_features_names = feature_names[best_features]

print '---'
print 'best features:', best_features_names


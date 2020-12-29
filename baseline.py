from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import sys


def apply_model(data_x, data_y, feature_selector, c_fire):
    sel = None
    num_features = data_x.shape[1]
    if feature_selector == 'var':
        sel = VarianceThreshold(threshold=0.1)
    elif feature_selector == 'k-best':
        ind = data_x < 0
        ind = ind.sum()
        if ind > 0:
            sel = SelectKBest(mutual_info_classif, k=int(num_features * 0.3))
        else:
            sel = SelectKBest(chi2, k=int(num_features * 0.3))
    elif feature_selector == 'pca':
        sel = PCA(n_components=20)
        
    if sel is None:
        raise Exception("feature selector should be one of the [var, k-best, pca]")
    clf = None
    if c_fire == 'nb':
        clf = GaussianNB()
    elif c_fire == 'dt':
        clf = DecisionTreeClassifier(random_state=0)
    elif c_fire == 'lr':
        clf = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='auto', max_iter=1000)
    if clf is None:
        raise Exception("Classifier should be one of the [nb, dt, lr]")
    x_hat = sel.fit_transform(data_x, data_y)
    # op_name = dataset_name + '/' + dataset_name + '_' + feature_selector + '_genex.npy'
    # np.save(dataset_name + '/' + dataset_name + '_' + feature_selector + '_genex.npy', x_hat)
    # print('Saved to', op_name)
    x_train, x_test, y_train, y_test = train_test_split(x_hat, data_y, test_size=0.25, random_state=0)
    y_pred = clf.fit(x_train, y_train)
    scores = clf.score(x_test, y_test)
    del clf
    del sel
    return scores


if __name__ == '__main__':
    # dataset_name = 'Human_Bone_marrow'
    dataset_name = sys.argv[1]
    X = np.load(dataset_name + '/' + dataset_name + '_genex.npy', allow_pickle=True).astype(float)
    y = np.load(dataset_name + '/' + dataset_name + '_labels.npy', allow_pickle=True).astype(float)
    classifiers = ['nb', 'dt', 'lr']
    feature_selectors = ['var', 'k-best', 'pca']
    # feature_selectors = ['pca']
    print('Dataset: ', dataset_name)
    for classifier in classifiers:
        for f_selector in feature_selectors:
            acc = apply_model(X, y, f_selector, 'nb')
            print('Classifier:', classifier, 'Feature_selector:', f_selector, 'Acc:', acc)

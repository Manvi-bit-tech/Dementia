from sklearn.svm import SVC

def train_svm(X_train_flat, y_train):
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X_train_flat, y_train)
    return svm

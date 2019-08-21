from stacking_classifier import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.datasets import load_digits
digits = load_digits()
X, y = digits['data'], digits['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

classifier = StackingClassifier(
    base_classifiers=[
        RandomForestClassifier(subsample_features_indices=[1,4,7,8]),#指定只使用第1,4,7,8列特征用于训练和预测,上层的参数不会覆盖此参数
        AdaBoostClassifier(subsample_features_rate=0.1),#随机选择10%的特征用于训练和预测,上层的参数不会覆盖此参数
        BaggingClassifier(),
        SVMClassifier(),
        StackingClassifier(
            base_classifiers=[
                LogisticRegression(),
                RandomForestClassifier(),
            ],
            meta_classifier=GradientBoostingClassifier(),
        )
    ],
    meta_classifier=LogisticRegression(),
    subsample_features_rate=0.5#该参数会向下传递到最底层的所有未指定subsample_features_rate参数的分类器，subsample_features_indices同理
)
classifier.build_model()
classifier.fit(train_x=X_train, train_y=y_train)
p_test = classifier.predict(X_test)
print(f1_score(y_test, p_test, average='macro'))
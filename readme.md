
### 特点
（1）方便扩展，比如扩展Sklearn,Keras,CatBoost等工具（只需继承stacking_classifier中的Classifier类，并实现相应方法即可）；  
（2）可以构建很深，很复杂的stacking结构  
（3）支持离散变量（为了方便lightgbm,catboost）  
（4）支持并行/并发训练 

接下来，我在手写数值识别上演示api使用示例：  


```python
from stacking_classifier import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.datasets import load_digits
digits = load_digits()
X, y = digits['data'], digits['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
```

### 一.基本分类器的使用
这里所有的分类器都需要实现Classifier类的接口，如果你是使用的Sklearn风格的分类器，只需要做如下操作即可，stacking_classifier中默认封装了SVMClassifier,RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier,BaggingClassifier,LogisticRegression,NaiveBayesClassifier,LightGBMClassifier,CatBoostClassifier等分类器


```python
class LogisticRegression(SklearnClassifier):
    def __init__(self, train_params=None, subsample_features_rate=None, subsample_features_indices=None,
                 categorical_feature_indices=None, n_jobs=1):
        from sklearn.linear_model import LogisticRegression
        SklearnClassifier.__init__(self, train_params, LogisticRegression, subsample_features_rate,
                                   subsample_features_indices, categorical_feature_indices, n_jobs)
```


```python
classifier = LogisticRegression()
classifier.build_model()
classifier.fit(X_train, y_train)
p_test = classifier.predict(X_test)
print(f1_score(y_test, p_test, average='macro'))
```

    0.9454545086848583
    

### 二.KFolds_Classifier_Training_Wrapper包装器的使用
```KFolds_Classifier_Training_Wrapper```可以将数据切分成```k_fold```份，并训练```k_fold```个分类器


```python
classifier = RandomForestClassifier()
classifier = KFolds_Classifier_Training_Wrapper(classifier,k_fold=5)#这里封装一下即可，默认k_fold=5
classifier.build_model()
classifier.fit(X_train, y_train)
p_test = classifier.predict(X_test)
print(f1_score(y_test, p_test, average='macro'))
```

    0.9364765325701125
    


```python
classifier = RandomForestClassifier()
#KFolds_Classifier_Training_Wrapper也可以嵌套封装，这样下面就有25个基分类器
classifier = KFolds_Classifier_Training_Wrapper(KFolds_Classifier_Training_Wrapper(classifier))
classifier.build_model()
classifier.fit(X_train, y_train)
p_test = classifier.predict(X_test)
print(f1_score(y_test, p_test, average='macro'))
```

    0.9403683409907246
    

### 三.StackingClassifier分类器的使用
```StackingClassifier```中的基分类器和元分类器可以是任意继承了Classifier类的子类，由于```KFolds_Classifier_Training_Wrapper```以及```StackingClassifier```都继承了```Classifier```类，所以意味着你可以任意嵌套...


```python
classifier = StackingClassifier(
    base_classifiers=[
        RandomForestClassifier(),
        AdaBoostClassifier(),
        BaggingClassifier(),
        SVMClassifier(),
    ],
    meta_classifier=LogisticRegression(),
    force_cv=False#默认为True,会对base_classifiers，meta_classifier进行KFolds_Classifier_Training_Wrapper包装
)
classifier.build_model()
classifier.fit(train_x=X_train, train_y=y_train)
p_test = classifier.predict(X_test)
print(f1_score(y_test, p_test, average='macro'))
```

    0.9286341275747775
    


```python
classifier = StackingClassifier(
    base_classifiers=[
        RandomForestClassifier(),
        AdaBoostClassifier(),
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
    base_k_fold=5,#基分类器分拆份数,force_cv=True时生效，
    meta_k_fold=5,#元分类器分拆份数,force_cv=True时生效，
)
classifier.build_model()
classifier.fit(train_x=X_train, train_y=y_train)
p_test = classifier.predict(X_test)
print(f1_score(y_test, p_test, average='macro'))
```

    0.9566932316459729
    

### 四.随机/指定选择训练和预测的feature
可以随机选择，通过```subsample_features_indices```指定选择训练的feature,```subsample_features_rate```随机选择训练的feature


```python
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
```

    0.9568685122123126
    

### 五.支持离散变量的输入
这里为了方便lightgbm,catboost操作而支持离散变量类型,注意：  
（1）**必须在最顶层指定str/object类型的变量**（这样底层不支持str/object类型的分类器才能过滤掉这些特征）；  
（2）lightgbm不支持'x','y','z'这种类型的离散变量，只支持‘1’,'2','3'或者int/float类型的离散变量，所以有时需要单独指定；  
（3）如果指定了```categorical_feature_indices```参数，```subsample_features_rate,subsample_features_indices```退化为只对剩余的非```categorical_feature_indices```特征生效


```python
#为原始数据添加两列：一列是数值的字符串，一列是随意的字符串
import numpy as np
new_column = np.asarray(['1'] * 1797)
new_column2 = np.asarray(['x'] * 1797)
X_new = np.concatenate([X, new_column.reshape(1797, 1), new_column2.reshape(1797, 1)], axis=1)
X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(X_new, y, test_size=0.7, random_state=42)
```


```python
X_new_train[0:1]
```




    array([['0.0', '0.0', '10.0', '13.0', '9.0', '1.0', '0.0', '0.0', '0.0',
            '2.0', '16.0', '7.0', '10.0', '8.0', '0.0', '0.0', '0.0', '0.0',
            '12.0', '12.0', '7.0', '11.0', '0.0', '0.0', '0.0', '3.0',
            '16.0', '16.0', '16.0', '7.0', '0.0', '0.0', '0.0', '0.0', '5.0',
            '8.0', '12.0', '10.0', '1.0', '0.0', '0.0', '0.0', '0.0', '0.0',
            '0.0', '11.0', '7.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
            '3.0', '15.0', '0.0', '0.0', '0.0', '11.0', '16.0', '16.0',
            '16.0', '8.0', '0.0', '1', 'x']], dtype='<U32')




```python
classifier = StackingClassifier(
    base_classifiers=[
        RandomForestClassifier(subsample_features_indices=[1,4,7,8]),
        AdaBoostClassifier(subsample_features_rate=0.1),
        LogisticRegression(),
        LightGBMClassifier(categorical_feature_indices=[64]),#第65列特征为'x','y'类型，ligthgbm底层不支持
        CatBoostClassifier(train_params={'depth': 3, 'iterations': 50}),#若不自定义，由顶层传下来的categorical_feature_indices覆盖
        StackingClassifier(
            base_classifiers=[
                LogisticRegression(),
                RandomForestClassifier(),
            ],
            meta_classifier=GradientBoostingClassifier(),
        )
    ],
    meta_classifier=LogisticRegression(),
    subsample_features_rate=0.5,
    categorical_feature_indices=[64,65],
)
classifier.build_model()
classifier.fit(train_x=X_new_train, train_y=y_new_train)
p_test = classifier.predict(X_new_test)
print(f1_score(y_new_test, p_test, average='macro'))
```

    0.952295719121581
    

### 六.超参设置
超参的设置通过```train_params```传入，具体参数的命名与底层封装的分类器一致，比如....


```python
classifier = StackingClassifier(
    base_classifiers=[
        RandomForestClassifier(subsample_features_indices=[1,4,7,8],train_params={'n_estimators':200}),
        AdaBoostClassifier(subsample_features_rate=0.1),
        LogisticRegression(train_params={'penalty':'l2','C':1.0}),
        LightGBMClassifier(),
        CatBoostClassifier(train_params={'depth': 3, 'iterations': 50}),
        StackingClassifier(
            base_classifiers=[
                LogisticRegression(train_params={'C':2.0}),
                RandomForestClassifier(),
            ],
            meta_classifier=GradientBoostingClassifier(),
        )
    ],
    meta_classifier=LogisticRegression(),
    subsample_features_rate=0.5,
)
classifier.build_model()
classifier.fit(train_x=X_train, train_y=y_train)
p_test = classifier.predict(X_test)
print(f1_score(y_test, p_test, average='macro'))
```

    0.9551347109139192
    

### 七.自定义分类器
这里使用Keras实现MLP来演示，由于Keras不是Sklearn风格的api，所以需要继承Classifier类


```python
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
class SimpleMLPClassifer(Classifier):
    def __init__(self, train_params=None, subsample_features_rate=None, subsample_features_indices=None,
                 categorical_feature_indices=None):
        """
        :param train_params:
        """
        Classifier.__init__(self, train_params, subsample_features_rate,
                                   subsample_features_indices, categorical_feature_indices)
        self._check_params()

    def _check_params(self):
        if 'input_num' not in self.train_params:
            raise RuntimeError('no input_num param in train_params!')
        if 'class_num' not in self.train_params:
            raise RuntimeError('no class_num param in train_params!')
        if 'batch_size' not in self.train_params:
            self.train_params['batch_size'] = 64
        if 'epochs' not in self.train_params:
            self.train_params['epochs'] = 5
        if 'shuffle' not in self.train_params:
            self.train_params['shuffle'] = True
        if 'validation_split' not in self.train_params:
            self.train_params['validation_split'] = 0.05

    def build_model(self):
        self.classifier_model = Sequential()
        self.classifier_model.add(Dense(512, input_shape=(self.train_params['input_num'],)))
        self.classifier_model.add(Activation('relu'))
        self.classifier_model.add(Dropout(0.5))
        self.classifier_model.add(Dense(self.train_params['class_num']))
        self.classifier_model.add(Activation('softmax'))
        self.classifier_model.compile(loss='categorical_crossentropy',
                                      optimizer='adam',
                                      metrics=['accuracy'])

    def fit(self, train_x, train_y):
        self.classifier_model.fit(x=train_x, y=utils.to_categorical(train_y, self.train_params['class_num']),
                                  batch_size=self.train_params['batch_size'], epochs=self.train_params['epochs'],
                                  validation_split=self.train_params['validation_split'],
                                  shuffle=self.train_params['shuffle'],
                                  verbose=False)

    def predict_categorical(self, test_x):
        categorical_labels = self.classifier_model.predict(test_x, batch_size=test_x.shape[0])
        new_categorical_result = np.zeros(shape=categorical_labels.shape)
        for index in range(0, len(categorical_labels)):
            categorical_label = categorical_labels[index].tolist()
            maxvalue_index = categorical_label.index(max(categorical_label))
            new_categorical_result[index][maxvalue_index] = 1
        return new_categorical_result

    def predict(self, test_x):
        p_categorical_probas = self.predict_categorical_proba(test_x)
        result = []
        for categorical_proba in p_categorical_probas:
            categorical_proba = categorical_proba.tolist()
            result.append(categorical_proba.index(max(categorical_proba)))
        return np.asarray(result)

    def predict_proba(self, test_x):
        return self.classifier_model.predict_proba(test_x, batch_size=test_x.shape[0])

    def predict_categorical_proba(self, test_x):
        probas = self.classifier_model.predict_proba(test_x)
        _, col = probas.shape
        if col > 1:
            return probas
        else:
            return np.asarray([[1 - proba, proba] for proba in probas])
```

    Using TensorFlow backend.
    


```python
#然后就可以嵌入到Stacking中了
classifier = StackingClassifier(
    base_classifiers=[
        LightGBMClassifier(),
        CatBoostClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        BaggingClassifier(),
        SVMClassifier(),
        StackingClassifier(
            base_classifiers=[
                SimpleMLPClassifer(train_params={'input_num':64,'class_num':10}),#比如放这儿
                RandomForestClassifier(),
            ],
            meta_classifier=GradientBoostingClassifier(),
        )
    ],
    meta_classifier=LogisticRegression()
)
classifier.build_model()
classifier.fit(train_x=X_train, train_y=y_train)
p_test = classifier.predict(X_test)
print(f1_score(y_test, p_test, average='macro'))
```

    0.9552219914166459
    

### 八.并行/并发训练
在Linux中采用多进程并行的方式训练，在Windows中采用多线程并发的方式训练，目前仅在Windows中做过简单测试，能比串行训练提速70%+左右（视具体Stacking结构的不同，提速效率也不一样，不建议将meta_classifier定义为复杂的结构，这部分没有做过多优化），使用方式很简单，在最顶层设置```n_jobs=-1```即可，该模块后面还会持续优化...


```python
classifier = StackingClassifier(
    base_classifiers=[
        RandomForestClassifier(subsample_features_indices=[1,4,7,8],train_params={'n_estimators':200}),
        AdaBoostClassifier(subsample_features_rate=0.1),
        LogisticRegression(train_params={'penalty':'l2','C':1.0}),
        LightGBMClassifier(),
        CatBoostClassifier(train_params={'depth': 3, 'iterations': 50}),
        StackingClassifier(
            base_classifiers=[
                LogisticRegression(train_params={'C':2.0}),
                RandomForestClassifier(),
            ],
            meta_classifier=GradientBoostingClassifier(),
        )
    ],
    meta_classifier=LogisticRegression(),
    subsample_features_rate=0.5,
    n_jobs=-1#这里
)
classifier.build_model()
classifier.fit(train_x=X_train, train_y=y_train)
p_test = classifier.predict(X_test)
print(f1_score(y_test, p_test, average='macro'))
```

    0.9576968080725516
    

### 九.模型保存与加载


```python
#保存
classifier.save_model('stacking.model')
```


```python
#加载
new_classifier=Classifier.load_model('stacking.model')#注意是Classifier类，不是classifier对象
```


```python
p_test = new_classifier.predict(X_test)
print(f1_score(y_test, p_test, average='macro'))
```

    0.9576968080725516
    

### 十.回归
回归的操作与Classifier类似，不再赘述，下面列一下对应关系：  
stacking_classifier->stacking_regressor   
Classifier->Regressor  
SklearnClassifier->SklearnRegressor  
KFolds_Classifier_Training_Wrapper->KFolds_Regressor_Training_Wrapper  
StackingClassifier->StackingRegressor  

```subsample_features_rate,subsample_features_indices,categorical_feature_indices,n_jobs```的相关内容还未在回归中实现，后续更新...


### 特点
（1）通过实现Classifier/SklearnClassifier接口（SklearnClassifier是Classifier的子类），可以方便将分类器扩展到Sklearn,Keras等其他开源工具；  
（2）可以构建很深，很复杂的stacking结构  

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
这里所有的分类器都需要实现Classifier类的接口，如果你是使用的Sklearn风格的分类器，只需要做如下操作即可，stacking_classifier中默认封装了SVMClassifier,RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier,BaggingClassifier,LogisticRegression,NaiveBayesClassifier等分类器


```python
class AdaBoostClassifier(SklearnClassifier):
    def __init__(self, train_params=None, subsample_features_rate=None, subsample_features_indices=None):
        from sklearn.ensemble import AdaBoostClassifier
        SklearnClassifier.__init__(self, train_params, AdaBoostClassifier, subsample_features_rate,
                                   subsample_features_indices)
```


```python
classifier = AdaBoostClassifier()
classifier.build_model()
classifier.fit(X_train, y_train)
p_test = classifier.predict(X_test)
print(f1_score(y_test, p_test, average='macro'))
```

    0.16511852016226553
    

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

    0.9268797312967999
    


```python
classifier = RandomForestClassifier()
#KFolds_Classifier_Training_Wrapper也可以嵌套封装，这样下面就有25个基分类器
classifier = KFolds_Classifier_Training_Wrapper(KFolds_Classifier_Training_Wrapper(classifier))
classifier.build_model()
classifier.fit(X_train, y_train)
p_test = classifier.predict(X_test)
print(f1_score(y_test, p_test, average='macro'))
```

    0.944350402788228
    

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

    0.9332683565603823
    


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

    0.9582680143374441
    

### 四.随机/指定选择训练和预测的feature
可以随机选择，指定选择训练的feature


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

    0.9557944315358944
    

### 五.模型保存与加载


```python
#保存
classifier.save_model('./classifier_model/stacking.model')
```


```python
#加载
new_classifier=Classifier.load_model('./classifier_model/stacking.model')
```


```python
p_test = new_classifier.predict(X_test)
print(f1_score(y_test, p_test, average='macro'))
```

    0.9557944315358944
    

### 六.自定义分类器
这里使用Keras实现MLP来演示，由于Keras不是Sklearn风格的api，所以需要继承Classifier类


```python
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
class SimpleMLPClassifer(Classifier):
    def __init__(self, train_params=None):
        """
        :param train_params:
        """
        Classifier.__init__(self, train_params)
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
        RandomForestClassifier(),
        AdaBoostClassifier(),
        BaggingClassifier(),
        SVMClassifier(),
        StackingClassifier(
            base_classifiers=[
                SimpleMLPClassifer(train_params={'input_num': 64, 'class_num': 10}),#放这儿
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

    0.9473223968636466
    

### 七.回归
回归的操作与Classifier类似，不再赘述，下面列一下对应关系：  
stacking_classifier->stacking_regressor   
Classifier->Regressor  
SklearnClassifier->SklearnRegressor  
KFolds_Classifier_Training_Wrapper->KFolds_Regressor_Training_Wrapper  
StackingClassifier->StackingRegressor  

# Stacking_Ensembles


A python implementation of Stacking_Ensembles, including Stacking Classification and Regression.

**Features:**
- Base classifiers or regressions are extensible, and you can implement them using **sklearn**, **keras**, **sparkML**, or even your own way.
- Flexible **cross training wrapper**, you can easily create hundreds of classifiers or regressor.
- The stacking classifier/regressor can be used as a base classifier/regressor, so you can build a **deep stacking network** classifier/regressor.


**TODO**
- Adding more base classifiers and regressors.
- Automatic parameter optimization of classifiers and regressors.


## Usage
**Step 1.**


First, you should install the dependency packages in the requirements.txt.

**Step 2.**

Load train data (I only demonstrate the usage of classifiers later, it's similar to the regressor.)
```python
from stacking_classifier import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.datasets import load_digits
digits = load_digits()
X, y = digits['data'], digits['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=227)
```

## Demo

**Base classifier/Regressor**

You can simply use the base classifier/regressor like the following.

```python
classifier = AdaBoostClassifier(where_store_classifier_model='./classifier_model/demo1_ada.model')#define base classifer
classifier.build_model()#build model
classifier.fit(X_train, y_train)#fit data
p_test = classifier.predict(X_test)#predict test data
print('demo1-1 ada : ', f1_score(y_test, p_test, average='macro'))#score
```



**Cross Training Wrapper:**

You can use **KFolds_Classifier_Training_Wrapper** or **KFolds_Regressor_Training_Wrapper** to wrap base classifier/regressor up. It divides the training data into ```k_fold``` parts to cross train ```k_fold``` models.
```python
classifier = RandomForestClassifier(where_store_classifier_model='./classifier_model/demo1_1_rf.model')#define a base classifier
classifier = KFolds_Classifier_Training_Wrapper(classifier)#wrap it,there are 5 models
classifier.build_model()
classifier.fit(X_train, y_train)
p_test = classifier.predict(X_test)
print('demo2-1 rf : ', f1_score(y_test, p_test, average='macro'))
```

You may have found the classifier/regressor is still packaged as a base classifier/regressor after being wrapped. So you can use the wrapper like this.

```python
classifier=KFolds_Classifier_Training_Wrapper(KFolds_Classifier_Training_Wrapper(classifier))#there are 25 models!
```


**Simple Stacking classifier/regressor**

You can use simple stacking classfier like this. Every base classifier and meta classfier will be wrapped with KFolds_Classifier_Training_Wrapper automatically.
```python
classifier = StackingClassifier(
    base_classifiers=[
        RandomForestClassifier(where_store_classifier_model='./classifier_model/demo_3_1_layer_2_rf_stack_cv.model'),
        AdaBoostClassifier(where_store_classifier_model='./classifier_model/demo_3_1_layer_2_ada_stack_cv.model'),
        BaggingClassifier(where_store_classifier_model='./classifier_model/demo_3_1_layer_2_bag_stack_cv.model'),
        SVMClassifier(where_store_classifier_model='./classifier_model/demo_3_1_layer_2_svm_stack_cv.model'),
    ],
    meta_classifier=LogisticRegression(where_store_classifier_model='./classifier_model/demo_3_1_layer_1_lr_stack_cv.model'),
)
classifier.build_model()
classifier.fit(train_x=X_train, train_y=y_train)
p_test = classifier.predict(X_test)
print('demo3-1 simple stack: ', f1_score(y_test, p_test, average='macro'))
```

Or you can use  ```force_cv=False``` to define single classifier wrapping with KFolds_Classifier_Training_Wrapper.

```python
classifier = StackingClassifier(
    base_classifiers=[
        KFolds_Classifier_Training_Wrapper(
            RandomForestClassifier(where_store_classifier_model='./classifier_model/demo_3_2_layer_2_rf_stack_cv.model'),
            k_fold=5),  # this base classifier will be wrapped only
        AdaBoostClassifier(where_store_classifier_model='./classifier_model/demo_3_2_layer_2_ada_stack_cv.model'),
        BaggingClassifier(where_store_classifier_model='./classifier_model/demo_3_2_layer_2_bag_stack_cv.model'),
        SVMClassifier(where_store_classifier_model='./classifier_model/demo_3_2_layer_2_svm_stack_cv.model'),
    ],
    meta_classifier=LogisticRegression(where_store_classifier_model='./classifier_model/demo_3_2_layer_1_lr_stack_cv.model'),
    force_cv=False
)
classifier.build_model()
classifier.fit(train_x=X_train, train_y=y_train)
p_test = classifier.predict(X_test)
print('demo3-2 simple stack: ', f1_score(y_test, p_test, average='macro'))
```

**Deep Stacking classifier/regressor**

You can find this stacking classifier/regressor is still packaged as a base classifier/regressor, so you can define deep stacking classifier/regressor.

```python
classifier = StackingClassifier(
    base_classifiers=[
        RandomForestClassifier(where_store_classifier_model='./classifier_model/demo_3_3_layer_2_rf_stack_cv.model'),
        AdaBoostClassifier(where_store_classifier_model='./classifier_model/demo_3_3_layer_2_ada_stack_cv.model'),
        BaggingClassifier(where_store_classifier_model='./classifier_model/demo_3_3_layer_2_bag_stack_cv.model'),
        SVMClassifier(where_store_classifier_model='./classifier_model/demo_3_3_layer_2_svm_stack_cv.model'),
        StackingClassifier(
            base_classifiers=[
                LogisticRegression(where_store_classifier_model='./classifier_model/demo_3_3_layer_3_lr_stack_cv.model'),
                RandomForestClassifier(where_store_classifier_model='./classifier_model/demo_3_3_layer_3_rf_stack_cv.model'),
            ],
            meta_classifier=GradientBoostingClassifier(
                where_store_classifier_model='./classifier_model/demo_3_3_layer_3_gbdt_stack_cv.model'),
        )
    ],
    meta_classifier=LogisticRegression(
        where_store_classifier_model='./classifier_model/demo_3_3_layer_1_lr_stack_cv.model'),
)
classifier.build_model()
classifier.fit(train_x=X_train, train_y=y_train)
p_test = classifier.predict(X_test)
print('demo3-3 deep stack: ', f1_score(y_test, p_test, average='macro'))
```

Obviously, it can be wrapped.

```python
classifier = StackingClassifier(...)
classifier=KFolds_Classifier_Training_Wrapper(classifier)
...
```

**Extensible Base/Meta Classifier/Regressor**

You can overwrite the methods in in Classifer/Regressor to define yourself classifer/regressor.

```python
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import  Dropout, Dense,Activation
class SimpleMLPClassifer(Classifier):
    def __init__(self, where_store_classifier_model=None, train_params=None):
        """
        :param where_store_classifier_model:
        :param train_params:
        """
        Classifier.__init__(self, where_store_classifier_model, train_params)
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
        self.classifier_model.fit(x=train_x, y=to_categorical(train_y, self.train_params['class_num']),
                                  batch_size=self.train_params['batch_size'], epochs=self.train_params['epochs'],
                                  validation_split=self.train_params['validation_split'],
                                  shuffle=self.train_params['shuffle'],
                                  verbose=False)

    def predict_categorical(self, test_x):
        categorical_labels=self.classifier_model.predict(test_x, batch_size=test_x.shape[0])
        new_categorical_result=np.zeros(shape=categorical_labels.shape)
        for index in range(0,len(categorical_labels)):
            categorical_label=categorical_labels[index].tolist()
            maxvalue_index=categorical_label.index(max(categorical_label))
            new_categorical_result[index][maxvalue_index]=1
        return new_categorical_result

    def predict(self, test_x):
        p_categorical_probas = self.predict_categorical_proba(test_x)
        result=[]
        for categorical_proba in p_categorical_probas:
            categorical_proba=categorical_proba.tolist()
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

    def save_model(self):
        self.classifier_model.save(self.classifier_model_path)

    def load_model(self):
        self.classifier_model = load_model(self.classifier_model_path)
```

The self-defined model can be user for base/meta classifier/regressor.

```python
classifier = StackingClassifier(
    base_classifiers=[
        RandomForestClassifier(where_store_classifier_model='./classifier_model/demo_3_4_layer_2_rf_stack_cv.model'),
        AdaBoostClassifier(where_store_classifier_model='./classifier_model/demo_3_4_layer_2_ada_stack_cv.model'),
        BaggingClassifier(where_store_classifier_model='./classifier_model/demo_3_4_layer_2_bag_stack_cv.model'),
        SVMClassifier(where_store_classifier_model='./classifier_model/demo_3_4_layer_2_svm_stack_cv.model'),
        SimpleMLPClassifer(where_store_classifier_model='./classifier_model/demo_3_4_layer_2_mlp_stack_cv.model',train_params={'input_num':64,'class_num':10})
    ],
    meta_classifier=LogisticRegression(where_store_classifier_model='./classifier_model/demo_3_4_layer_1_lr_stack_cv.model'),
)
classifier.build_model()
classifier.fit(train_x=X_train, train_y=y_train)
p_test = classifier.predict(X_test)
print('demo3-4 simple stack: ', f1_score(y_test, p_test, average='macro'))
```

Or wrapping.

```python
classifier=SimpleMLPClassifer(...)
classifier=KFolds_Classifier_Training_Wrapper(classifier)
...
```

**For more details, you can view the \*_examples.py**
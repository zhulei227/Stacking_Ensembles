from stacking_classifer import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.datasets import load_digits

'''
加载测试数据:数字识别(0~9)
input_num:64
class_num:10
'''
digits = load_digits()
X, y = digits['data'], digits['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=227)

''''
demo1:重构XXXClassfier,只需实现重写Classfier中的接口即可,可基于sparkml,keras,sklearn等扩展,比如下面两个例子分别基于sklearn和keras扩展
主要包括:
build_model:构建模型
fit:拟合模型
save_model:保存模型
load_model:加载模型
predict:预测分类标签0~N
predict_categorical:预测分类的one-hot形式标签
predict_categorical_proba:预测分类的one-hot形式概率分布
predict_proba:预测二分类正样的概率值,多分类与predict_categorical_proba一样
'''
classifier = AdaClassifier(where_store_classifier_model='./classifier_model/demo1_ada.model')
classifier.build_model()
classifier.fit(X_train, y_train)
p_test = classifier.predict(X_test)
print('demo1-1 ada : ', f1_score(y_test, p_test, average='macro'))

classifier = SimpleMLPClassifer(where_store_classifier_model='./classifier_model/demo1_mlp.model',
                                train_params={'input_num': 64, 'class_num': 10})  # 分类器的训练参数通过train_params传入
classifier.build_model()
classifier.fit(X_train, y_train)
p_test = classifier.predict(X_test)
print('demo1-2 mlp: ', f1_score(y_test, p_test, average='macro'))

'''
#demo2:交叉训练包装器,包装后依然当作Classifer使用，所以可以无限包装，但分类器数量会指数增涨
该部分主要为Stacking分类的cv提供协助,独立使用的例子如下:
'''
classifier = RFClassifier(where_store_classifier_model='./classifier_model/demo2_1_rf.model')
classifier = KFolds_Training_Wrapper(classifier)  # 默认2-fold,可指定KFolds_Training_Wrapper(classifier,k_fold=5)
classifier.build_model()
classifier.fit(X_train, y_train)
p_test = classifier.predict(X_test)
print('demo2-1 rf : ', f1_score(y_test, p_test, average='macro'))

classifier = RFClassifier(where_store_classifier_model='./classifier_model/demo2_2_rf.model')
classifier = KFolds_Training_Wrapper(KFolds_Training_Wrapper(classifier, k_fold=5), k_fold=5)  # 这样会训练25个分类器
classifier.build_model()
classifier.fit(X_train, y_train)
p_test = classifier.predict(X_test)
print('demo2-2 rf : ', f1_score(y_test, p_test, average='macro'))

'''
demo3:Stacking集成分类器的使用:Stacking分类器可以利用其它分类器输出,作为另一分类器的输入，前者称基分类器，后者称元分类器
创建方式：classifier=StackClassifer(base_classifiers,meta_classifier,use_probas=True, force_cv=True)
base_classifers:基分类器列表
meta_classifier:元分类器
use_probas:使用使用基分类器的概率预测标签作为元分类器的输入,默认使用
force_cv:是否强制为每个基分类器和元分类器添加KFolds_Training_Wrapper包装,默认添加(如果已经包装的不会再包装),个别分类器不想使用CV方式训练，可以定义force_cv=False,然后单独定义每个基分类器和元分类器
'''

'''
demo3-1:比如利用RF,Ada,Bag,SVM作为基分类器,LR作为元分类器做集成
'''
classifier = StackingClassifier(
    base_classifiers=[
        RFClassifier(where_store_classifier_model='./classifier_model/demo_3_1_layer_2_rf_stack_cv.model'),
        AdaClassifier(where_store_classifier_model='./classifier_model/demo_3_1_layer_2_ada_stack_cv.model'),
        BagClassifier(where_store_classifier_model='./classifier_model/demo_3_1_layer_2_bag_stack_cv.model'),
        SVMClassifier(where_store_classifier_model='./classifier_model/demo_3_1_layer_2_svm_stack_cv.model'),
    ],
    meta_classifier=LRClassifier(where_store_classifier_model='./classifier_model/demo_3_1_layer_1_lr_stack_cv.model'),
)
classifier.train(train_x=X_train, train_y=y_train)
classifier.load_model()
p_test = classifier.predict(X_test)
print('demo3-1 simple stack: ', f1_score(y_test, p_test, average='macro'))
'''
demo3-2:单独为某些分类器包装KFolds_Training_Wrapper时,设置force_cv=False
'''
classifier = StackingClassifier(
    base_classifiers=[
        KFolds_Training_Wrapper(
            RFClassifier(where_store_classifier_model='./classifier_model/demo_3_2_layer_2_rf_stack_cv.model'),
            k_fold=5),  # 仅该基分类器使用CV方式训练
        AdaClassifier(where_store_classifier_model='./classifier_model/demo_3_2_layer_2_ada_stack_cv.model'),
        BagClassifier(where_store_classifier_model='./classifier_model/demo_3_2_layer_2_bag_stack_cv.model'),
        SVMClassifier(where_store_classifier_model='./classifier_model/demo_3_2_layer_2_svm_stack_cv.model'),
    ],
    meta_classifier=LRClassifier(where_store_classifier_model='./classifier_model/demo_3_2_layer_1_lr_stack_cv.model'),
)
classifier.train(train_x=X_train, train_y=y_train)
classifier.load_model()
p_test = classifier.predict(X_test)
print('demo3-2 simple stack: ', f1_score(y_test, p_test, average='macro'))
'''
demo3-3:StackingClassifier也可以作为基分类器使用,所以可以堆叠很深的结构
'''
classifier = StackingClassifier(
    base_classifiers=[
        RFClassifier(where_store_classifier_model='./classifier_model/demo_3_3_layer_2_rf_stack_cv.model'),
        AdaClassifier(where_store_classifier_model='./classifier_model/demo_3_3_layer_2_ada_stack_cv.model'),
        BagClassifier(where_store_classifier_model='./classifier_model/demo_3_3_layer_2_bag_stack_cv.model'),
        SVMClassifier(where_store_classifier_model='./classifier_model/demo_3_3_layer_2_svm_stack_cv.model'),
        StackingClassifier(
            base_classifiers=[
                LRClassifier(where_store_classifier_model='./classifier_model/demo_3_3_layer_3_lr_stack_cv.model'),
                RFClassifier(where_store_classifier_model='./classifier_model/demo_3_3_layer_3_rf_stack_cv.model'),
            ],
            meta_classifier=GBDTClassifier(
                where_store_classifier_model='./classifier_model/demo_3_3_layer_3_gbdt_stack_cv.model'),
        )
    ],
    meta_classifier=SimpleMLPClassifer(
        where_store_classifier_model='./classifier_model/demo_3_3_layer_1_lr_stack_cv.model',
        train_params={'input_num': 50, 'class_num': 10}),
)
classifier.train(train_x=X_train, train_y=y_train)
classifier.load_model()
p_test = classifier.predict(X_test)
print('demo3-3 deep stack: ', f1_score(y_test, p_test, average='macro'))

'''
demo3-4:StackingClassifier也可以被KFolds_Training_Wrapper包装
'''
classifier = KFolds_Training_Wrapper(StackingClassifier(
    base_classifiers=[
        RFClassifier(where_store_classifier_model='./classifier_model/demo_3_4_layer_2_rf_stack_cv.model'),
        AdaClassifier(where_store_classifier_model='./classifier_model/demo_3_4_layer_2_ada_stack_cv.model'),
        BagClassifier(where_store_classifier_model='./classifier_model/demo_3_4_layer_2_bag_stack_cv.model'),
        SVMClassifier(where_store_classifier_model='./classifier_model/demo_3_4_layer_2_svm_stack_cv.model'),
        StackingClassifier(
            base_classifiers=[
                LRClassifier(where_store_classifier_model='./classifier_model/demo_3_4_layer_3_lr_stack_cv.model'),
                RFClassifier(where_store_classifier_model='./classifier_model/demo_3_4_layer_3_rf_stack_cv.model'),
            ],
            meta_classifier=GBDTClassifier(
                where_store_classifier_model='./classifier_model/demo_3_4_layer_3_gbdt_stack_cv.model')
        )
    ],
    meta_classifier=LRClassifier(
        where_store_classifier_model='./classifier_model/demo_3_4_layer_1_lr_stack_cv.model'),
))
classifier.train(train_x=X_train, train_y=y_train)
classifier.load_model()
p_test = classifier.predict(X_test)
print('demo3-4 deep deep stack: ', f1_score(y_test, p_test, average='macro'))

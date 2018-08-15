基于sklearn,keras的stacking ensemble实现,主要功能:

1.继承Classifier类,可自己扩展基分类器,比如classifier_examples.py中的SimpleMLPClassifer类

2.对分类器的CV训练包装:

classifier = RFClassifier(where_store_classifier_model='./classifier_model/demo2_1_rf.model')
classifier = KFolds_Training_Wrapper(classifier)  # 默认2-fold,可指定KFolds_Training_Wrapper(classifier,k_fold=5)
classifier.build_model()
classifier.fit(X_train, y_train)
p_test = classifier.predict(X_test)
print('demo2-1 rf : ', f1_score(y_test, p_test, average='macro'))

3.堆叠任意深结构的stacking 分类器

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
            use_probas=False
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

相关demo查看*_examples.py文件
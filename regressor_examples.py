from stacking_regressor import *
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=227)


''''
demo1:重构XXXRegressor,只需实现重写Regressor中的接口即可,可基于sparkml,keras,sklearn等扩展,比如下面两个例子分别基于sklearn和keras扩展
主要包括:
build_model:构建模型
fit:拟合模型
save_model:保存模型
load_model:加载模型
predict:预测
'''
regressor = AdaBoostRegressor(where_store_regressor_model='./regressor_model/demo1_ada.model')
regressor.build_model()
regressor.fit(X_train, y_train)
p_test = regressor.predict(X_test)
print('demo1-1 ada : ', mean_squared_error(y_test, p_test))

regressor = SimpleMLPRegressor(where_store_regressor_model='./regressor_model/demo1_mlp.model',
                                train_params={'input_num': 10})  # 回归器的训练参数通过train_params传入
regressor.build_model()
regressor.fit(X_train, y_train)
p_test = regressor.predict(X_test)
print('demo1-2 mlp: ', mean_squared_error(y_test, p_test))

'''
#demo2:交叉训练包装器,包装后依然当作Regressor使用，所以可以无限包装，但回归器数量会指数增涨
该部分主要为Stacking回归的cv提供协助,独立使用的例子如下:
'''
regressor = ElasticNetCV(where_store_regressor_model='./regressor_model/demo2_1_en.model')
regressor = KFolds_Regressor_Training_Wrapper(regressor)  # 默认2-fold,可指定KFolds_Regressor_Training_Wrapper(regressor,k_fold=5)
regressor.build_model()
regressor.fit(X_train, y_train)
p_test = regressor.predict(X_test)
print('demo2-1 rf : ', mean_squared_error(y_test, p_test))

regressor = ElasticNetCV(where_store_regressor_model='./regressor_model/demo2_2_en.model')
regressor = KFolds_Regressor_Training_Wrapper(KFolds_Regressor_Training_Wrapper(regressor, k_fold=2), k_fold=2)  # 这样会训练4个回归器
regressor.build_model()
regressor.fit(X_train, y_train)
p_test = regressor.predict(X_test)
print('demo2-2 rf : ', mean_squared_error(y_test, p_test))

'''
demo3:Stacking集成回归器的使用:Stacking回归器可以利用其它回归器输出,作为另一回归器的输入，前者称基回归器，后者称元回归器
创建方式：regressor=StackRegressor(base_regressors,meta_regressor,use_probas=True, force_cv=True)
base_classifers:基回归器列表
meta_regressor:元回归器
force_cv:是否强制为每个基回归器和元回归器添加KFolds_Training_Wrapper包装,默认添加(如果已经包装的不会再包装),个别回归器不想使用CV方式训练，可以定义force_cv=False,然后单独定义每个基回归器和元回归器
'''

'''
demo3-1:比如利用RF,Ada,Bag,SVM作为基回归器,LR作为元回归器做集成
'''
regressor = StackingRegressor(
    base_regressors=[
        ElasticNetCV(where_store_regressor_model='./regressor_model/demo_3_1_layer_2_en_stack_cv.model'),
        AdaBoostRegressor(where_store_regressor_model='./regressor_model/demo_3_1_layer_2_ada_stack_cv.model'),
        BaggingRegressor(where_store_regressor_model='./regressor_model/demo_3_1_layer_2_bag_stack_cv.model'),
        SVRRegressor(where_store_regressor_model='./regressor_model/demo_3_1_layer_2_svr_stack_cv.model'),
    ],
    meta_regressor=LinearRegression(where_store_regressor_model='./regressor_model/demo_3_1_layer_1_lr_stack_cv.model'),
)
regressor.build_model()
regressor.fit(train_x=X_train, train_y=y_train)
p_test = regressor.predict(X_test)
print('demo3-1 simple stack: ', mean_squared_error(y_test, p_test))
'''
demo3-2:单独为某些回归器包装KFolds_Training_Wrapper时,设置force_cv=False
'''
regressor = StackingRegressor(
    base_regressors=[
        KFolds_Regressor_Training_Wrapper(
            ElasticNetCV(where_store_regressor_model='./regressor_model/demo_3_2_layer_2_en_stack_cv.model'),
            k_fold=5),  # 仅该基回归器使用CV方式训练
        AdaBoostRegressor(where_store_regressor_model='./regressor_model/demo_3_2_layer_2_ada_stack_cv.model'),
        BaggingRegressor(where_store_regressor_model='./regressor_model/demo_3_2_layer_2_bag_stack_cv.model'),
        SVRRegressor(where_store_regressor_model='./regressor_model/demo_3_2_layer_2_svr_stack_cv.model'),
    ],
    meta_regressor=LinearRegression(where_store_regressor_model='./regressor_model/demo_3_2_layer_1_lr_stack_cv.model'),
)
regressor.build_model()
regressor.fit(train_x=X_train, train_y=y_train)
p_test = regressor.predict(X_test)
print('demo3-2 simple stack: ', mean_squared_error(y_test, p_test))
'''
demo3-3:StackingRegressor也可以作为基回归器使用,所以可以堆叠很深的结构
'''
regressor = StackingRegressor(
    base_regressors=[
        ElasticNetCV(where_store_regressor_model='./regressor_model/demo_3_3_layer_2_en_stack_cv.model'),
        AdaBoostRegressor(where_store_regressor_model='./regressor_model/demo_3_3_layer_2_ada_stack_cv.model'),
        BaggingRegressor(where_store_regressor_model='./regressor_model/demo_3_3_layer_2_bag_stack_cv.model'),
        SVRRegressor(where_store_regressor_model='./regressor_model/demo_3_3_layer_2_svr_stack_cv.model'),
        StackingRegressor(
            base_regressors=[
                LinearRegression(where_store_regressor_model='./regressor_model/demo_3_3_layer_3_lr_stack_cv.model'),
                ElasticNetCV(where_store_regressor_model='./regressor_model/demo_3_3_layer_3_en_stack_cv.model'),
            ],
            meta_regressor=GradientBoostingRegressor(
                where_store_regressor_model='./regressor_model/demo_3_3_layer_3_gbdt_stack_cv.model'),
        )
    ],
    meta_regressor=LinearRegression(
        where_store_regressor_model='./regressor_model/demo_3_3_layer_1_lr_stack_cv.model'),
)
regressor.build_model()
regressor.fit(train_x=X_train, train_y=y_train)
p_test = regressor.predict(X_test)
print('demo3-3 deep stack: ', mean_squared_error(y_test, p_test))

'''
demo3-4:StackingRegressor也可以被KFolds_Regressor_Training_Wrapper包装
'''
regressor = KFolds_Regressor_Training_Wrapper(StackingRegressor(
    base_regressors=[
        ElasticNetCV(where_store_regressor_model='./regressor_model/demo_3_4_layer_2_en_stack_cv.model'),
        AdaBoostRegressor(where_store_regressor_model='./regressor_model/demo_3_4_layer_2_ada_stack_cv.model'),
        BaggingRegressor(where_store_regressor_model='./regressor_model/demo_3_4_layer_2_bag_stack_cv.model'),
        SVRRegressor(where_store_regressor_model='./regressor_model/demo_3_4_layer_2_svr_stack_cv.model'),
        StackingRegressor(
            base_regressors=[
                LinearRegression(where_store_regressor_model='./regressor_model/demo_3_4_layer_3_lr_stack_cv.model'),
                ElasticNetCV(where_store_regressor_model='./regressor_model/demo_3_4_layer_3_en_stack_cv.model'),
            ],
            meta_regressor=GradientBoostingRegressor(
                where_store_regressor_model='./regressor_model/demo_3_4_layer_3_gbdt_stack_cv.model')
        )
    ],
    meta_regressor=LinearRegression(
        where_store_regressor_model='./regressor_model/demo_3_4_layer_1_lr_stack_cv.model'),
))
regressor.build_model()
regressor.fit(train_x=X_train, train_y=y_train)
p_test = regressor.predict(X_test)
print('demo3-4 deep deep stack: ', mean_squared_error(y_test, p_test))
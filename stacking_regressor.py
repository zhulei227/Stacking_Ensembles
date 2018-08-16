from sklearn.externals import joblib
from keras.models import Sequential, load_model
from keras.layers import Dense
import copy
import warnings

warnings.filterwarnings("ignore")
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 切换成CPU
from sklearn.model_selection import KFold
import numpy as np


class Regressor(object):
    '''
    定义回归器接口
    '''

    def __init__(self, where_store_regressor_model=None, train_params=None):
        """
        :param where_store_regressor_model:模型保存路径
        :param train_params: 训练参数
        """
        self.regressor_model_path = where_store_regressor_model
        self.train_params = {} if train_params is None else train_params

    def build_model(self):
        """
        创建模型
        :return:
        """
        raise RuntimeError("need to implement!")

    def fit(self, train_x, train_y):
        """
        拟合数据
        :return:
        """
        raise RuntimeError("need to implement!")

    def predict(self, test_x):
        """
        预测结果
        :param test_x:
        :return:
        """
        raise RuntimeError("need to implement!")

    def save_model(self):
        """
        存储模型
        :return:
        """
        raise RuntimeError("need to implement!")

    def load_model(self):
        """
        加载模型
        :return:
        """
        raise RuntimeError("need to implement!")


class SklearnRegressor(Regressor):
    """
    基于sklearn api的regressor实现
    """

    def __init__(self, where_store_regressor_model=None, train_params=None, regressor_class=None):
        Regressor.__init__(self, where_store_regressor_model, train_params)
        self.regressor_class = regressor_class

    def build_model(self):
        self.regressor_model = self.regressor_class(**self.train_params)

    def fit(self, train_x, train_y):
        self.regressor_model.fit(train_x, train_y)

    def predict(self, test_x):
        return self.regressor_model.predict(test_x)

    def save_model(self):
        joblib.dump(self.regressor_model, self.regressor_model_path)

    def load_model(self):
        self.regressor_model = joblib.load(self.regressor_model_path)


class DecisionTreeRegressor(SklearnRegressor):
    def __init__(self, where_store_regressor_model=None, train_params=None):
        from sklearn.tree import DecisionTreeRegressor
        SklearnRegressor.__init__(self, where_store_regressor_model, train_params, DecisionTreeRegressor)


class LinearRegression(SklearnRegressor):
    def __init__(self, where_store_regressor_model=None, train_params=None):
        from sklearn.linear_model import LinearRegression
        SklearnRegressor.__init__(self, where_store_regressor_model, train_params, LinearRegression)


class KNeighborsRegressor(SklearnRegressor):
    def __init__(self, where_store_regressor_model=None, train_params=None):
        from sklearn.neighbors import KNeighborsRegressor
        SklearnRegressor.__init__(self, where_store_regressor_model, train_params, KNeighborsRegressor)


class AdaBoostRegressor(SklearnRegressor):
    def __init__(self, where_store_regressor_model=None, train_params=None):
        from sklearn.ensemble import AdaBoostRegressor
        SklearnRegressor.__init__(self, where_store_regressor_model, train_params, AdaBoostRegressor)


class GradientBoostingRegressor(SklearnRegressor):
    def __init__(self, where_store_regressor_model=None, train_params=None):
        from sklearn.ensemble import GradientBoostingRegressor
        SklearnRegressor.__init__(self, where_store_regressor_model, train_params, GradientBoostingRegressor)


class BaggingRegressor(SklearnRegressor):
    def __init__(self, where_store_regressor_model=None, train_params=None):
        from sklearn.ensemble import BaggingRegressor
        SklearnRegressor.__init__(self, where_store_regressor_model, train_params, BaggingRegressor)


class ExtraTreeRegressor(SklearnRegressor):
    def __init__(self, where_store_regressor_model=None, train_params=None):
        from sklearn.tree import ExtraTreeRegressor
        SklearnRegressor.__init__(self, where_store_regressor_model, train_params, ExtraTreeRegressor)


class SVRRegressor(SklearnRegressor):
    def __init__(self, where_store_regressor_model=None, train_params=None):
        from sklearn.svm import SVR
        SklearnRegressor.__init__(self, where_store_regressor_model, train_params, SVR)


class LinearSVR(SklearnRegressor):
    def __init__(self, where_store_regressor_model=None, train_params=None):
        from sklearn.svm import LinearSVR
        SklearnRegressor.__init__(self, where_store_regressor_model, train_params, LinearSVR)


class ElasticNet(SklearnRegressor):
    def __init__(self, where_store_regressor_model=None, train_params=None):
        from sklearn.linear_model import ElasticNet
        SklearnRegressor.__init__(self, where_store_regressor_model, train_params, ElasticNet)


class ElasticNetCV(SklearnRegressor):
    def __init__(self, where_store_regressor_model=None, train_params=None):
        from sklearn.linear_model import ElasticNetCV
        SklearnRegressor.__init__(self, where_store_regressor_model, train_params, ElasticNetCV)


class BayesianRidge(SklearnRegressor):
    def __init__(self, where_store_regressor_model=None, train_params=None):
        from sklearn.linear_model import BayesianRidge
        SklearnRegressor.__init__(self, where_store_regressor_model, train_params, BayesianRidge)


'''
DNN回归模型,该部分利用keras简单实现MLP回归
'''


class SimpleMLPRegressor(Regressor):
    def __init__(self, where_store_regressor_model=None, train_params=None):
        """
        :param where_store_regressor_model:
        :param train_params:
        """
        Regressor.__init__(self, where_store_regressor_model, train_params)
        self._check_params()

    def _check_params(self):
        if 'input_num' not in self.train_params:
            raise RuntimeError('no input_num param in train_params!')
        if 'batch_size' not in self.train_params:
            self.train_params['batch_size'] = 64
        if 'epochs' not in self.train_params:
            self.train_params['epochs'] = 5
        if 'shuffle' not in self.train_params:
            self.train_params['shuffle'] = True
        if 'validation_split' not in self.train_params:
            self.train_params['validation_split'] = 0.1

    def build_model(self):
        self.regressor_model = Sequential()
        self.regressor_model.add(
            Dense(10, input_dim=self.train_params['input_num'], init='normal', activation='linear'))
        self.regressor_model.add(Dense(1))
        self.regressor_model.compile(loss='mse', optimizer='adam')

    def fit(self, train_x, train_y):
        self.regressor_model.fit(x=train_x, y=train_y,
                                 batch_size=self.train_params['batch_size'], epochs=self.train_params['epochs'],
                                 validation_split=self.train_params['validation_split'],
                                 shuffle=self.train_params['shuffle'],
                                 verbose=False)

    def predict(self, test_x):
        return self.regressor_model.predict(test_x, batch_size=test_x.shape[0])

    def save_model(self):
        self.regressor_model.save(self.regressor_model_path)

    def load_model(self):
        self.regressor_model = load_model(self.regressor_model_path)


class KFolds_Regressor_Training_Wrapper(Regressor):
    '''
    对训练的回归器进行交叉式训练，是对原始回归器的扩展，可独立使用
    '''

    def __init__(self, base_regressor=None, k_fold=5):
        """

        :param base_regressor:
        :param k_fold:
        """
        Regressor.__init__(self)
        self.base_regressor = base_regressor
        self.k_fold = k_fold
        self._suffix_for_cv = None  # 用于再次被KFolds_Training_Wrapper包装

    def _append_model_path(self):
        self.extend_regressors_path_list = ['_cv' + str(index) for index in range(0, self.k_fold)]

    def build_model(self):
        """
        创建模型
        :return:
        """
        self._append_model_path()
        self.extend_regressors = []
        for append_path in self.extend_regressors_path_list:
            new_regressor = copy.deepcopy(self.base_regressor)
            if new_regressor.regressor_model_path is None:
                new_regressor._suffix_for_cv = append_path + (
                    self._suffix_for_cv if self._suffix_for_cv is not None else '')
            else:
                new_regressor.regressor_model_path = self.base_regressor.regressor_model_path + append_path + (
                    self._suffix_for_cv if self._suffix_for_cv is not None else '')
            new_regressor.build_model()
            self.extend_regressors.append(new_regressor)

    def fit(self, train_x, train_y):
        """
        拟合数据:切分数据并训练
        :return:
        """
        self.train_x = train_x
        self.train_y = train_y
        kf = KFold(n_splits=self.k_fold, shuffle=False, random_state=227)
        index = 0
        for train_index, _ in kf.split(train_x):
            X_train = train_x[train_index]
            y_train = train_y[train_index]
            self.extend_regressors[index].fit(X_train, y_train)
            index += 1

    def _extract_k_fold_data_features(self):
        """
        抽取每个回归器的预测结果,并组合
        :return:
        """
        regression_results = []
        kf = KFold(n_splits=self.k_fold, shuffle=False, random_state=227)
        kf.get_n_splits(self.train_x)
        index = 0
        for _, test_index in kf.split(self.train_x):
            X_test = self.train_x[test_index]
            regression_results.append(self.extend_regressors[index].predict(X_test))
            index += 1
        return np.concatenate(regression_results, axis=0)

    def predict(self, test_x):
        """
        预测
        :param test_x:
        :return:
        """
        regression_result = self.extend_regressors[0].predict(test_x)
        for regressor_id in range(1, len(self.extend_regressors)):
            regression_result += self.extend_regressors[regressor_id].predict(test_x)
        return regression_result / (1.0 * self.k_fold)

    def save_model(self):
        """
        存储模型
        :return:
        """
        for regressor in self.extend_regressors:
            regressor.save_model()

    def load_model(self):
        """
        加载模型
        :return:
        """
        self._append_model_path()
        self.extend_regressors = []
        for append_path in self.extend_regressors_path_list:
            new_regressor = copy.deepcopy(self.base_regressor)
            if new_regressor.regressor_model_path is None:
                new_regressor._suffix_for_cv = append_path + (
                    self._suffix_for_cv if self._suffix_for_cv is not None else '')
            else:
                new_regressor.regressor_model_path = self.base_regressor.regressor_model_path + append_path + (
                    self._suffix_for_cv if self._suffix_for_cv is not None else '')
            new_regressor.load_model()
            self.extend_regressors.append(new_regressor)


class StackingRegressor(Regressor):
    def __init__(self, base_regressors=list(), meta_regressor=None, force_cv=True):
        """
        为cv训练方式提供更好的支持

        :param regressors: 回归器
        :param meta_regressor: 元回归器(基于基回归器的结果再次训练)
        :param force_cv 是否强制使用cv的方式训练所有基回归器以及元回归器(建议直接True),如果基回归器和未被KFolds_Regreesor_Training_Warpper包装,会被强制包装一次
        """
        Regressor.__init__(self)
        self.base_regressors = base_regressors
        self.meta_regressor = meta_regressor
        self.meta_train_x = None
        self.meta_train_y = None
        self.force_cv = force_cv
        self._suffix_for_cv = None  # 被KFolds_Regressor_Training_Warpper包装时,存放添加的后缀
        if self.force_cv:
            for index in range(0, len(self.base_regressors)):
                if not isinstance(self.base_regressors[index], KFolds_Regressor_Training_Wrapper):
                    self.base_regressors[index] = KFolds_Regressor_Training_Wrapper(self.base_regressors[index])
            if not isinstance(self.meta_regressor, KFolds_Regressor_Training_Wrapper):
                self.meta_regressor = KFolds_Regressor_Training_Wrapper(self.meta_regressor)

    def _build_base_regressor_models(self):
        """
        构建基回归器
        :return:
        """
        for regressor in self.base_regressors:
            if regressor.regressor_model_path is not None:
                regressor.regressor_model_path += (self._suffix_for_cv if self._suffix_for_cv is not None else '')
            else:
                regressor._suffix_for_cv = self._suffix_for_cv
            regressor.build_model()

    def _build_meta_regressor_model(self):
        """
        构建元回归器
        :return:
        """
        if self.meta_regressor.regressor_model_path is not None:
            self.meta_regressor.regressor_model_path += (
                self._suffix_for_cv if self._suffix_for_cv is not None else '')
        else:
            self.meta_regressor._suffix_for_cv = self._suffix_for_cv
        self.meta_regressor.build_model()

    def build_model(self):
        """
        构建全部回归器
        :return:
        """
        self._build_base_regressor_models()
        self._build_meta_regressor_model()

    def _fit_base_regressors(self, train_x, train_y):
        """
        训练基回归器
        :return:
        """
        for regressor in self.base_regressors:
            regressor.fit(train_x, train_y)

    def _fit_meta_regressor(self):
        """
        训练元回归器
        :return:

        """
        self.meta_regressor.fit(self.meta_train_x, self.meta_train_y)

    def fit(self, train_x, train_y):
        """
        训练全部回归器
        :param train_x:
        :param train_y:
        :return:
        """
        self._fit_base_regressors(train_x, train_y)
        self.meta_train_x = self._get_base_regressors_training_data(train_x)
        self.meta_train_y = train_y
        self._fit_meta_regressor()

    def _get_base_regressors_training_data(self, train_x):
        """
        获取基回归器的训练数据
        :return:
        """
        _all_regression_results = []
        for regressor in self.base_regressors:
            try:
                current_regressor_result = regressor._extract_k_fold_data_features()  # 使用KFolds_Regressor_Training_wrapper包装过的回归器会调用该api
            except:
                current_regressor_result = regressor.predict(train_x)
            _all_regression_results.append(current_regressor_result.reshape(-1,1))
        return np.concatenate(_all_regression_results, axis=-1)

    def _combine_base_regressor_predict(self, test_x=None):
        """
        基回归器预测结果
        :param test_x:
        :return:
        """
        _all_regression_results = [(regressor.predict(test_x)).reshape(-1,1) for regressor in self.base_regressors]
        return np.concatenate(_all_regression_results, axis=-1)

    def predict(self, test_x):
        """
        预测结果
        :param test_x:
        :return:
        """
        return self.meta_regressor.predict(self._combine_base_regressor_predict(test_x))

    def _save_base_regressor_models(self):
        """
        保存基回归器
        :return:
        """
        for regressor in self.base_regressors:
            regressor.save_model()

    def _save_meta_regressor_model(self):
        """
        保存元回归器
        :return:
        """
        self.meta_regressor.save_model()

    def save_model(self):
        """
        保存所有回归器
        :return:
        """
        self._save_base_regressor_models()
        self._save_meta_regressor_model()

    def _load_base_regressor_models(self):
        """
        加载基回归器
        :return:

        base_regressor should like XXXRegressor(where_store_regressor_model='/.../')
        """
        for regressor in self.base_regressors:
            if regressor.regressor_model_path is None:
                regressor._suffix_for_cv = self._suffix_for_cv if self._suffix_for_cv is not None else ''
            regressor.load_model()

    def _load_meta_regressor_model(self):
        """
        加载元回归器
        :return:

        meta_regressor should like XXXregressor(where_store_regressor_model='/.../')
        """
        if self.meta_regressor.regressor_model_path is None:
            self.meta_regressor._suffix_for_cv = self._suffix_for_cv if self._suffix_for_cv is not None else ''
        self.meta_regressor.load_model()

    def load_model(self):
        """
        加载模型
        所有基回归器以及元回归器必须指定存储路径
        :return:
        """
        self._load_base_regressor_models()
        self._load_meta_regressor_model()

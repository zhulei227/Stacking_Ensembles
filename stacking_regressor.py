import pickle
import copy
import numpy as np
from sklearn.model_selection import KFold
import warnings

warnings.filterwarnings("ignore")


class Regressor(object):
    '''
    定义回归器接口
    '''

    def __init__(self, train_params=None):
        """
        :param train_params: 训练参数
        """
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

    def save_model(self, model_path):
        """
        存储模型
        :return:
        """
        with open(model_path, 'wb') as model_file:
            pickle.dump(self, model_file)

    @staticmethod
    def load_model(model_path):
        """
        加载模型
        :return:
        """
        with open(model_path, 'rb') as model_file:
            new_model = pickle.load(model_file)
        return new_model


class SklearnRegressor(Regressor):
    """
    基于sklearn api的regressor实现
    """

    def __init__(self, train_params=None, regressor_class=None):
        Regressor.__init__(self, train_params)
        self.regressor_class = regressor_class

    def build_model(self):
        self.regressor_model = self.regressor_class(**self.train_params)

    def fit(self, train_x, train_y):
        self.regressor_model.fit(train_x, train_y)

    def predict(self, test_x):
        return self.regressor_model.predict(test_x)


class DecisionTreeRegressor(SklearnRegressor):
    def __init__(self, train_params=None):
        from sklearn.tree import DecisionTreeRegressor
        SklearnRegressor.__init__(self, train_params, DecisionTreeRegressor)


class LinearRegression(SklearnRegressor):
    def __init__(self, train_params=None):
        from sklearn.linear_model import LinearRegression
        SklearnRegressor.__init__(self, train_params, LinearRegression)


class KNeighborsRegressor(SklearnRegressor):
    def __init__(self, train_params=None):
        from sklearn.neighbors import KNeighborsRegressor
        SklearnRegressor.__init__(self, train_params, KNeighborsRegressor)


class AdaBoostRegressor(SklearnRegressor):
    def __init__(self, train_params=None):
        from sklearn.ensemble import AdaBoostRegressor
        SklearnRegressor.__init__(self, train_params, AdaBoostRegressor)


class GradientBoostingRegressor(SklearnRegressor):
    def __init__(self, train_params=None):
        from sklearn.ensemble import GradientBoostingRegressor
        SklearnRegressor.__init__(self, train_params, GradientBoostingRegressor)


class BaggingRegressor(SklearnRegressor):
    def __init__(self, train_params=None):
        from sklearn.ensemble import BaggingRegressor
        SklearnRegressor.__init__(self, train_params, BaggingRegressor)


class ExtraTreeRegressor(SklearnRegressor):
    def __init__(self, train_params=None):
        from sklearn.tree import ExtraTreeRegressor
        SklearnRegressor.__init__(self, train_params, ExtraTreeRegressor)


class SVRRegressor(SklearnRegressor):
    def __init__(self, train_params=None):
        from sklearn.svm import SVR
        SklearnRegressor.__init__(self, train_params, SVR)


class LinearSVR(SklearnRegressor):
    def __init__(self, train_params=None):
        from sklearn.svm import LinearSVR
        SklearnRegressor.__init__(self, train_params, LinearSVR)


class ElasticNet(SklearnRegressor):
    def __init__(self, train_params=None):
        from sklearn.linear_model import ElasticNet
        SklearnRegressor.__init__(self, train_params, ElasticNet)


class ElasticNetCV(SklearnRegressor):
    def __init__(self, train_params=None):
        from sklearn.linear_model import ElasticNetCV
        SklearnRegressor.__init__(self, train_params, ElasticNetCV)


class BayesianRidge(SklearnRegressor):
    def __init__(self, train_params=None):
        from sklearn.linear_model import BayesianRidge
        SklearnRegressor.__init__(self, train_params, BayesianRidge)


class Lasso(SklearnRegressor):
    def __init__(self, train_params=None):
        from sklearn.linear_model import Lasso
        SklearnRegressor.__init__(self, train_params, Lasso)


class KFolds_Regressor_Training_Wrapper(Regressor):
    '''
    对训练的回归器进行交叉式训练，是对原始回归器的扩展，可独立使用
    '''

    def __init__(self, base_regressor=None, k_fold=5,random_state=42):
        """

        :param base_regressor:
        :param k_fold:
        """
        Regressor.__init__(self)
        self.base_regressor = base_regressor
        self.k_fold = k_fold
        self.random_state=random_state

    def build_model(self):
        """
        创建模型
        :return:
        """
        self.extend_regressors = []
        for _ in range(0, self.k_fold):
            new_regressor = copy.deepcopy(self.base_regressor)
            new_regressor.build_model()
            self.extend_regressors.append(new_regressor)

    def fit(self, train_x, train_y):
        """
        拟合数据:切分数据并训练
        :return:
        """
        self.train_x = train_x
        self.train_y = train_y
        kf = KFold(n_splits=self.k_fold, shuffle=False, random_state=self.random_state)
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
        kf = KFold(n_splits=self.k_fold, shuffle=False, random_state=self.random_state)
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


class StackingRegressor(Regressor):
    def __init__(self, base_regressors=list(), meta_regressor=None, force_cv=True, base_k_fold=5, meta_k_fold=5):
        """
        为cv训练方式提供更好的支持

        :param regressors: 回归器
        :param meta_regressor: 元回归器(基于基回归器的结果再次训练)
        :param force_cv 是否强制使用cv的方式训练所有基回归器以及元回归器(建议直接True),如果基回归器和未被KFolds_Regreesor_Training_Warpper包装,会被强制包装一次
        :param base_k_fold:基回归器的k_fold
        :param meta_k_fold:元回归器的k_fold
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
                    self.base_regressors[index] = KFolds_Regressor_Training_Wrapper(self.base_regressors[index],
                                                                                    k_fold=base_k_fold)
            if not isinstance(self.meta_regressor, KFolds_Regressor_Training_Wrapper):
                self.meta_regressor = KFolds_Regressor_Training_Wrapper(self.meta_regressor, k_fold=meta_k_fold)

    def _build_base_regressor_models(self):
        """
        构建基回归器
        :return:
        """
        for regressor in self.base_regressors:
            regressor.build_model()

    def _build_meta_regressor_model(self):
        """
        构建元回归器
        :return:
        """
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
            _all_regression_results.append(current_regressor_result.reshape(-1, 1))
        return np.concatenate(_all_regression_results, axis=-1)

    def _combine_base_regressor_predict(self, test_x=None):
        """
        基回归器预测结果
        :param test_x:
        :return:
        """
        _all_regression_results = [(regressor.predict(test_x)).reshape(-1, 1) for regressor in self.base_regressors]
        return np.concatenate(_all_regression_results, axis=-1)

    def predict(self, test_x):
        """
        预测结果
        :param test_x:
        :return:
        """
        return self.meta_regressor.predict(self._combine_base_regressor_predict(test_x))

from sklearn.model_selection import KFold, train_test_split
from multiprocessing import Pool, cpu_count
import threading
import numpy as np
import pickle
import copy
import random
import platform
import warnings

warnings.filterwarnings("ignore")

'''
常用函数
'''

'''
类别标签转one-hot
'''


def to_categorical(y, num_classes=None, dtype='float32'):
    # copy from keras
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


'''
保证输入数据类型为numpy
'''


def force2ndarray(fn):
    def clean_data(*args, **kwargs):
        if len(kwargs) != 0:
            append_args = []
            keys = kwargs.keys()
            if 'train_x' in keys:
                append_args.append(kwargs['train_x'])
            if 'test_x' in keys:
                append_args.append(kwargs['test_x'])
            if 'train_y' in keys:
                append_args.append(kwargs['train_y'])
            args += tuple(append_args)

        if args[1].__class__.__name__ == 'DataFrame':
            inputs_0 = args[1].values
        elif args[1].__class__.__name__ == 'list':
            inputs_0 = np.asarray(args[1])
        elif args[1].__class__.__name__ == 'ndarray':
            inputs_0 = args[1]
        else:
            raise RuntimeError('未知数据类型:', args[1].__class__.__name__)

        if len(args) == 3:
            if args[2].__class__.__name__ == 'Series':
                inputs_1 = args[2].values
            elif args[2].__class__.__name__ == 'list':
                inputs_1 = np.asarray(args[2])
            elif args[2].__class__.__name__ == 'ndarray':
                inputs_1 = args[2]
            else:
                raise RuntimeError('未知数据类型:', args[2].__class__.__name__)
        if len(args) == 2:
            return fn(args[0], inputs_0)
        else:
            return fn(args[0], inputs_0, inputs_1)

    return clean_data


"""
分类器接口
"""


class Classifier(object):
    """
    定义分类器接口
    """

    def __init__(self, train_params=None, subsample_features_rate=None, subsample_features_indices=None,
                 categorical_feature_indices=None, n_jobs=1):
        """
        :param train_params: 训练参数
        """
        self.train_params = {} if train_params is None else train_params
        self.subsample_features_rate = subsample_features_rate
        self.subsample_features_indices = subsample_features_indices
        self.categorical_feature_indices = categorical_feature_indices
        self.n_jobs = n_jobs

    def reshape_features(self, features):
        """
        读取features指定列用于训练或者随机选择某几列训练
        :param features:
        :return:
        """
        _, columns = features.shape
        indices = list(range(0, columns))
        # 默认会排除字符串变量
        no_categorical_feature_indices = []
        if self.categorical_feature_indices is not None:
            for index in indices:
                if index not in self.categorical_feature_indices:
                    no_categorical_feature_indices.append(index)
        else:
            no_categorical_feature_indices = indices

        if self.subsample_features_indices is None and self.subsample_features_rate is not None:
            random.shuffle(no_categorical_feature_indices)
            self.subsample_features_indices = no_categorical_feature_indices[
                                              :int(len(no_categorical_feature_indices) * self.subsample_features_rate)]
        if self.subsample_features_indices is not None:
            return features[:, self.subsample_features_indices]
        return features[:, no_categorical_feature_indices]

    @staticmethod
    def update_params(current_classifier, subsample_features_rate, subsample_features_indices,
                      categorical_feature_indices):
        '''
        递归向下更新参数
        :return:
        '''
        if current_classifier.subsample_features_rate is None:
            current_classifier.subsample_features_rate = subsample_features_rate
        if current_classifier.subsample_features_indices is None:
            current_classifier.subsample_features_indices = subsample_features_indices
        if current_classifier.categorical_feature_indices is None:
            current_classifier.categorical_feature_indices = categorical_feature_indices

        if current_classifier.__class__.__name__ == 'KFolds_Classifier_Training_Wrapper':
            Classifier.update_params(current_classifier.base_classifier, current_classifier.subsample_features_rate,
                                     current_classifier.subsample_features_indices,
                                     current_classifier.categorical_feature_indices)
        if current_classifier.__class__.__name__ == 'StackingClassifier':
            for base_classifier in current_classifier.base_classifiers:
                Classifier.update_params(base_classifier, current_classifier.subsample_features_rate,
                                         current_classifier.subsample_features_indices,
                                         current_classifier.categorical_feature_indices)

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
        预测标签
        :param test_x:
        :return:
        """
        raise RuntimeError("need to implement!")

    def predict_categorical(self, test_x):
        """
        预测标签分布
        :param test_x:
        :return:[0,0,1,0,...]
        """
        raise RuntimeError("need to implement!")

    def predict_proba(self, test_x):
        """
        预测标签概率(分布)
        :param test_x:
        :return:
        """

    def predict_categorical_proba(self, test_x):
        """
        预测标签概率分布
        :param test_x:
        :return:
        """

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


class SklearnClassifier(Classifier):
    """
    基于sklearn api的classifier实现
    """

    def __init__(self, train_params=None, classifier_class=None, subsample_features_rate=None,
                 subsample_features_indices=None, categorical_feature_indices=None, n_jobs=1):
        Classifier.__init__(self, train_params, subsample_features_rate, subsample_features_indices,
                            categorical_feature_indices, n_jobs)
        self.classifier_class = classifier_class

    def build_model(self):
        self.classifier_model = self.classifier_class(**self.train_params)

    @force2ndarray
    def fit(self, train_x, train_y):

        self.class_num = len(set(train_y))
        self.classifier_model.fit(self.reshape_features(train_x).astype('float64'), train_y)

    @force2ndarray
    def predict(self, test_x):
        return self.classifier_model.predict(self.reshape_features(test_x))

    @force2ndarray
    def predict_categorical(self, test_x):
        return to_categorical(self.predict(test_x), self.class_num)

    @force2ndarray
    def predict_proba(self, test_x):
        return self.classifier_model.predict_proba(self.reshape_features(test_x).astype('float64'))

    @force2ndarray
    def predict_categorical_proba(self, test_x):
        probas = self.classifier_model.predict_proba(self.reshape_features(test_x).astype('float64'))
        _, col = probas.shape
        if col > 1:
            return probas
        else:
            return np.asarray([[1 - proba, proba] for proba in probas])


class SVMClassifier(SklearnClassifier):
    def __init__(self, train_params=None, subsample_features_rate=None, subsample_features_indices=None,
                 categorical_feature_indices=None, n_jobs=1):
        from sklearn.svm import SVC
        if train_params is None:
            train_params = {'probability': True}
        else:
            train_params['probability'] = True
        SklearnClassifier.__init__(self, train_params, SVC, subsample_features_rate, subsample_features_indices,
                                   categorical_feature_indices, n_jobs)


class RandomForestClassifier(SklearnClassifier):
    def __init__(self, train_params=None, subsample_features_rate=None, subsample_features_indices=None,
                 categorical_feature_indices=None, n_jobs=1):
        from sklearn.ensemble import RandomForestClassifier
        SklearnClassifier.__init__(self, train_params, RandomForestClassifier, subsample_features_rate,
                                   subsample_features_indices, categorical_feature_indices, n_jobs)


class GradientBoostingClassifier(SklearnClassifier):
    def __init__(self, train_params=None, subsample_features_rate=None, subsample_features_indices=None,
                 categorical_feature_indices=None, n_jobs=1):
        from sklearn.ensemble import GradientBoostingClassifier
        SklearnClassifier.__init__(self, train_params, GradientBoostingClassifier, subsample_features_rate,
                                   subsample_features_indices, categorical_feature_indices, n_jobs)


class AdaBoostClassifier(SklearnClassifier):
    def __init__(self, train_params=None, subsample_features_rate=None, subsample_features_indices=None,
                 categorical_feature_indices=None, n_jobs=1):
        from sklearn.ensemble import AdaBoostClassifier
        SklearnClassifier.__init__(self, train_params, AdaBoostClassifier, subsample_features_rate,
                                   subsample_features_indices, categorical_feature_indices, n_jobs)


class BaggingClassifier(SklearnClassifier):
    def __init__(self, train_params=None, subsample_features_rate=None, subsample_features_indices=None,
                 categorical_feature_indices=None, n_jobs=1):
        from sklearn.ensemble import BaggingClassifier
        SklearnClassifier.__init__(self, train_params, BaggingClassifier, subsample_features_rate,
                                   subsample_features_indices, categorical_feature_indices, n_jobs)


class LogisticRegression(SklearnClassifier):
    def __init__(self, train_params=None, subsample_features_rate=None, subsample_features_indices=None,
                 categorical_feature_indices=None, n_jobs=1):
        from sklearn.linear_model import LogisticRegression
        SklearnClassifier.__init__(self, train_params, LogisticRegression, subsample_features_rate,
                                   subsample_features_indices, categorical_feature_indices, n_jobs)


class NaiveBayesClassifier(SklearnClassifier):
    def __init__(self, train_params=None, subsample_features_rate=None, subsample_features_indices=None,
                 categorical_feature_indices=None, n_jobs=1):
        from sklearn.naive_bayes import GaussianNB
        SklearnClassifier.__init__(self, train_params, GaussianNB, subsample_features_rate, subsample_features_indices,
                                   categorical_feature_indices, n_jobs)


class KFolds_Classifier_Training_Wrapper(Classifier):
    '''
    对训练的分类器进行交叉式训练，是对原始分类器的扩展，可独立使用
    '''

    def __init__(self, base_classifer=None, k_fold=5, random_state=42, subsample_features_rate=None,
                 subsample_features_indices=None, categorical_feature_indices=None, n_jobs=1):
        """

        :param base_classifer:
        :param k_fold:
        """
        Classifier.__init__(self)
        self.base_classifier = base_classifer
        self.k_fold = k_fold
        self.random_state = random_state
        self.n_jobs = n_jobs
        # subsample_features_rate,subsample_features_indices,categorical_feature_indices参数向下递归传递给具体的base_classifiers
        Classifier.update_params(self, subsample_features_rate, subsample_features_indices, categorical_feature_indices)

    def build_model(self):
        """
        创建模型
        :return:
        """
        self.extend_classifiers = []
        for _ in range(0, self.k_fold):
            new_classifier = copy.deepcopy(self.base_classifier)
            new_classifier.build_model()
            self.extend_classifiers.append(new_classifier)

    @force2ndarray
    def fit(self, train_x, train_y):
        """
        :param train_x: 训练特征
        :param train_y: 训练标签
        :return:
        """
        if self.n_jobs not in [None, 0, 1]:
            # 并行训练
            mpt = MultiProcessTrainer(self.n_jobs)
            mpt.build_trainer_tree(self, train_x, train_y)
            mpt.fit()
        else:
            kf = KFold(n_splits=self.k_fold, shuffle=False, random_state=self.random_state)
            index = 0
            for train_index, _ in kf.split(train_x):
                X_train = train_x[train_index]
                y_train = train_y[train_index]
                self.extend_classifiers[index].fit(X_train, y_train)
                index += 1

    @force2ndarray
    def extract_k_fold_data_catogorical_features(self, train_x):
        """
        抽取交叉分割数据后的标签分布预测结果
        :return:
        """
        catogorical_results = []
        kf = KFold(n_splits=self.k_fold, shuffle=False, random_state=self.random_state)
        kf.get_n_splits(train_x)
        index = 0
        for _, test_index in kf.split(train_x):
            X_test = train_x[test_index]
            catogorical_results.append(self.extend_classifiers[index].predict_categorical(X_test))
            index += 1
        return np.concatenate(catogorical_results, axis=0)

    @force2ndarray
    def extract_k_fold_data_catogorical_proba_features(self, train_x):
        """
        抽取交叉分割数据后的标签概率分布预测结果
        :return:
        """
        catogorical_proba_results = []
        kf = KFold(n_splits=self.k_fold, shuffle=False, random_state=self.random_state)
        index = 0
        for _, test_index in kf.split(train_x):
            X_test = train_x[test_index]
            catogorical_proba_results.append(
                self.extend_classifiers[index].predict_categorical_proba(X_test))
            index += 1
        return np.concatenate(catogorical_proba_results, axis=0)

    @force2ndarray
    def predict(self, test_x):
        """
        预测标签
        :param test_x:
        :return:
        """
        categorical_result = self.extend_classifiers[0].predict_categorical(test_x)
        for classifier_id in range(1, len(self.extend_classifiers)):
            categorical_result += self.extend_classifiers[classifier_id].predict_categorical(test_x)
        new_result = []
        for current_index in range(0, len(categorical_result)):
            current_row = categorical_result[current_index].tolist()
            maxvalue_index = current_row.index(max(current_row))
            new_result.append(maxvalue_index)
        return new_result

    @force2ndarray
    def predict_categorical(self, test_x):
        """
        预测标签分布
        :param test_x:
        :return:[0,0,1,0,...]
        """
        categorical_result = self.extend_classifiers[0].predict_categorical(test_x)
        for classifier_id in range(1, len(self.extend_classifiers)):
            categorical_result += self.extend_classifiers[classifier_id].predict_categorical(test_x)
        new_categorical_result = np.zeros(shape=categorical_result.shape, dtype=int)
        for current_index in range(0, len(categorical_result)):
            current_row = categorical_result[current_index].tolist()
            maxvalue_index = current_row.index(max(current_row))
            new_categorical_result[current_index][maxvalue_index] = 1
        return new_categorical_result

    @force2ndarray
    def predict_proba(self, test_x):
        """
        预测标签概率(分布)
        :param test_x:
        :return:
        """
        proba_result = self.extend_classifiers[0].predict_proba(test_x)
        for classifier_id in range(1, len(self.extend_classifiers)):
            proba_result += self.extend_classifiers[classifier_id].predict_proba(test_x)
        return proba_result / (len(self.extend_classifiers) * 1.0)

    @force2ndarray
    def predict_categorical_proba(self, test_x):
        """
        预测标签概率分布
        :param test_x:
        :return:
        """
        categorical_proba_result = self.extend_classifiers[0].predict_categorical_proba(test_x)
        for classifier_id in range(1, len(self.extend_classifiers)):
            categorical_proba_result += self.extend_classifiers[classifier_id].predict_categorical_proba(test_x)
        return categorical_proba_result / (len(self.extend_classifiers) * 1.0)


class StackingClassifier(Classifier):
    def __init__(self, base_classifiers=list(), meta_classifier=None, use_probas=True, force_cv=True, base_k_fold=5,
                 meta_k_fold=5, subsample_features_rate=None, subsample_features_indices=None,
                 categorical_feature_indices=None, n_jobs=1):
        """
        为cv训练方式提供更好的支持

        :param base_classifiers: 基分类器列表
        :param meta_classifier: 元分类器(对基分类器的预测结果再次训练)
        :param use_probas: 基于基分类器的概率预测分布训练(默认使用类别标签的分布)
        :param force_cv 是否强制使用cv的方式训练所有基分类器以及元分类器(建议直接True),如果基分类器和未被KFolds_Training_Warpper包装,会被强制包装一次
        :param base_k_fold:包装基分类器的k_fold
        :param meta_k_fold:包装元分类器的k_fold
        """
        Classifier.__init__(self)
        self.base_classifiers = base_classifiers
        self.meta_classifier = meta_classifier
        self.use_probas = use_probas
        self.n_jobs = n_jobs
        self.force_cv = force_cv
        if self.force_cv:
            for index in range(0, len(self.base_classifiers)):
                if not isinstance(self.base_classifiers[index], KFolds_Classifier_Training_Wrapper):
                    self.base_classifiers[index] = KFolds_Classifier_Training_Wrapper(self.base_classifiers[index],
                                                                                      k_fold=base_k_fold)
            if not isinstance(self.meta_classifier, KFolds_Classifier_Training_Wrapper):
                self.meta_classifier = KFolds_Classifier_Training_Wrapper(self.meta_classifier, k_fold=meta_k_fold)

        # subsample_features_rate,subsample_features_indices,categorical_feature_indices参数向下递归传递给具体的base_classifiers
        Classifier.update_params(self, subsample_features_rate, subsample_features_indices, categorical_feature_indices)

    def build_model(self):
        """
        构建全部分类器
        :return:
        """
        for classifier in self.base_classifiers:
            classifier.build_model()
        self.meta_classifier.build_model()

    @force2ndarray
    def fit(self, train_x, train_y):
        """
        训练全部分类器
        :param train_x:
        :param train_y:
        :return:
        """
        if self.n_jobs not in [None, 0, 1]:
            # 并行训练
            mpt = MultiProcessTrainer(self.n_jobs)
            mpt.build_trainer_tree(self, train_x, train_y)
            mpt.fit()
        else:
            for classifier in self.base_classifiers:
                classifier.fit(train_x, train_y)

            if self.use_probas:
                meta_train_x = self.get_base_classifier_training_categorical_proba(train_x)
            else:
                meta_train_x = self.get_base_classifier_training_categorical(train_x)

            self.meta_classifier.fit(meta_train_x, train_y)

    @force2ndarray
    def get_base_classifier_training_categorical_proba(self, train_x):
        """
        获取基分类器的训练数据
        :return:
        """
        _all_categorical_probas = []
        for classifier in self.base_classifiers:
            try:
                current_category_labels = classifier.extract_k_fold_data_catogorical_proba_features(
                    train_x)  # 使用KFolds_Training_wrapper包装过的分类器会调用该api
            except:
                current_category_labels = classifier.predict_categorical_proba(train_x)
            _all_categorical_probas.append(current_category_labels)
        return np.concatenate(_all_categorical_probas, axis=-1)

    @force2ndarray
    def get_base_classifier_training_categorical(self, train_x):
        """
        获取基分类器的训练数据
        :return:
        """
        _all_categorical_labels = []
        for classifier in self.base_classifiers:
            try:
                current_category_labels = classifier.extract_k_fold_data_catogorical_features(
                    train_x)  # 使用KFolds_Training_wrapper包装过的分类器会调用该api
            except:
                current_category_labels = classifier.predict_categorical(train_x)
            _all_categorical_labels.append(current_category_labels)
        return np.concatenate(_all_categorical_labels, axis=-1)

    @force2ndarray
    def combine_base_classifier_predict_categorical(self, test_x=None):
        """
        基分类器预测标签分布的组合
        :param test_x:
        :return:
        """
        _all_categorical_labels = [classifier.predict_categorical(test_x) for classifier in self.base_classifiers]
        return np.concatenate(_all_categorical_labels, axis=-1)

    @force2ndarray
    def combine_base_classifier_predict_categorical_proba(self, test_x=None):
        """
        基分类器预测标签概率分布的组合
        :param test_x:
        :return:
        """
        _all_categorical_probas = [classifier.predict_categorical_proba(test_x) for classifier in self.base_classifiers]
        return np.concatenate(_all_categorical_probas, axis=-1)

    @force2ndarray
    def predict(self, test_x):
        """
        预测标签
        :param test_x:
        :return:
        """
        return self.meta_classifier.predict(self.combine_base_classifier_predict_categorical_proba(
            test_x)) if self.use_probas else self.meta_classifier.predict(
            self.combine_base_classifier_predict_categorical(test_x))

    @force2ndarray
    def predict_categorical(self, test_x):
        """
        预测标签分布
        :param test_x:
        :return:[0,0,1,0,...]
        """
        return self.meta_classifier.predict_categorical(self.combine_base_classifier_predict_categorical_proba(
            test_x)) if self.use_probas else self.meta_classifier.predict_categorical(
            self.combine_base_classifier_predict_categorical(test_x))

    @force2ndarray
    def predict_proba(self, test_x):
        """
        预测标签概率(分布)
        :param test_x:
        :return:
        """
        return self.meta_classifier.predict_proba(self.combine_base_classifier_predict_categorical_proba(
            test_x)) if self.use_probas else self.meta_classifier.predict_proba(
            self.combine_base_classifier_predict_categorical(test_x))

    @force2ndarray
    def predict_categorical_proba(self, test_x):
        """
        预测标签概率分布
        :param test_x:
        :return:
        """
        return self.meta_classifier.predict_categorical_proba(self.combine_base_classifier_predict_categorical_proba(
            test_x)) if self.use_probas else self.meta_classifier.predict_categorical_proba(
            self.combine_base_classifier_predict_categorical(test_x))


'''
LightGBMClassifier封装,主要是对添加进的categorical_feature进行处理，
注意：categorical_feature可以是int、float、str类型，如果是str必须是数值，比如'1','2',而不能是'x','y'
更多：https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#
'''


class LightGBMClassifier(SklearnClassifier):
    def __init__(self, train_params=None, subsample_features_rate=None, subsample_features_indices=None,
                 categorical_feature_indices=None):
        from lightgbm import LGBMClassifier
        SklearnClassifier.__init__(self, train_params, LGBMClassifier, subsample_features_rate,
                                   subsample_features_indices, categorical_feature_indices=None)
        self.self_define_categorical_feature_indices = categorical_feature_indices

    # 由于LGBMClassifier允许字符串变量，这里需要重写reshape_features
    def reshape_features(self, features):
        """
        读取features指定列用于训练或者随机选择某几列训练
        :param features:
        :return:
        """
        self.training_categorical_feature_indices = None
        _, columns = features.shape
        indices = list(range(0, columns))
        # 默认会排除字符串变量
        no_categorical_feature_indices = []
        if self.categorical_feature_indices is not None or self.self_define_categorical_feature_indices is not None:
            combine_categorical_feature_indices = set(
                [] if self.categorical_feature_indices is None else self.categorical_feature_indices) | set(
                [] if self.self_define_categorical_feature_indices is None else self.self_define_categorical_feature_indices)
            for index in indices:
                if index not in combine_categorical_feature_indices:
                    no_categorical_feature_indices.append(index)
        else:
            no_categorical_feature_indices = indices

        if self.subsample_features_indices is None and self.subsample_features_rate is not None:
            random.shuffle(no_categorical_feature_indices)
            self.subsample_features_indices = no_categorical_feature_indices[
                                              :int(len(no_categorical_feature_indices) * self.subsample_features_rate)]
        # 单独将categorical_feature放到最前面
        if self.self_define_categorical_feature_indices is not None:
            top_categorical_feature_indices = self.self_define_categorical_feature_indices
        else:
            top_categorical_feature_indices = self.categorical_feature_indices

        if self.subsample_features_indices is not None:
            if top_categorical_feature_indices is None:
                return features[:, self.subsample_features_indices]
            else:
                self.training_categorical_feature_indices = list(range(0, len(top_categorical_feature_indices)))
                return np.concatenate(
                    [features[:, top_categorical_feature_indices], features[:, self.subsample_features_indices]],
                    axis=1)
        if top_categorical_feature_indices is None:
            return features[:, no_categorical_feature_indices]
        else:
            self.training_categorical_feature_indices = list(range(0, len(top_categorical_feature_indices)))
            return np.concatenate(
                [features[:, top_categorical_feature_indices], features[:, no_categorical_feature_indices]],
                axis=1)

    # 添加是否有离散值情况的判断
    @force2ndarray
    def fit(self, train_x, train_y):
        self.class_num = len(set(train_y))
        reshape_train_x = self.reshape_features(train_x)
        if self.training_categorical_feature_indices is None:
            self.classifier_model.fit(reshape_train_x, train_y)
        else:
            self.classifier_model.fit(reshape_train_x, train_y,
                                      categorical_feature=self.training_categorical_feature_indices)

    # 允许numpy中含有字符串
    @force2ndarray
    def predict_proba(self, test_x):
        return self.classifier_model.predict_proba(self.reshape_features(test_x))

    @force2ndarray
    def predict_categorical_proba(self, test_x):
        probas = self.classifier_model.predict_proba(self.reshape_features(test_x))
        _, col = probas.shape
        if col > 1:
            return probas
        else:
            return np.asarray([[1 - proba, proba] for proba in probas])


'''
对CatBoostClassifier封装
'''


class CatBoostClassifier(SklearnClassifier):
    def __init__(self, train_params=None, subsample_features_rate=None, subsample_features_indices=None,
                 categorical_feature_indices=None):
        from catboost import CatBoostClassifier
        SklearnClassifier.__init__(self, train_params, CatBoostClassifier, subsample_features_rate,
                                   subsample_features_indices, categorical_feature_indices=None)
        self.self_define_categorical_feature_indices = categorical_feature_indices

    # 由于CatBoostClassifier允许字符串变量，这里需要重写reshape_features
    def reshape_features(self, features):
        """
        读取features指定列用于训练或者随机选择某几列训练
        :param features:
        :return:
        """
        self.training_categorical_feature_indices = None
        _, columns = features.shape
        indices = list(range(0, columns))
        # 默认会排除字符串变量
        no_categorical_feature_indices = []
        if self.categorical_feature_indices is not None or self.self_define_categorical_feature_indices is not None:
            combine_categorical_feature_indices = set(
                [] if self.categorical_feature_indices is None else self.categorical_feature_indices) | set(
                [] if self.self_define_categorical_feature_indices is None else self.self_define_categorical_feature_indices)
            for index in indices:
                if index not in combine_categorical_feature_indices:
                    no_categorical_feature_indices.append(index)
        else:
            no_categorical_feature_indices = indices

        if self.subsample_features_indices is None and self.subsample_features_rate is not None:
            random.shuffle(no_categorical_feature_indices)
            self.subsample_features_indices = no_categorical_feature_indices[
                                              :int(len(no_categorical_feature_indices) * self.subsample_features_rate)]
        # 单独将categorical_feature放到最前面
        if self.self_define_categorical_feature_indices is not None:
            top_categorical_feature_indices = self.self_define_categorical_feature_indices
        else:
            top_categorical_feature_indices = self.categorical_feature_indices

        if self.subsample_features_indices is not None:
            if top_categorical_feature_indices is None:
                return features[:, self.subsample_features_indices]
            else:
                self.training_categorical_feature_indices = list(range(0, len(top_categorical_feature_indices)))
                return np.concatenate(
                    [features[:, top_categorical_feature_indices], features[:, self.subsample_features_indices]],
                    axis=1)
        if top_categorical_feature_indices is None:
            return features[:, no_categorical_feature_indices]
        else:
            self.training_categorical_feature_indices = list(range(0, len(top_categorical_feature_indices)))
            return np.concatenate(
                [features[:, top_categorical_feature_indices], features[:, no_categorical_feature_indices]],
                axis=1)

    # 添加是否有离散值情况的判断
    @force2ndarray
    def fit(self, train_x, train_y):
        self.class_num = len(set(train_y))
        reshape_train_x = self.reshape_features(train_x)
        # 切分一部分出来做eval data
        X_new_train, X_new_eval, y_new_train, y_new_eval = train_test_split(reshape_train_x, train_y)
        if self.training_categorical_feature_indices is None:
            self.classifier_model.fit(X_new_train, y_new_train, eval_set=(X_new_eval, y_new_eval), use_best_model=True,
                                      verbose=False)
        else:
            self.classifier_model.fit(X_new_train, y_new_train, eval_set=(X_new_eval, y_new_eval), use_best_model=True,
                                      cat_features=self.training_categorical_feature_indices, verbose=False)

    # 允许numpy中含有字符串
    @force2ndarray
    def predict_proba(self, test_x):
        return self.classifier_model.predict_proba(self.reshape_features(test_x))

    @force2ndarray
    def predict_categorical_proba(self, test_x):
        probas = self.classifier_model.predict_proba(self.reshape_features(test_x))
        _, col = probas.shape
        if col > 1:
            return probas
        else:
            return np.asarray([[1 - proba, proba] for proba in probas])


'''
训练树结构，进行多进程训练的节点结构
'''


class TrainerNode(object):
    def __init__(self, classifier=None, train_x=None, train_y=None, if_stacking=False):
        self.classifier = classifier
        self.train_x = train_x
        self.train_y = train_y
        self.if_stacking = if_stacking
        self.children_nodes = []

    def train(self):
        if self.if_stacking is False:
            self.classifier.fit(self.train_x, self.train_y)
        else:
            # 计算meta_train_x
            if self.classifier.use_probas:
                meta_train_x = self.classifier.get_base_classifier_training_categorical_proba(self.train_x)
            else:
                meta_train_x = self.classifier.get_base_classifier_training_categorical(self.train_x)
            if self.classifier.meta_classifier.__class__.__name__ in ['KFolds_Classifier_Training_Wrapper',
                                                                      'StackingClassifier']:
                # 并行训练
                mpt = MultiProcessTrainer(self.classifier.meta_classifier.n_jobs)
                mpt.build_trainer_tree(self.classifier.meta_classifier, meta_train_x, self.train_y)
                mpt.fit()
            else:
                self.classifier.meta_classifier.fit(meta_train_x, self.train_y)


'''
协助模型进行多进程训练
'''


class MultiProcessTrainer(object):
    def __init__(self, n_jobs):
        self.n_jobs = n_jobs

    '''
    构建训练树结构
    '''

    def build_trainer_tree(self, classifier, train_x, train_y):
        """
        :param classifier: 当前分类器
        :param train_x: 训练特征
        :param train_y: 训练标签
        :return:
        """
        # 创建空根节点
        self.root_node = TrainerNode(None, None, None)

        # 递归创建子节点
        if classifier.__class__.__name__ == 'StackingClassifier':
            self.build_stacking_node(self.root_node, classifier, train_x, train_y)
        elif classifier.__class__.__name__ == 'KFolds_Classifier_Training_Wrapper':
            self.build_cv_node(self.root_node, classifier, train_x, train_y)
        else:
            self.build_normal_node(self.root_node, classifier, train_x, train_y)

    '''
    构建stacking树节点
    '''

    def build_stacking_node(self, parent_node, current_classifier, X_train, y_train):
        stacking_node = TrainerNode(current_classifier, train_x=X_train, train_y=y_train,
                                    if_stacking=True)
        parent_node.children_nodes.append(stacking_node)
        # 构建stacking的子节点
        for child_classifier in current_classifier.base_classifiers:
            if child_classifier.__class__.__name__ == 'StackingClassifier':
                self.build_stacking_node(stacking_node, child_classifier, X_train, y_train)
            elif child_classifier.__class__.__name__ == 'KFolds_Classifier_Training_Wrapper':
                self.build_cv_node(stacking_node, child_classifier, X_train, y_train)
            else:
                self.build_normal_node(stacking_node, child_classifier, X_train, y_train)

    '''
    构建cv树节点
    '''

    def build_cv_node(self, parent_node, current_classifier, train_x, train_y):
        kf = KFold(n_splits=current_classifier.k_fold, shuffle=False, random_state=current_classifier.random_state)
        index = 0
        for train_index, _ in kf.split(train_x):
            X_train = train_x[train_index]
            y_train = train_y[train_index]
            if current_classifier.extend_classifiers[index].__class__.__name__ == 'StackingClassifier':
                self.build_stacking_node(parent_node, current_classifier.extend_classifiers[index], X_train, y_train)
            elif current_classifier.extend_classifiers[
                index].__class__.__name__ == 'KFolds_Classifier_Training_Wrapper':
                self.build_cv_node(parent_node, current_classifier.extend_classifiers[index], X_train, y_train)
            else:
                self.build_normal_node(parent_node, current_classifier.extend_classifiers[index], X_train, y_train)
            index += 1

    '''
    构建normal树节点
    '''

    def build_normal_node(self, parent_node, current_classifier, train_x, train_y):
        normal_node = TrainerNode(classifier=current_classifier, train_x=train_x, train_y=train_y, if_stacking=False)
        parent_node.children_nodes.append(normal_node)

    '''
    并行训练模型
    '''

    def fit(self):
        def trainer_fit(node):
            node.train()

        if self.n_jobs == -1:
            max_cpu_count = cpu_count()
        else:
            max_cpu_count = min(cpu_count(), self.n_jobs)

        # 构建训练的层次结构索引
        self.trainer_level_dict = {}

        # 检索层次结构
        self.search_trainer_level(1, self.root_node)

        # 多进程/线程训练
        for index in range(99, 1, -1):
            trainers = self.trainer_level_dict.get(index)
            if trainers is not None:
                # if platform.system() == 'Linux':
                #     # 多进程支持,linux中生效
                #     p = Pool(min(max_cpu_count, len(trainers)))
                #     for i in range(len(trainers)):
                #         p.apply_async(trainer_fit, args=(trainers[i],))
                #     p.close()
                #     p.join()
                # else:
                #     # 多线程支持,windows中生效
                #     tasks = []
                #     for i in range(len(trainers)):
                #         task = threading.Thread(target=trainer_fit, args=(trainers[i],))
                #         task.start()
                #         tasks.append(task)
                #     for task in tasks:
                #         task.join()
                try:
                    # 先尝试多进程
                    p = Pool(min(max_cpu_count, len(trainers)))
                    # for i in range(len(trainers)):
                    #     p.apply_async(trainer_fit, args=(trainers[i],))
                    # p.close()
                    # p.join()
                    p.map(trainer_fit, trainers)
                except:
                    # 失败再尝试多线程
                    tasks = []
                    for i in range(len(trainers)):
                        task = threading.Thread(target=trainer_fit, args=(trainers[i],))
                        task.start()
                        tasks.append(task)
                    for task in tasks:
                        task.join()

    '''
    检索训练器的层次结构
    '''

    def search_trainer_level(self, current_level, current_node):
        if self.trainer_level_dict.get(current_level) is None:
            self.trainer_level_dict[current_level] = [current_node]
        else:
            self.trainer_level_dict[current_level].append(current_node)

        if len(current_node.children_nodes) > 0:
            for children_node in current_node.children_nodes:
                self.search_trainer_level(current_level + 1, children_node)
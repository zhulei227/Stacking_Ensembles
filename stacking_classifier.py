from sklearn.externals import joblib
from sklearn.model_selection import KFold
import pickle
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dropout, Dense, Activation
import copy
import numpy as np
import warnings

warnings.filterwarnings("ignore")
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 切换CPU

"""
样本平衡模块
"""


class ImbalanceSample(object):
    def __init__(self):
        from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE, BorderlineSMOTE, RandomOverSampler
        from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, InstanceHardnessThreshold, NearMiss, \
            TomekLinks, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN, OneSidedSelection, \
            CondensedNearestNeighbour, NeighbourhoodCleaningRule
        from imblearn.ensemble import EasyEnsemble, EasyEnsembleClassifier, BalancedBaggingClassifier, \
            BalancedRandomForestClassifier, BalanceCascade, RUSBoostClassifier

        self.oversamplers = {'ADASYN': ADASYN(), 'RandomOverSampler': RandomOverSampler(),
                             'SMOTE': SMOTE(), 'BorderlineSMOTE': BorderlineSMOTE(), 'SVMSMOTE': SVMSMOTE()}
        self.undersamplers = {
            'ClusterCentroids': ClusterCentroids(), 'RandomUnderSampler': RandomUnderSampler(),
            'InstanceHardnessThreshold': InstanceHardnessThreshold(),
            'NearMiss': NearMiss(), 'TomekLinks': TomekLinks(), 'EditedNearestNeighbours': EditedNearestNeighbours(),
            'RepeatedEditedNearestNeighbours': RepeatedEditedNearestNeighbours(), 'AllKNN': AllKNN(),
            'OneSidedSelection': OneSidedSelection(),
            'CondensedNearestNeighbour': CondensedNearestNeighbour(),
            'NeighbourhoodCleaningRule': NeighbourhoodCleaningRule()
        }
        self.ensemblesamplers = {'EasyEnsemble': EasyEnsemble(), 'EasyEnsembleClassifier': EasyEnsembleClassifier(),
                                 'BalancedBaggingClassifier': BalancedBaggingClassifier(),
                                 'BalanceCascade': BalanceCascade(),
                                 'BalancedRandomForestClassifier': BalancedRandomForestClassifier,
                                 'RUSBoostClassifier': RUSBoostClassifier()}

    def build_model(self, method="SMOTE"):
        if method in self.oversamplers:
            return self.oversamplers[method]
        if method in self.undersamplers:
            return self.undersamplers[method]
        if method in self.ensemblesamplers:
            return self.ensemblesamplers[method]
        return self.oversamplers['SMOTE']


"""
分类器接口
"""


class Classifier(object):
    """
    定义分类器接口
    """

    def __init__(self, where_store_classifier_model=None, train_params=None):
        """
        :param where_store_classifier_model:模型保存路径
        :param train_params: 训练参数
        """
        self.classifier_model_path = where_store_classifier_model
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


class SklearnClassifier(Classifier):
    """
    基于sklearn api的classifier实现
    """

    def __init__(self, where_store_classifier_model=None, train_params=None, classifier_class=None,
                 how_balance_sample="SMOTE"):
        Classifier.__init__(self, where_store_classifier_model, train_params)
        self.classifier_class = classifier_class
        self.how_balance_sample = how_balance_sample
        self.imbalance_sample_model = ImbalanceSample()

    def build_model(self):
        self.classifier_model = self.classifier_class(**self.train_params)

    def fit(self, train_x, train_y):
        if self.how_balance_sample is not None:
            train_x, train_y = self.imbalance_sample_model.build_model(self.how_balance_sample).fit_resample(train_x,
                                                                                                             train_y)
        self.class_num = len(set(train_y))
        self.classifier_model.fit(train_x, train_y)

    def predict(self, test_x):
        return self.classifier_model.predict(test_x)

    def predict_categorical(self, test_x):
        return to_categorical(self.predict(test_x), self.class_num)

    def predict_proba(self, test_x):
        return self.classifier_model.predict_proba(test_x)

    def predict_categorical_proba(self, test_x):
        probas = self.classifier_model.predict_proba(test_x)
        _, col = probas.shape
        if col > 1:
            return probas
        else:
            return np.asarray([[1 - proba, proba] for proba in probas])

    def save_model(self):
        joblib.dump(self.classifier_model, self.classifier_model_path)
        with open(self.classifier_model_path + '_class_num', 'wb') as w_class_num_file:
            pickle.dump(self.class_num, w_class_num_file)

    def load_model(self):
        self.classifier_model = joblib.load(self.classifier_model_path)
        with open(self.classifier_model_path + '_class_num', 'rb') as r_class_num_file:
            self.class_num = pickle.load(r_class_num_file)


class SVMClassifier(SklearnClassifier):
    def __init__(self, where_store_classifier_model=None, train_params=None):
        from sklearn.svm import SVC
        if train_params is None:
            train_params = {'probability': True}
        else:
            train_params['probability'] = True
        SklearnClassifier.__init__(self, where_store_classifier_model, train_params, SVC)


class RandomForestClassifier(SklearnClassifier):
    def __init__(self, where_store_classifier_model=None, train_params=None, how_balance_sample="SMOTE"):
        from sklearn.ensemble import RandomForestClassifier
        SklearnClassifier.__init__(self, where_store_classifier_model, train_params, RandomForestClassifier,
                                   how_balance_sample)


class GradientBoostingClassifier(SklearnClassifier):
    def __init__(self, where_store_classifier_model=None, train_params=None, how_balance_sample="SMOTE"):
        from sklearn.ensemble import GradientBoostingClassifier
        SklearnClassifier.__init__(self, where_store_classifier_model, train_params, GradientBoostingClassifier,
                                   how_balance_sample)


class AdaBoostClassifier(SklearnClassifier):
    def __init__(self, where_store_classifier_model=None, train_params=None, how_balance_sample="SMOTE"):
        from sklearn.ensemble import AdaBoostClassifier
        SklearnClassifier.__init__(self, where_store_classifier_model, train_params, AdaBoostClassifier,
                                   how_balance_sample)


class BaggingClassifier(SklearnClassifier):
    def __init__(self, where_store_classifier_model=None, train_params=None, how_balance_sample="SMOTE"):
        from sklearn.ensemble import BaggingClassifier
        SklearnClassifier.__init__(self, where_store_classifier_model, train_params, BaggingClassifier,
                                   how_balance_sample)


class LogisticRegression(SklearnClassifier):
    def __init__(self, where_store_classifier_model=None, train_params=None, how_balance_sample="SMOTE"):
        from sklearn.linear_model import LogisticRegression
        SklearnClassifier.__init__(self, where_store_classifier_model, train_params, LogisticRegression,
                                   how_balance_sample)


class NaiveBayesClassifier(SklearnClassifier):
    def __init__(self, where_store_classifier_model=None, train_params=None, how_balance_sample="SMOTE"):
        from sklearn.naive_bayes import GaussianNB
        SklearnClassifier.__init__(self, where_store_classifier_model, train_params, GaussianNB, how_balance_sample)


"""
DNN分类模型,该部分利用keras简单实现MLP分类
"""


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

    def save_model(self):
        self.classifier_model.save(self.classifier_model_path)

    def load_model(self):
        self.classifier_model = load_model(self.classifier_model_path)


class KFolds_Classifier_Training_Wrapper(Classifier):
    '''
    对训练的分类器进行交叉式训练，是对原始分类器的扩展，可独立使用
    '''

    def __init__(self, base_classifer=None, k_fold=5):
        """

        :param base_classifer:
        :param k_fold:
        """
        Classifier.__init__(self)
        self.base_classifier = base_classifer
        self.k_fold = k_fold
        self._suffix_for_cv = None  # 用于再次被KFolds_Training_Wrapper包装

    def _append_model_path(self):
        self.extend_classifiers_path_list = ['_cv' + str(index) for index in range(0, self.k_fold)]

    def build_model(self):
        """
        创建模型
        :return:
        """
        self._append_model_path()
        self.extend_classifiers = []
        for append_path in self.extend_classifiers_path_list:
            new_classifier = copy.deepcopy(self.base_classifier)
            if new_classifier.classifier_model_path is None:
                new_classifier._suffix_for_cv = append_path + (
                    self._suffix_for_cv if self._suffix_for_cv is not None else '')
            else:
                new_classifier.classifier_model_path = self.base_classifier.classifier_model_path + append_path + (
                    self._suffix_for_cv if self._suffix_for_cv is not None else '')
            new_classifier.build_model()
            self.extend_classifiers.append(new_classifier)

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
            self.extend_classifiers[index].fit(X_train, y_train)
            index += 1

    def _extract_k_fold_data_catogorical_features(self):
        """
        抽取交叉分割数据后的标签分布预测结果
        :return:
        """
        catogorical_results = []
        kf = KFold(n_splits=self.k_fold, shuffle=False, random_state=227)
        kf.get_n_splits(self.train_x)
        index = 0
        for _, test_index in kf.split(self.train_x):
            X_test = self.train_x[test_index]
            catogorical_results.append(self.extend_classifiers[index].predict_categorical(X_test))
            index += 1
        return np.concatenate(catogorical_results, axis=0)

    def _extract_k_fold_data_catogorical_proba_features(self):
        """
        抽取交叉分割数据后的标签概率分布预测结果
        :return:
        """
        catogorical_proba_results = []
        kf = KFold(n_splits=self.k_fold, shuffle=False, random_state=227)
        index = 0
        for _, test_index in kf.split(self.train_x):
            X_test = self.train_x[test_index]
            catogorical_proba_results.append(self.extend_classifiers[index].predict_categorical_proba(X_test))
            index += 1
        return np.concatenate(catogorical_proba_results, axis=0)

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

    def save_model(self):
        """
        存储模型
        :return:
        """
        for classifier in self.extend_classifiers:
            classifier.save_model()

    def load_model(self):
        """
        加载模型
        :return:
        """
        self._append_model_path()
        self.extend_classifiers = []
        for append_path in self.extend_classifiers_path_list:
            new_classifier = copy.deepcopy(self.base_classifier)
            if new_classifier.classifier_model_path is None:
                new_classifier._suffix_for_cv = append_path + (
                    self._suffix_for_cv if self._suffix_for_cv is not None else '')
            else:
                new_classifier.classifier_model_path = self.base_classifier.classifier_model_path + append_path + (
                    self._suffix_for_cv if self._suffix_for_cv is not None else '')
            new_classifier.load_model()
            self.extend_classifiers.append(new_classifier)


class StackingClassifier(Classifier):
    def __init__(self, base_classifiers=list(), meta_classifier=None, use_probas=True, force_cv=True, base_k_fold=5,
                 meta_k_fold=5):
        """
        为cv训练方式提供更好的支持

        :param classifiers: 基分类器列表
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
        self.meta_train_x = None
        self.meta_train_y = None
        self.force_cv = force_cv
        self._suffix_for_cv = None  # 被KFolds_Training_Warpper包装时,存放添加的后缀
        if self.force_cv:
            for index in range(0, len(self.base_classifiers)):
                if not isinstance(self.base_classifiers[index], KFolds_Classifier_Training_Wrapper):
                    self.base_classifiers[index] = KFolds_Classifier_Training_Wrapper(self.base_classifiers[index],
                                                                                      k_fold=base_k_fold)
            if not isinstance(self.meta_classifier, KFolds_Classifier_Training_Wrapper):
                self.meta_classifier = KFolds_Classifier_Training_Wrapper(self.meta_classifier, k_fold=meta_k_fold)

    def _build_base_classifier_models(self):
        """
        构建基分类器
        :return:
        """
        for classifier in self.base_classifiers:
            if classifier.classifier_model_path is not None:
                classifier.classifier_model_path += (self._suffix_for_cv if self._suffix_for_cv is not None else '')
            else:
                classifier._suffix_for_cv = self._suffix_for_cv
            classifier.build_model()

    def _build_meta_classifier_model(self):
        """
        构建元分类器
        :return:
        """
        if self.meta_classifier.classifier_model_path is not None:
            self.meta_classifier.classifier_model_path += (
                self._suffix_for_cv if self._suffix_for_cv is not None else '')
        else:
            self.meta_classifier._suffix_for_cv = self._suffix_for_cv
        self.meta_classifier.build_model()

    def build_model(self):
        """
        构建全部分类器
        :return:
        """
        self._build_base_classifier_models()
        self._build_meta_classifier_model()

    def _fit_base_classifiers(self, train_x, train_y):
        """
        训练基分类器
        :return:
        """
        for classifier in self.base_classifiers:
            classifier.fit(train_x, train_y)

    def _fit_meta_classifier(self):
        """
        训练元分类器
        :return:

        """
        self.meta_classifier.fit(self.meta_train_x, self.meta_train_y)

    def fit(self, train_x, train_y):
        """
        训练全部分类器
        :param train_x:
        :param train_y:
        :return:
        """
        self._fit_base_classifiers(train_x, train_y)
        if self.use_probas:
            self.meta_train_x = self._get_base_classifier_training_categorical_proba(train_x)
        else:
            self.meta_train_x = self._get_base_classifier_training_categorical(train_x)
        self.meta_train_y = train_y
        self._fit_meta_classifier()

    def _get_base_classifier_training_categorical_proba(self, train_x):
        """
        获取基分类器的训练数据
        :return:
        """
        _all_categorical_probas = []
        for classifier in self.base_classifiers:
            try:
                current_category_labels = classifier._extract_k_fold_data_catogorical_proba_features()  # 使用KFolds_Training_wrapper包装过的分类器会调用该api
            except:
                current_category_labels = classifier.predict_categorical_proba(train_x)
            _all_categorical_probas.append(current_category_labels)
        return np.concatenate(_all_categorical_probas, axis=-1)

    def _get_base_classifier_training_categorical(self, train_x):
        """
        获取基分类器的训练数据
        :return:
        """
        _all_categorical_labels = []
        for classifier in self.base_classifiers:
            try:
                current_category_labels = classifier._extract_k_fold_data_catogorical_features()  # 使用KFolds_Training_wrapper包装过的分类器会调用该api
            except:
                current_category_labels = classifier.predict_categorical(train_x)
            _all_categorical_labels.append(current_category_labels)
        return np.concatenate(_all_categorical_labels, axis=-1)

    def _combine_base_classifier_predict_categorical(self, test_x=None):
        """
        基分类器预测标签分布的组合
        :param test_x:
        :return:
        """
        _all_categorical_labels = [classifier.predict_categorical(test_x) for classifier in self.base_classifiers]
        return np.concatenate(_all_categorical_labels, axis=-1)

    def _combine_base_classifier_predict_categorical_proba(self, test_x=None):
        """
        基分类器预测标签概率分布的组合
        :param test_x:
        :return:
        """
        _all_categorical_probas = [classifier.predict_categorical_proba(test_x) for classifier in self.base_classifiers]
        return np.concatenate(_all_categorical_probas, axis=-1)

    def predict(self, test_x):
        """
        预测标签
        :param test_x:
        :return:
        """
        return self.meta_classifier.predict(self._combine_base_classifier_predict_categorical_proba(
            test_x)) if self.use_probas else self.meta_classifier.predict(
            self._combine_base_classifier_predict_categorical(test_x))

    def predict_categorical(self, test_x):
        """
        预测标签分布
        :param test_x:
        :return:[0,0,1,0,...]
        """
        return self.meta_classifier.predict_categorical(self._combine_base_classifier_predict_categorical_proba(
            test_x)) if self.use_probas else self.meta_classifier.predict_categorical(
            self._combine_base_classifier_predict_categorical(test_x))

    def predict_proba(self, test_x):
        """
        预测标签概率(分布)
        :param test_x:
        :return:
        """
        return self.meta_classifier.predict_proba(self._combine_base_classifier_predict_categorical_proba(
            test_x)) if self.use_probas else self.meta_classifier.predict_proba(
            self._combine_base_classifier_predict_categorical(test_x))

    def predict_categorical_proba(self, test_x):
        """
        预测标签概率分布
        :param test_x:
        :return:
        """
        return self.meta_classifier.predict_categorical_proba(self._combine_base_classifier_predict_categorical_proba(
            test_x)) if self.use_probas else self.meta_classifier.predict_categorical_proba(
            self._combine_base_classifier_predict_categorical(test_x))

    def _save_base_classifier_models(self):
        """
        保存基分类器
        :return:
        """
        for classifier in self.base_classifiers:
            classifier.save_model()

    def _save_meta_classifier_model(self):
        """
        保存元分类器
        :return:
        """
        self.meta_classifier.save_model()

    def save_model(self):
        """
        保存所有分类器
        :return:
        """
        self._save_base_classifier_models()
        self._save_meta_classifier_model()

    def _load_base_classifier_models(self):
        """
        加载基分类器
        :return:

        base_classifier should like XXXClassifier(where_store_classifier_model='/.../')
        """
        for classifier in self.base_classifiers:
            if classifier.classifier_model_path is None:
                classifier._suffix_for_cv = self._suffix_for_cv if self._suffix_for_cv is not None else ''
            classifier.load_model()

    def _load_meta_classifier_model(self):
        """
        加载元分类器
        :return:

        meta_classifier should like XXXClassifier(where_store_classifier_model='/.../')
        """
        if self.meta_classifier.classifier_model_path is None:
            self.meta_classifier._suffix_for_cv = self._suffix_for_cv if self._suffix_for_cv is not None else ''
        self.meta_classifier.load_model()

    def load_model(self):
        """
        加载模型
        所有基分类器以及元分类器必须指定存储路径
        :return:
        """
        self._load_base_classifier_models()
        self._load_meta_classifier_model()

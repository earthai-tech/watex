"""
The 'Exlib' sub-package aggregates prominent Machine Learning libraries for use in 
prediction tasks with common datasets within the `watex.models` framework, notably 
including `scikit-learn <https://scikit-learn.org/>`__.
"""

import importlib

def __getattr__(name):
    """
    Dynamically imports scikit-learn components upon access through the 
    watex.exlib.sklearn namespace, facilitating seamless integration of scikit-learn 
    objects into watex workflows.

    Parameters:
    - name (str): The name of the scikit-learn component (module, class, or function) 
      being accessed.

    Returns:
    - The dynamically imported scikit-learn component.

    Raises:
    - AttributeError: If the specified scikit-learn component cannot be found or 
      does not exist within the scikit-learn library.
    
    This method leverages a predefined mapping (`sklearn_component_mapping`) that 
    associates simplified attribute names with their corresponding modules in 
    scikit-learn, enabling both explicit and implicit module path resolution.
    """
    sklearn_component_mapping = {
        'BaseEstimator': 'sklearn.base',
        'TransformerMixin': 'sklearn.base',
        'ClassifierMixin': 'sklearn.base',
        'clone': 'sklearn.base',
        'KMeans': 'sklearn.cluster',
        'make_column_transformer': 'sklearn.compose',
        'make_column_selector': 'sklearn.compose',
        'ColumnTransformer': 'sklearn.compose',
        'ShrunkCovariance': 'sklearn.covariance',
        'LedoitWolf': 'sklearn.covariance',
        'FactorAnalysis': 'sklearn.decomposition',
        'PCA': 'sklearn.decomposition',
        'IncrementalPCA': 'sklearn.decomposition',
        'KernelPCA': 'sklearn.decomposition',
        'DummyClassifier': 'sklearn.dummy',
        'SelectKBest': 'sklearn.feature_selection',
        'f_classif': 'sklearn.feature_selection',
        'SelectFromModel': 'sklearn.feature_selection',
        'SimpleImputer': 'sklearn.impute',
        'permutation_importance': 'sklearn.inspection',
        'LogisticRegression': 'sklearn.linear_model',
        'SGDClassifier': 'sklearn.linear_model',
        'confusion_matrix': 'sklearn.metrics',
        'classification_report': 'sklearn.metrics',
        'mean_squared_error': 'sklearn.metrics',
        'f1_score': 'sklearn.metrics',
        'accuracy_score': 'sklearn.metrics',
        'precision_recall_curve': 'sklearn.metrics',
        'precision_score': 'sklearn.metrics',
        'recall_score': 'sklearn.metrics',
        'roc_auc_score': 'sklearn.metrics',
        'roc_curve': 'sklearn.metrics',
        'silhouette_samples': 'sklearn.metrics',
        'make_scorer': 'sklearn.metrics',
        'matthews_corrcoef': 'sklearn.metrics',
        'train_test_split': 'sklearn.model_selection',
        'validation_curve': 'sklearn.model_selection',
        'StratifiedShuffleSplit': 'sklearn.model_selection',
        'RandomizedSearchCV': 'sklearn.model_selection',
        'GridSearchCV': 'sklearn.model_selection',
        'learning_curve': 'sklearn.model_selection',
        'cross_val_score': 'sklearn.model_selection',
        'cross_val_predict': 'sklearn.model_selection',
        'KNeighborsClassifier': 'sklearn.neighbors',
        'Pipeline': 'sklearn.pipeline',
        'make_pipeline': 'sklearn.pipeline',
        'FeatureUnion': 'sklearn.pipeline',
        '_name_estimators': 'sklearn.utils',
        'OneHotEncoder': 'sklearn.preprocessing',
        'PolynomialFeatures': 'sklearn.preprocessing',
        'RobustScaler': 'sklearn.preprocessing',
        'OrdinalEncoder': 'sklearn.preprocessing',
        'StandardScaler': 'sklearn.preprocessing',
        'MinMaxScaler': 'sklearn.preprocessing',
        'LabelBinarizer': 'sklearn.preprocessing',
        'Normalizer': 'sklearn.preprocessing',
        'LabelEncoder': 'sklearn.preprocessing',
        'SVC': 'sklearn.svm',
        'LinearSVC': 'sklearn.svm',
        'LinearSVR': 'sklearn.svm',
        'DecisionTreeClassifier': 'sklearn.tree',
        'RandomForestClassifier': 'sklearn.ensemble',
        'AdaBoostClassifier': 'sklearn.ensemble',
        'VotingClassifier': 'sklearn.ensemble',
        'BaggingClassifier': 'sklearn.ensemble',
        'StackingClassifier': 'sklearn.ensemble',
        'ExtraTreesClassifier': 'sklearn.ensemble',
    }
    # Check if the component is in the predefined mapping
    if name in sklearn_component_mapping:
        full_module_path = sklearn_component_mapping[name]
        module = importlib.import_module(full_module_path)
        return getattr(module, name)
    
    # Default import behavior for unmapped names
    try:
        module = importlib.import_module(f"sklearn.{name.lower()}")
        return module
    except ImportError as e:
        raise AttributeError(f"scikit-learn component '{name}' not found") from e





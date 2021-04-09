import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from baselearner import BaseLearner

class LDA(BaseLearner):

    def __init__(self):


        self.model_args = {
            'gridsearch': False
        }
        self.grid = {
            'learner__n_components': [None, 1, 2, 5]
        }

        self.score_args = {}
        self.eval_args = {}
    
        self.coefs = []
        self.intercepts = []

        self.classifiers = []

    def train(self, train_x, train_y, test_x):
        self.pipeline = self.get_pipeline()
        
        if self.model_args['gridsearch']:
            cv = StratifiedKFold(n_splits=10, shuffle=False)
            grid = RandomizedSearchCV(self.pipeline, param_distributions=self.grid,
                                      cv=cv, scoring='roc_auc_ovo', n_jobs=-2,
                                      verbose=False, n_iter=15)
            grid.fit(train_x, train_y)
            print(grid.best_estimator_)
            clf = grid.best_estimator_
        else:
            clf = self.pipeline
            clf.fit(train_x, train_y)

        self.coefs.append(clf.named_steps['learner'].coef_)
        self.intercepts.append(clf.named_steps['learner'].intercept_)

        test_y_hat = clf.predict_proba(test_x)
        self.classifiers += [clf]
        return clf, test_y_hat


    def score(self, clf, 
              train_x, train_y, 
              test_x, test_y, test_y_hat,
              protocol='1v1'):
        # Calculate metrics
        aucs = super().multiclass_auc(test_y, test_y_hat, 
                                      clf.classes_,
                                      protocol=protocol)

        # Permutation importance
        # TODO: Remove randomstate and up repeats to >10
        n_repeats = 50
        sig = 1.96 # 99%: 2.576
        permutations = permutation_importance(clf, test_x, test_y, 
                                              n_repeats=n_repeats)#, random_state=0)
        
        # Add to scores dictionary
        scores = {}
        for comparison, score in aucs.items():
            name = 'AUC [{}]'.format(comparison)
            scores[name] = score

        scores['importances_mean'] = permutations.importances_mean
        scores['importances_std'] = permutations.importances_std
        scores['importances_sem'] = sig*permutations.importances_std/np.sqrt(n_repeats)
        

        return scores

    def evaluate(self, scores):
        results = {}

        all_metrics = scores[0].keys()
        for metric in all_metrics:
            name = '{:s}'.format(metric)
            name_ci = '{:s}_ci'.format(metric)
            current_scores = [score[metric] for score in scores if metric in score.keys()]
            results[name] = np.mean(current_scores)
            results[name_ci] = 1.96 * np.std(current_scores) / np.sqrt(len(current_scores))
        imps = np.array([score['importances_mean'] for score in scores])
        results['importances_mean'] = imps.mean(axis=0)
        results['importances_std'] = imps.std(axis=0)
        return results


    def get_pipeline(self):
        # Feature selection?
        # PCA ?

        steps = [
            # ('imputer', SimpleImputer()),
            # ('scaler', StandardScaler()),
            # ('selector', SelectKBest(k=2)),
            # ('learner', RandomForestClassifier())
            ('learner', LinearDiscriminantAnalysis(
                            solver='svd')
                            # shrinkage='auto')
            )
        ]

        return Pipeline(steps)
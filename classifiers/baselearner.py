from itertools import combinations

import numpy as np
from sklearn.metrics import roc_auc_score


class BaseLearner:

    def __init__(self):
        self.model_args = {}
        self.grid = {}
        self.score_args = {}
        self.eval_args = {}
        self.classifiers = []

    def multiclass_auc(self, y, y_hat, classes, protocol='1vr',):
        '''
        protocol options:
            1vr: Computes the problem as a 1 versus rest
                 generating n_class aucs
            1v1: Computes the AUC of all possible pairwise
                 combinations, generating 
                 n_classes*(nclasses - 1) / 2  aucs  
        TODO: Create option to compare specific labels  
              Specifically left vs right                      
        '''
        if protocol not in ['1vr', '1v1']:
            print("Multiclass_auc: Unknown protocol: {}. Using 1vr."\
                  .format(protocol))
               
        aucs = {}
        if protocol in ['1vr', 'all']:
            for i, class_ in enumerate(classes):
                auc_1vr = []
                name = '{} vs rest'.format(class_)
                y_binary = [1 if label == class_ else 0 for label in y]
                auc = roc_auc_score(y_binary, y_hat[:, i])
                auc_1vr += [auc]
                aucs[name] = auc
            aucs['avg 1vr'] = sum(aucs.values()) / len(aucs.values())

        if protocol in ['1v1', 'all']:
            for combination in combinations(classes, 2):
                auc_1v1 = []
                name = '{} vs {}'.format(combination[0], combination[1])
                y_binary = np.array([0 if label==combination[0] else \
                                     1 if label==combination[1] else -1 \
                                     for label in y])
                trial_idc = np.where(y_binary!=-1)[0]

                i = np.where(classes==combination[1])[0]
                if len(np.unique(y_binary[trial_idc]))<=1:
                    print("\nWARNING: Only a single class in this fold. AUC is not defined. Skipping...")
                    # TODO: calculate a score anyway
                else:
                    auc = roc_auc_score(y_binary[trial_idc], y_hat[trial_idc, i])
                    auc_1v1 += [auc]
                    aucs[name] = auc
            aucs['Avg 1v1'] = sum(aucs.values()) / len(aucs.values())

        # Move vs Rest
        y_binary = np.array([0 if label in ['Links', 'Rechts'] else 1 \
                            for label in y])
        auc = roc_auc_score(y_binary, y_hat[:, i])
        aucs['Move vs Rest'] = auc
         
        # TODO: Double check if this labelling is correct
        # Links = 0
        # Rechts = 1
        # Rest = 2
        return aucs

    def analyse_input_set(self):
        # Retrieve trainingsets
        
        # Retrieve transformed test set
        input_set = np.concatenate(
            [clf['clf'][:-1].transform(clf['test_x']) \
             for clf in self.classifiers])
        y = np.concatenate([clf['test_y'] for clf in self.classifiers])
        y_hat = np.concatenate([clf['test_y_hat'] for clf in self.classifiers])
        

        print('')
        
    

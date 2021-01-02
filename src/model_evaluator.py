from re import T
from sklearn.metrics import accuracy_score
import pandas as pd

class ModelEvaluator():
    
    @classmethod
    def get_total_accuracy(cls, targets, predictions):
        accuracy = accuracy_score(targets, predictions)
        return accuracy
    
    @classmethod
    def get_accuracy_per_label(cls, labels, targets, predictions):
        accuracy_per_label = {}
        i = 0
        for label in labels:
            label_targets = targets[:,i]
            label_predicitons = predictions[:,i]

            label_accuracy = accuracy_score(label_targets, label_predicitons)

            accuracy_per_label[label] = label_accuracy

            i += 1
        
        accuracy_per_label["no topic"] = cls.get_accuracy_comments_with_no_topic(targets, predictions)
        
        return accuracy_per_label
    
    @classmethod
    def get_accuracy_comments_with_no_topic(cls, targets, predictions):

        targets = pd.DataFrame(targets).reset_index(drop=True)
        predictions = pd.DataFrame(predictions).reset_index(drop=True)

        #Get all comments with no topic
        targets = targets.loc[(targets==0).all(axis=1)]

        #Get the predicitions of those comments with no topic
        predictions = predictions[predictions.index.isin(targets.index)]

        accuracy = accuracy_score(targets, predictions)

        return accuracy





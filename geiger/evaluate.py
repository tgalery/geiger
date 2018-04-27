from sklearn.metrics import roc_auc_score, fbeta_score
from keras.callbacks import Callback


def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    return fbeta_score(y_true, y_pred, beta=1)


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch + 1, score))


# class SummaryEvaluation(Callback):
#     def __init__(self, validation_data=(), interval=1):
#         super(Callback, self).__init__()
#
#         self.interval = interval
#         self.X_val, self.y_val = validation_data
#
#     def on_epoch_end(self, epoch, logs={}):
#         if epoch % self.interval == 0:
#             y_pred = self.model.predict(self.X_val, verbose=0)
#             score = classification_report(self.y_val, y_pred)
#             print("\n F1Score - epoch: %d - score: %.6f \n" % (epoch + 1, score))

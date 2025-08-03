from sklearn.ensemble import VotingClassifier

def build_fusion_model(svm_model, dl_model_wrapper):
    voting = VotingClassifier(
        estimators=[("svm", svm_model), ("dl", dl_model_wrapper)],
        voting="soft"
    )
    return voting

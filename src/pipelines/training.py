import os
from pathlib import Path

from metaflow import (
    Parameter,
    card,
    current,
    environment,
    step,
)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from common.pipeline import Pipeline, dataset


environment_variables = {
    "KERAS_BACKEND": os.getenv("KERAS_BACKEND", "tensorflow"),
    "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING": os.getenv("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "true"),
}

def build_features_transformer():
    """Build a Scikit-Learn transformer to preprocess the feature columns."""
    from sklearn.compose import ColumnTransformer, make_column_selector
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
    )

    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        # We can use the `handle_unknown="ignore"` parameter to ignore unseen categories
        # during inference. When encoding an unknown category, the transformer will
        # return an all-zero vector.
        OneHotEncoder(handle_unknown="ignore"),
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                # We'll apply the numeric transformer to all columns that are not
                # categorical (object).
                make_column_selector(dtype_exclude="object"),
            ),
            (
                "categorical",
                categorical_transformer,
                # We want to make sure we ignore the target column which is also a
                # categorical column. To accomplish this, we can specify the column
                # names we only want to encode.
                ["island", "sex"],
            ),
        ],
    )


def build_target_transformer():
    """Build a Scikit-Learn transformer to preprocess the target column."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder

    return ColumnTransformer(
        transformers=[("species", OrdinalEncoder(), ["species"])],
    )


def build_model(input_shape, learning_rate=0.01):
    """Build and compile the neural network to predict the species of a penguin."""
    from keras import Input, layers, models, optimizers

    model = models.Sequential(
        [
            Input(shape=(input_shape,)),
            layers.Dense(10, activation="relu"),
            layers.Dense(8, activation="relu"),
            layers.Dense(3, activation="softmax"),
        ],
    )

    model.compile(
        optimizer=optimizers.SGD(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

model_registry = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "xgboost": XGBClassifier,
    "keras": build_model
}

class Training(Pipeline):
    """Training pipeline.

    This pipeline trains, evaluates, and registers a model to predict the species of
    a given penguin.
    """

    training_epochs = Parameter(
        "training-epochs",
        help="Number of epochs that will be used to train the model.",
        default=50,
    )

    training_batch_size = Parameter(
        "training-batch-size",
        help="Batch size that will be used to train the model.",
        default=32,
    )

    accuracy_threshold = Parameter(
        "accuracy-threshold",
        help="Minimum accuracy threshold required to register the model.",
        default=0.7,
    )

    model_type = Parameter(
        "model-type",
        help="Type of model to train from model registry.",
        default="keras"
    )

    validation_split = Parameter(
        "validation-split",
        help="Fraction of training data to use for validation during Keras training.",
        default=0.2,
    )

    early_stopping_patience = Parameter(
        "early-stopping-patience",
        help="Number of epochs with no improvement after which training will be stopped for Keras models.",
        default=5,
    )

    @dataset
    @card
    @step
    def start(self):
        """Start and prepare the Training pipeline."""
        import mlflow

        self.logger.info("MLflow tracking server: %s", self.mlflow_tracking_uri)

        self.mode = "production" if current.is_production else "development"
        self.logger.info("Running flow in %s mode.", self.mode)

        try:
            # Let's start a new MLflow run to track the execution of this flow. We want
            # to set the name of the MLflow run to the Metaflow run ID so we can easily
            # recognize how they relate to each other.
            run = mlflow.start_run(run_name=current.run_id)
            self.mlflow_run_id = run.info.run_id
        except Exception as e:
            message = f"Failed to connect to MLflow server {self.mlflow_tracking_uri}."
            raise RuntimeError(message) from e

        # Now that everything is set up, we want to run a cross-validation process
        # to evaluate the model and train a final model on the entire dataset. Since
        # these two steps are independent, we can run them in parallel.
        self.next(self.cross_validation, self.transform)

    @card
    @step
    def cross_validation(self):
        """Generate the indices to split the data for the cross-validation process."""
        from sklearn.model_selection import KFold

        # We are going to use a 5-fold cross-validation process. We'll shuffle the data
        # before splitting it into batches.
        kfold = KFold(n_splits=5, shuffle=True)

        # We can now generate the indices to split the dataset into training and test
        # sets. This will return a tuple with the fold number and the training and test
        # indices for each of 5 folds.
        self.folds = list(enumerate(kfold.split(self.data)))

        # We can use a `foreach` to run every fold on a separate branch. Notice how we
        # pass the tuple with the fold number and the indices to next step.
        self.next(self.transform_fold, foreach="folds")

    @step
    def transform_fold(self):
        """Transform the data to build a model during the cross-validation process.

        This step will run for each fold in the cross-validation process. It uses
        a SciKit-Learn pipeline to preprocess the dataset before training a model.
        """
        # Let's start by unpacking the indices representing the training and test data
        # for the current fold.
        self.fold, (self.train_indices, self.test_indices) = self.input
        self.logger.info("Transforming fold %d...", self.fold)

        # We can use the indices to split the data into training and test sets.
        train_data = self.data.iloc[self.train_indices]
        test_data = self.data.iloc[self.test_indices]

        # Let's build the SciKit-Learn pipeline to process the feature columns,
        # fit it to the training data and transform both the training and test data.
        features_transformer = build_features_transformer()
        self.x_train = features_transformer.fit_transform(train_data)
        self.x_test = features_transformer.transform(test_data)

        # Finally, we can build the SciKit-Learn pipeline to process the target column,
        # fit it to the training data and transform both the training and test data.
        target_transformer = build_target_transformer()
        self.y_train = target_transformer.fit_transform(train_data)
        self.y_test = target_transformer.transform(test_data)

        # After processing the data and storing it as artifacts in the flow, we can move
        # to the training step.
        self.next(self.train_fold)

    @card
    @environment(vars=environment_variables)
    @step
    def train_fold(self):
        """Train a model as part of the cross-validation process.

        This step will run for each fold in the cross-validation process. It trains the
        model using the data we processed in the previous step.
        """
        import mlflow

        self.logger.info("Training fold %d...", self.fold)

        # We want to track the training process under the same MLflow run we started at
        # the beginning of the flow. Since we are running cross-validation, we will
        # create a nested run for each fold to keep track of each model individually.
        with (
            mlflow.start_run(run_id=self.mlflow_run_id),
            mlflow.start_run(
                run_name=f"cross-validation-fold-{self.fold}",
                nested=True,
            ) as run,
        ):
            # Let's store the identifier of the nested run in an artifact so we can
            # reuse it later when we evaluate the model.
            self.mlflow_fold_run_id = run.info.run_id

            # We are currently training a model corresponding to an individual fold,
            # so we don't want to log that model because it's useless.
            mlflow.autolog(log_models=False)
            # log model_type
            mlflow.log_params(
                {"model_type": self.model_type},
            )

            # Let's now build and fit the model on the training data we processed in the
            # previous step.
            if self.model_type == "keras":
                from keras.callbacks import EarlyStopping

                self.model = build_model(self.x_train.shape[1])
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    min_delta=0.01,
                    patience=self.early_stopping_patience,
                    restore_best_weights=True,
                    mode='auto'
                )
                history = self.model.fit(
                    self.x_train,
                    self.y_train,
                    epochs=self.training_epochs,
                    batch_size=self.training_batch_size,
                    validation_split=self.validation_split,
                    callbacks=[early_stopping],
                    verbose=0,
                )

                # Log the optimal epoch and validation metrics
                self.best_epoch = len(history.history['loss'])  # epochs actually trained
                mlflow.log_metrics({
                    "best_epoch": self.best_epoch,
                    "val_loss": history.history['val_loss'][-1],
                    "val_accuracy": history.history['val_accuracy'][-1],
                }, run_id=self.mlflow_fold_run_id)

                # Log per-epoch validation metrics
                for epoch in range(len(history.history['val_loss'])):
                    mlflow.log_metrics({
                        "val_loss": history.history['val_loss'][epoch],
                        "val_accuracy": history.history['val_accuracy'][epoch],
                    }, step=epoch, run_id=self.mlflow_fold_run_id)

                self.logger.info(
                    "Fold %d - train_loss: %f - train_accuracy: %f",
                    self.fold,
                    history.history["loss"][-1],
                    history.history["accuracy"][-1],
                )
            else:
                ModelClass = model_registry[self.model_type]
                self.model = ModelClass()
                self.model.fit(self.x_train, self.y_train.ravel())


        # After training a model for this fold, we want to evaluate it.
        self.next(self.evaluate_fold)

    @card(type="html")
    @environment(vars=environment_variables)
    @step
    def evaluate_fold(self):
        """Evaluate the model we created as part of the cross-validation process.

        This step will run for each fold in the cross-validation process. It evaluates
        the model using the test data associated with the current fold.
        """
        import mlflow

        self.logger.info("Evaluating fold %d...", self.fold)

        y_true = self.y_test.ravel()

        if self.model_type == "keras":
            # Let's evaluate the model using the test data we processed before.
            self.test_loss, self.test_accuracy = self.model.evaluate(
                self.x_test,
                y_true,
                verbose=0,
            )

            from keras.metrics import Precision, Recall
            import numpy as np

            predictions = self.model.predict(self.x_test)
            num_classes = predictions.shape[1]

            precision_values = []
            recall_values = []
            for class_id in range(num_classes):
                precision_metric = Precision(class_id=class_id)
                recall_metric = Recall(class_id=class_id)
                precision_metric.update_state(y_true, predictions)
                recall_metric.update_state(y_true, predictions)
                precision_values.append(float(precision_metric.result().numpy()))
                recall_values.append(float(recall_metric.result().numpy()))

            self.test_precision = float(np.mean(precision_values))
            self.test_recall = float(np.mean(recall_values))
        else:
            from sklearn.metrics import accuracy_score, precision_score, recall_score

            predictions = self.model.predict(self.x_test)
            self.test_accuracy = accuracy_score(y_true, predictions)
            self.test_precision = precision_score(y_true, predictions, average='macro')
            self.test_recall = recall_score(y_true, predictions, average='macro')
            self.test_loss = 0  # Placeholder, since sklearn doesn't have loss

        if self.model_type == "keras":
            self.logger.info(
                "Fold %d - test_loss: %f - test_accuracy: %f - test_precision: %f - test_recall: %f",
                self.fold,
                self.test_loss,
                self.test_accuracy,
                self.test_precision,
                self.test_recall,
            )
        else:
            self.logger.info(
                "Fold %d - test_accuracy: %f - test_precision: %f - test_recall: %f",
                self.fold,
                self.test_accuracy,
                self.test_precision,
                self.test_recall,
            )

        # Let's track the evaluation metrics under the nested MLflow run corresponding
        # to the current fold.
        metrics = {
            "test_accuracy": self.test_accuracy,
            "test_precision": self.test_precision,
            "test_recall": self.test_recall,
        }
        if self.model_type == "keras":
            metrics["test_loss"] = self.test_loss
        mlflow.log_metrics(metrics, run_id=self.mlflow_fold_run_id)

        from io import BytesIO
        import base64
        import matplotlib.pyplot as plt

        if self.model_type == "keras":
            predicted_classes = self.model.predict(self.x_test).argmax(axis=1)
        else:
            predicted_classes = self.model.predict(self.x_test)

        cm = confusion_matrix(
            y_true,
            predicted_classes,
        )
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

        self.html = f"""
        <h2>Fold results</h2>
        <h3>Confusion Matrix</h3>
        <img src="data:image/png;base64,{img_base64}" />
        """

        # When we finish evaluating the models in the cross-validation process, we want
        # to average the scores to determine the overall model performance.
        self.next(self.average_scores)

    @card
    @step
    def average_scores(self, inputs):
        """Averages the scores computed for each individual model."""
        import mlflow
        import numpy as np

        # We need access to the `mlflow_run_id` artifact that we set at the start of
        # the flow, but since we are in a join step, we need to merge the artifacts
        # from the incoming branches to make `mlflow_run_id` available. This merge will
        # discard every artifact that was created in the previous branches and keep only
        # the `mlflow_run_id` artifact.
        self.merge_artifacts(inputs, include=["mlflow_run_id"])

        # Let's calculate the mean and standard deviation of the accuracy, loss,
        # precision and recall from all the cross-validation folds.
        accuracies = [i.test_accuracy for i in inputs]
        precisions = [i.test_precision for i in inputs]
        recalls = [i.test_recall for i in inputs]

        self.test_accuracy = np.mean(accuracies)
        self.test_precision = np.mean(precisions)
        self.test_recall = np.mean(recalls)
        self.test_accuracy_std = np.std(accuracies)
        self.test_precision_std = np.std(precisions)
        self.test_recall_std = np.std(recalls)

        if self.model_type == "keras":
            losses = [i.test_loss for i in inputs]
            self.test_loss = np.mean(losses)
            self.test_loss_std = np.std(losses)
            self.logger.info("Loss: %f ±%f", self.test_loss, self.test_loss_std)
        else:
            self.test_loss = 0
            self.test_loss_std = 0

        self.logger.info("Accuracy: %f ±%f", self.test_accuracy, self.test_accuracy_std)
        self.logger.info("Precision: %f ±%f", self.test_precision, self.test_precision_std)
        self.logger.info("Recall: %f ±%f", self.test_recall, self.test_recall_std)

        # Let's log the model metrics on the parent run.
        metrics = {
            "test_accuracy": self.test_accuracy,
            "test_accuracy_std": self.test_accuracy_std,
            "test_precision": self.test_precision,
            "test_precision_std": self.test_precision_std,
            "test_recall": self.test_recall,
            "test_recall_std": self.test_recall_std,
        }
        if self.model_type == "keras":
            metrics.update({
                "test_loss": self.test_loss,
                "test_loss_std": self.test_loss_std,
            })
        mlflow.log_metrics(metrics, run_id=self.mlflow_run_id)

        # After we finish evaluating the cross-validation process, we can send the flow
        # to the registration step to register the final version of the model.
        self.next(self.register)

    @card
    @step
    def transform(self):
        """Apply the transformation pipeline to the entire dataset.

        We'll use the entire dataset to build the final model, so we need to transform
        the dataset before training.

        We want to store the transformers as artifacts so we can later use them
        to transform the input data during inference.
        """
        # Let's build the SciKit-Learn pipeline and transform the dataset features.
        self.features_transformer = build_features_transformer()
        self.x = self.features_transformer.fit_transform(self.data)

        # Let's build the SciKit-Learn pipeline and transform the target column.
        self.target_transformer = build_target_transformer()
        self.y = self.target_transformer.fit_transform(self.data)

        # Now that we have transformed the data, we can train the final model.
        self.next(self.train)

    @card
    @environment(vars=environment_variables)
    @step
    def train(self):
        """Train the final model using the entire dataset."""
        import mlflow

        self.logger.info("Training final model...")
        mlflow.log_params(
            {"model_type": self.model_type},
            run_id=self.mlflow_run_id,
        )

        # Let's log the training process under the current MLflow run.
        with mlflow.start_run(run_id=self.mlflow_run_id):
            # We want to log the model manually, so let's disable automatic logging.
            mlflow.autolog(log_models=False)

            # Let's now build and fit the model on the entire dataset.
            if self.model_type == "keras":
                from keras.callbacks import EarlyStopping

                self.model = build_model(self.x.shape[1])
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    min_delta=0.01,
                    patience=self.early_stopping_patience,
                    restore_best_weights=True,
                    mode='auto'
                )
                history = self.model.fit(
                    self.x,
                    self.y.ravel(),
                    epochs=self.training_epochs,
                    batch_size=self.training_batch_size,
                    validation_split=self.validation_split,
                    callbacks=[early_stopping],
                    verbose=2,
                )

                # Log the optimal epoch and final validation metrics
                self.best_epoch = len(history.history['loss'])
                mlflow.log_metrics({
                    "best_epoch": self.best_epoch,
                    "final_val_loss": history.history['val_loss'][-1],
                    "final_val_accuracy": history.history['val_accuracy'][-1],
                }, run_id=self.mlflow_run_id)

                # Log per-epoch validation metrics for the final model
                for epoch in range(len(history.history['val_loss'])):
                    mlflow.log_metrics({
                        "val_loss": history.history['val_loss'][epoch],
                        "val_accuracy": history.history['val_accuracy'][epoch],
                    }, step=epoch, run_id=self.mlflow_run_id)
            else:
                ModelClass = model_registry[self.model_type]
                self.model = ModelClass()
                self.model.fit(self.x, self.y.ravel())

        # After we finish training the model, we want to create the feature importance card.
        self.next(self.feature_importance_card)

    @card(type="html")
    @step
    def feature_importance_card(self):
        """Create a feature importance visualization card."""
        # Compute feature importance
        feature_importance_pairs = self._compute_feature_importance()

        # Create HTML visualization
        html_parts = [
            "<h2>Feature Importance</h2>",
            "<p>This chart shows the relative importance of each feature in predicting penguin species.</p>",
            "<div style='margin: 20px 0;'>",
            "<table style='border-collapse: collapse; width: 100%;'>",
            "<thead>",
            "<tr style='background-color: #f2f2f2;'>",
            "<th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>Feature</th>",
            "<th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>Importance</th>",
            "<th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>Bar</th>",
            "</tr>",
            "</thead>",
            "<tbody>"
        ]

        # Find max importance for scaling
        max_importance = max(imp for _, imp in feature_importance_pairs) if feature_importance_pairs else 1

        for feature_name, importance in feature_importance_pairs:
            # Create a simple bar using CSS
            bar_width = int((importance / max_importance) * 200) if max_importance > 0 else 0
            bar_html = f"<div style='width: {bar_width}px; height: 20px; background-color: #4CAF50; border-radius: 3px;'></div>"

            html_parts.append(
                f"<tr>"
                f"<td style='border: 1px solid #ddd; padding: 8px;'>{feature_name}</td>"
                f"<td style='border: 1px solid #ddd; padding: 8px;'>{importance:.4f}</td>"
                f"<td style='border: 1px solid #ddd; padding: 8px;'>{bar_html}</td>"
                f"</tr>"
            )

        html_parts.extend([
            "</tbody>",
            "</table>",
            "</div>",
            f"<p><strong>Model Type:</strong> {self.model_type}</p>",
            f"<p><strong>Number of Features:</strong> {len(feature_importance_pairs)}</p>"
        ])

        self.html = "\n".join(html_parts)

        # After creating the card, proceed to model registration
        self.next(self.register)

    @environment(vars=environment_variables)
    @step
    def register(self, inputs):
        """Register the model in the model registry.

        This function will prepare and register the final model in the model registry
        if its accuracy is above a predefined threshold.
        """
        import tempfile

        import mlflow

        # Since this is a join step, we need to merge the artifacts from the incoming
        # branches to make them available here.
        self.merge_artifacts(inputs)

        # We only want to register the model if its accuracy is above the
        # `accuracy_threshold` parameter.
        if self.test_accuracy >= self.accuracy_threshold:
            self.registered = True
            self.logger.info("Registering model...")

            # We'll register the model under the current MLflow run. We also need to
            # create a temporary directory to store the model artifacts.
            with (
                mlflow.start_run(run_id=self.mlflow_run_id),
                tempfile.TemporaryDirectory() as directory,
            ):
                self.artifacts = self._get_model_artifacts(directory)
                self.pip_requirements = self._get_model_pip_requirements()

                # Let's point to the `/src` folder containing the pipeline code.
                root = Path(__file__).parent.parent
                self.code_paths = [(root / "inference" / "backend.py").as_posix()]

                # We can now register the model in the model registry. This will
                # automatically create a new version of the model.
                mlflow.pyfunc.log_model(
                    name="model",
                    python_model=root / "inference" / "model.py",
                    registered_model_name="penguins",
                    code_paths=self.code_paths,
                    artifacts=self.artifacts,
                    pip_requirements=self.pip_requirements,
                )

        else:
            self.registered = False
            self.logger.info(
                "The accuracy of the model (%.2f) is lower than the accuracy threshold "
                "(%.2f). Skipping model registration.",
                self.test_accuracy,
                self.accuracy_threshold,
            )

        # Let's now move to the final step of the pipeline.
        self.next(self.end)

    @step
    def end(self):
        """End the Training pipeline."""
        self.logger.info("The pipeline finished successfully.")

    def _get_model_artifacts(self, directory: str):
        """Return the list of artifacts that will be included with model.

        The model must preprocess the raw input data before making a prediction, so we
        need to include the Scikit-Learn transformers as part of the model package.
        """
        import joblib

        # Let's start by saving the model inside the supplied directory.
        if self.model_type == "keras":
            model_path = (Path(directory) / "model.keras").as_posix()
            self.model.save(model_path)
        else:
            model_path = (Path(directory) / "model.joblib").as_posix()
            joblib.dump(self.model, model_path)

        # We also want to save the Scikit-Learn transformers so we can package them
        # with the model and use them during inference.
        features_transformer_path = (Path(directory) / "features.joblib").as_posix()
        target_transformer_path = (Path(directory) / "target.joblib").as_posix()
        joblib.dump(self.features_transformer, features_transformer_path)
        joblib.dump(self.target_transformer, target_transformer_path)

        return {
            "model": model_path,
            "features_transformer": features_transformer_path,
            "target_transformer": target_transformer_path,
        }

    def _get_model_pip_requirements(self):
        """Return the list of required packages to run the model in production."""
        import numpy as np
        import pandas as pd
        import sklearn

        requirements = [
            f"scikit-learn=={sklearn.__version__}",
            f"pandas=={pd.__version__}",
            f"numpy=={np.__version__}",
            "joblib",
        ]

        if self.model_type == "keras":
            import keras
            import tensorflow as tf
            requirements.extend([
                f"keras=={keras.__version__}",
                f"tensorflow=={tf.__version__}",
            ])
        elif self.model_type == "xgboost":
            import xgboost
            requirements.append(f"xgboost=={xgboost.__version__}")

        return requirements


    def _compute_feature_importance(self):
        """Compute feature importance for the trained model.

        Returns a list of tuples (feature_name, importance) sorted by importance descending.
        """
        import numpy as np
        from sklearn.inspection import permutation_importance

        if self.model_type == "keras":
            # For Keras models, use permutation importance on the training data
            # We'll use a subset of the data for efficiency
            n_samples = min(1000, len(self.x))  # Use up to 1000 samples
            indices = np.random.choice(len(self.x), n_samples, replace=False)
            x_sample = self.x[indices]
            # Permutation importance expects integer class labels for classification
            y_sample = self.y[indices].ravel().astype(int)

            # We need to wrap the Keras model so its `predict` method returns class
            # labels (indices) instead of probabilities. This is required for
            # `permutation_importance` with `scoring='accuracy'`.
            class KerasClassifierWrapper:
                def __init__(self, model, classes):
                    self.model = model
                    self._estimator_type = "classifier"
                    self.classes_ = classes

                def fit(self, X, y=None):
                    # Dummy fit method to pass scikit-learn validation
                    return self

                def get_params(self, deep=True):
                    # Dummy get_params method to pass scikit-learn validation
                    return {}

                def predict(self, X):
                    predictions = self.model.predict(X, verbose=0)
                    if predictions.shape[1] > 1:  # Multi-class
                        return np.argmax(predictions, axis=1)
                    return (predictions > 0.5).astype(int).ravel()

            # Compute permutation importance using the wrapped model
            perm_importance = permutation_importance(
                KerasClassifierWrapper(self.model, np.unique(y_sample)),
                x_sample,
                y_sample,
                n_repeats=5,
                random_state=42,
                scoring="accuracy",
            )
            importances = perm_importance.importances_mean

        elif self.model_type in ["random_forest", "xgboost"]:
            # Tree-based models have built-in feature importance
            importances = self.model.feature_importances_

        elif self.model_type == "logistic_regression":
            # Linear models have coefficients
            importances = np.abs(self.model.coef_[0])  # Take absolute values

        else:
            raise ValueError(f"Feature importance not supported for model type: {self.model_type}")

        # Get feature names from the transformer
        feature_names = self.features_transformer.get_feature_names_out()

        # Create list of (name, importance) tuples and sort by importance descending
        feature_importance_pairs = list(zip(feature_names, importances))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)

        return feature_importance_pairs


if __name__ == "__main__":
    Training()

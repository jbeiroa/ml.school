import importlib
import json
import logging
import os
import time
import hashlib
from contextlib import suppress
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import pydantic
from mlflow.models import set_model
from mlflow.pyfunc.model import PythonModelContext


class Input(pydantic.BaseModel):
    """Prediction input that will be received from the client.

    This class is responsible for defining the structure of the input data that the
    model will receive from the client. The input data will be automatically validated
    by MLflow against this schema before making a prediction.
    """

    island: str | None = None
    culmen_length_mm: float | None = None
    culmen_depth_mm: float | None = None
    flipper_length_mm: float | None = None
    body_mass_g: float | None = None
    sex: str | None = None


class Output(pydantic.BaseModel):
    """Prediction output that will be returned to the client.

    This class is responsible for defining the structure of the output data that the
    model will return to the client.
    """

    prediction: str | None = None
    confidence: float | None = None

class PredictionMetrics(pydantic.BaseModel):
    """Prediction metrics that will be logged for each prediction."""

    latency_input_ms: float
    latency_inference_ms: float
    latency_output_ms: float
    latency_total_ms: float
    sample_count: int
    error: str | None = None
    error_phase: str | None = None

class Model(mlflow.pyfunc.PythonModel):
    """A custom model implementing an inference pipeline to classify penguins.

    This inference pipeline has three phases: processing the input data, prediction, and
    processing the output before generating the response to the client. The pipeline
    will optionally store the input requests and predictions.

    The [Custom MLflow Models with mlflow.pyfunc](https://mlflow.org/blog/custom-pyfunc)
    blog post is a great reference to understand how to use custom Python models in
    MLflow.
    """

    def __init__(self) -> None:
        """Initialize the model."""
        self.backend = None

    def load_context(self, context: PythonModelContext | None) -> None:
        """Load and prepare the model context to make predictions.

        This function is called only once as soon as the model is constructed. It loads
        the transformers and the Keras model specified as artifacts.
        """
        self._configure_logging()
        self._initialize_backend()
        self._load_artifacts(context)

        self.logger.info("Model is ready to receive requests")

    def predict(
        self,
        context,  # noqa: ARG002
        model_input: list[Input],
        params: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> Output:
        """Handle the request received from the client.

        This method is responsible for processing the input data received from the
        client, making a prediction using the model, and returning a readable response
        to the client.
        """
        # set timer for prediction metrics
        start_time = time.perf_counter()
        # Let's convert the input data into a DataFrame so we can process it
        # using the Scikit-Learn transformers.
        model_input = pd.DataFrame([sample.model_dump() for sample in model_input])

        if model_input.empty:
            self.logger.warning("Received an empty request.")
            return []

        # initialize metrics
        metrics = {
            "latency_input_ms": 0.0,
            "latency_inference_ms": 0.0,
            "latency_output_ms": 0.0,
            "latency_total_ms": 0.0,
            "sample_count": len(model_input),
            "error": None,
            "error_phase": None,
        }

        self.logger.info(
            "Received prediction request with %d %s",
            len(model_input),
            "samples" if len(model_input) > 1 else "sample",
        )

        model_output = []

        # Input processing phase
        input_start = time.perf_counter()
        transformed_payload = self.process_input(model_input)
        metrics["latency_input_ms"] = (time.perf_counter() - input_start) * 1000

        if transformed_payload is None:
            self.logger.warning("The request payload could not be processed.")
            metrics["error"] = "The request payload could not be processed."
            metrics["error_phase"] = "input_processing"
            return []

        # Inference phase
        inference_start = time.perf_counter()

        try:
            self.logger.info("Making a prediction using the transformed payload...")
            if self.is_keras:
                predictions = self.model.predict(transformed_payload, verbose=0)
            else:
                predictions = self.model.predict(transformed_payload)
            metrics["latency_inference_ms"] = (time.perf_counter() - inference_start) * 1000
        except Exception:
            self.logger.exception("There was an error during inference.")
            metrics["error"] = "There was an error during inference."
            metrics["error_phase"] = "inference"
            return []

        # Output processing phase
        output_start = time.perf_counter()
        try:
            model_output = self.process_output(predictions)
        except Exception:
            self.logger.exception("There was an error processing the output.")
            metrics["error"] = "There was an error processing the output."
            metrics["error_phase"] = "output_processing"
            return []

        # Total latency
        metrics["latency_output_ms"] = (time.perf_counter() - output_start) * 1000
        self._log_metrics(metrics)

        if self.backend is not None:
            self.backend.save(model_input, model_output)

        self.logger.info("Returning prediction to the client")
        self.logger.debug("%s", model_output)

        return model_output

    def process_input(self, payload: pd.DataFrame) -> pd.DataFrame | None:
        """Process the input data received from the client.

        This method is responsible for transforming the input data received from the
        client into a format that can be used by the model.
        """
        self.logger.info("Transforming payload...")

        # We need to transform the payload using the transformer. This can raise an
        # exception if the payload is not valid, in which case we should return None
        # to indicate that the prediction should not be made.
        try:
            result = self.features_transformer.transform(payload)
        except Exception:
            self.logger.exception("There was an error processing the payload.")
            return None

        return result

    def process_output(self, output: np.ndarray) -> list[dict[str, Any]]:
        """Process the prediction received from the model.

        This method is responsible for transforming the prediction received from the
        model into a readable format that will be returned to the client.
        """
        self.logger.info("Processing prediction received from the model...")
        # Debug: Log what we're actually receiving
        self.logger.info(f"Output type: {type(output)}")
        self.logger.info(f"Output shape: {output.shape if hasattr(output, 'shape') else 'N/A'}")
        self.logger.info(f"Output dtype: {output.dtype if hasattr(output, 'dtype') else type(output[0]) if len(output) > 0 else 'empty'}")
        self.logger.info(f"First prediction sample: {output[0] if hasattr(output, '__getitem__') else output}")
        self.logger.info(f"is_keras flag: {self.is_keras}")

        result = []
        if output is not None:
            if self.is_keras:
                prediction = np.argmax(output, axis=1)
                confidence = np.max(output, axis=1)
                # Let's transform the prediction index back to the
                # original species. We can use the target transformer
                # to access the list of classes.
                classes = self.target_transformer.named_transformers_[
                    "species"
                ].categories_[0]
                prediction = np.vectorize(lambda x: classes[x])(prediction)
            else:
                classes = self.target_transformer.named_transformers_[
                    "species"
                ].categories_[0]
                # Cast the numerical float output to int so we can index the classes array
                prediction = np.vectorize(lambda x: classes[int(x)])(output)
                confidence = np.full(len(output), None)  # no confidence for sklearn

            # We can now return the prediction and the confidence from the model.
            # Notice that we need to unwrap the numpy values so we can serialize the
            # output as JSON.
            result = [
                {"prediction": p.item(), "confidence": c.item() if c is not None else None}
                for p, c in zip(prediction, confidence, strict=True)
            ]

        return result

    def _initialize_backend(self):
        """Initialize the model backend that the pipeline will use to store the data.

        The backend is responsible for storing the input requests and the predictions
        from the model. The inference pipeline will dynamically create an instance of
        the specified backend and use it to store the data.
        """
        # For the configuration to remain clean and easy to remember, we want to
        # reference backend classes as "backend.<class_name>" without having to include
        # their full class path. To accomplish this, we need to import the
        # inference.backend module so it's available to the `import_module` call.
        with suppress(ImportError):
            import inference.backend  # noqa: F401

        self.logger.info("Initializing model backend...")
        backend_class = os.getenv("MODEL_BACKEND", "backend.Local")

        if backend_class is not None:
            # We can optionally load a JSON configuration file and use it to initialize
            # the backend instance.
            backend_config = os.getenv("MODEL_BACKEND_CONFIG", None)

            try:
                if backend_config is not None:
                    backend_config = Path(backend_config)
                    backend_config = (
                        json.loads(backend_config.read_text())
                        if backend_config.exists()
                        else None
                    )

                module, cls = backend_class.rsplit(".", 1)
                module = importlib.import_module(module)
                self.backend = getattr(module, cls)(config=backend_config)
            except Exception:
                self.logger.exception(
                    'There was an error initializing backend "%s".',
                    backend_class,
                )

        self.logger.info("Backend: %s", backend_class if self.backend else None)

    def _load_artifacts(self, context: PythonModelContext | None):
        if context is None:
            self.logger.warning("No model context was provided.")
            return

        # By default, we want to use the TensorFlow backend for Keras.
        if not os.getenv("KERAS_BACKEND"):
            os.environ["KERAS_BACKEND"] = "tensorflow"

        import keras

        self.logger.info("Keras backend: %s", os.environ.get("KERAS_BACKEND"))

        # First, we need to load the transformation pipelines from the model artifacts.
        # These will help us transform the input data and the output predictions.
        self.features_transformer = joblib.load(
            context.artifacts["features_transformer"],
        )
        self.target_transformer = joblib.load(context.artifacts["target_transformer"])

        # Then, we can load the model based on the file type.
        model_path = context.artifacts["model"]
        if model_path.endswith(".keras"):
            self.model = keras.saving.load_model(model_path)
            self.is_keras = True
        else:
            self.model = joblib.load(model_path)
            self.is_keras = False

    def _configure_logging(self):
        """Configure how the logging system will behave."""
        import sys

        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            level=logging.INFO,
        )

        self.logger = logging.getLogger("model")

    def _log_metrics(self, metrics: dict[str, Any]) -> None:
        """Log the prediction metrics to MLflow."""
        if metrics["error"] is None:
            mlflow.log_metrics({
                "latency_input_ms": metrics["latency_input_ms"],
                "latency_inference_ms": metrics["latency_inference_ms"],
                "latency_output_ms": metrics["latency_output_ms"],
                "latency_total_ms": metrics["latency_total_ms"],
            })
        else:
            mlflow.log_metrics({
                "error": metrics["error"],
                "error_phase": metrics["error_phase"]})


set_model(Model())

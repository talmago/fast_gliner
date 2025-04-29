from pathlib import Path
from typing import List, Optional, Union

from huggingface_hub import snapshot_download

from .fast_gliner import PyFastGliNER


class FastGLiNER:
    """
    This is a wrapper class for PyFastGliNER.

    It is used to initialize the PyFastGliNER class and call its methods.

    ```python
    from fast_gliner import FastGLiNER

    model = FastGLiNER.from_pretrained(
        model_id="juampahc/gliner_multi-v2.1-onnx",
        onnx_path="model.onnx"
    )

    model.predict_entities("I am James Bond", ["person"])
    ```

    Output:
    ```
    [
        {
            'text': 'James Bond',
            'label': 'person',
            'score': 0.9012733697891235,
            'start': 5,
            'end': 15
        }
    ]
    ```
    """

    def __init__(self, model_path: str, onnx_path: Optional[str] = "onnx/model.onnx"):
        self.model = PyFastGliNER(model_path, onnx_path)

    def predict_entities(
        self,
        texts: Union[str, List[str]],
        labels: List[str],
    ) -> List[List[dict]]:
        """Predict entities in the given texts.

        Args:
            texts (List[str]): A list of texts to predict entities for.

        Returns:
            List[List[dict]]: A list of lists of dictionaries containing the predicted entities.
        """
        if isinstance(texts, str):
            texts = [texts]

        results = self.model.predict_entities(texts, labels)

        if len(results) == 1:
            return results[0]
        return results

    @classmethod
    def from_pretrained(cls, model_id: str, onnx_path: Optional[str] = "onnx/model.onnx", **kwargs) -> "FastGLiNER":
        """Load a pretrained model from the Hugging Face Model Hub.

        Args:
            model_id (str): The name of the model on the Hugging Face Model Hub or a local path.
            onnx_path (str, optional): The path to the onnx model in the model directory. Defaults to "onnx/model.onnx".
            **kwargs: extra args for the `huggingface_hub.snapshot_download` method.

        Returns:
            FastGLiNER: An instance of the FastGLiNER class.

        Raises:
            [`~utils.RepositoryNotFoundError`]
                If the repository to download from cannot be found. This may be because it doesn't exist,
                or because it is set to `private` and you do not have access.
            [`~utils.RevisionNotFoundError`]
                If the revision to download from cannot be found.
            [`EnvironmentError`](https://docs.python.org/3/library/exceptions.html#EnvironmentError)
                If `token=True` and the token cannot be found.
            [`OSError`](https://docs.python.org/3/library/exceptions.html#OSError) if
                ETag cannot be determined.
            [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
                if some parameter value is invalid.
        """
        model_dir = Path(model_id)

        if not model_dir.exists():
            model_dir = snapshot_download(repo_id=model_id, allow_patterns=["*.json", onnx_path], **kwargs)
        else:
            model_file = model_dir / onnx_path

            if not model_file.exists():
                raise FileNotFoundError(f"The ONNX model can't be loaded from {model_file}.")

            config_file = model_dir / "gliner_config.json"

            if not config_file.exists():
                raise FileNotFoundError(f"The config file can't be loaded from {config_file}.")

            model_dir = str(model_dir.resolve())

        return cls(model_dir, onnx_path)


__version__ = "0.1.1"

__all__ = ["FastGLiNER"]

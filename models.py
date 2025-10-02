from abc import ABC, abstractmethod
from typing import Any, Dict
from dataclasses import dataclass, field

from transformers import pipeline
from PIL import Image

from .decorators import timeit, ensure_initialized


# ---------------------------
# Mixins (Multiple Inheritance)
# ---------------------------
class LoggingMixin:
    def log(self, message: str) -> None:
        if getattr(self, "_logger_enabled", True):
            print(f"[LOG] {self.__class__.__name__}: {message}")


# ---------------------------------
# Abstract Base Runner (Encapsulation)
# ---------------------------------
@dataclass
class BaseModelRunner(ABC, LoggingMixin):
    model_id: str
    task: str
    category: str               # e.g., "Text", "Vision", "Audio"
    short_description: str = "" # brief human description

    _initialized: bool = field(default=False, init=False, repr=False)
    _pipe: Any = field(default=None, init=False, repr=False)
    last_runtime_s: float = field(default=0.0, init=False)

    @property
    def initialized(self) -> bool:
        return self._initialized

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def run(self, user_input: Any, **kwargs) -> Dict[str, Any]:
        ...

    def meta(self) -> Dict[str, str]:
        return {
            "name": self.model_id,
            "category": self.category,
            "short_description": self.short_description or self.describe(),
        }

    def describe(self) -> str:
        """Base description; overridden in subclasses."""
        return f"Generic model runner for task='{self.task}', model='{self.model_id}'."


# ---------------------------
# Text Model Runner (Polymorphism + Overriding)
# ---------------------------
class TextModelRunner(BaseModelRunner):
    @timeit
    def load(self) -> None:
        self.log(f"Loading text pipeline: task={self.task}, model={self.model_id}")
        self._pipe = pipeline(task=self.task, model=self.model_id)
        self._initialized = True

    @ensure_initialized
    @timeit
    def run(self, user_input: str, **kwargs) -> Dict[str, Any]:
        self.log(f"Running text model on input length={len(user_input)}")
        results = self._pipe(user_input)
        return {
            "results": results,
            "runtime_s": self.last_runtime_s,
            "model": self.model_id,
            "task": self.task,
        }

    def describe(self) -> str:
        return (
            "TextModelRunner overrides describe(): runs a text pipeline (e.g., sentiment-analysis).\n"
            f"Model: {self.model_id}\nTask: {self.task}"
        )


# ---------------------------
# Image Model Runner (Polymorphism + Overriding)
# ---------------------------
class ImageModelRunner(BaseModelRunner):
    def _preprocess(self, img: Image.Image) -> Image.Image:
        # Overridable preprocessing step; identity for now
        return img

    @timeit
    def load(self) -> None:
        self.log(f"Loading image pipeline: task={self.task}, model={self.model_id}")
        self._pipe = pipeline(task=self.task, model=self.model_id)
        self._initialized = True

    @ensure_initialized
    @timeit
    def run(self, user_input: Any, **kwargs) -> Dict[str, Any]:
        # Accept a path or a PIL.Image
        if isinstance(user_input, (str,)):
            img = Image.open(user_input).convert("RGB")
        elif isinstance(user_input, Image.Image):
            img = user_input
        else:
            raise TypeError("user_input must be a path or PIL.Image.Image for image tasks.")

        img = self._preprocess(img)
        top_k = int(kwargs.get("top_k", 5))
        self.log(f"Running image model (top_k={top_k}) on provided image...")
        results = self._pipe(img, top_k=top_k)
        return {
            "results": results,
            "runtime_s": self.last_runtime_s,
            "model": self.model_id,
            "task": self.task,
        }

    def describe(self) -> str:
        return (
            "ImageModelRunner overrides describe(): runs an image pipeline (e.g., image-classification).\n"
            f"Model: {self.model_id}\nTask: {self.task}"
        )


# ---------------------------
# Factory helpers (two models from different categories)
# ---------------------------
def make_text_sentiment_runner() -> TextModelRunner:
    return TextModelRunner(
        model_id="distilbert-base-uncased-finetuned-sst-2-english",
        task="sentiment-analysis",
        category="Text",
        short_description="Binary sentiment (POSITIVE/NEGATIVE) for short English text.",
    )

def make_image_classifier_runner() -> ImageModelRunner:
    return ImageModelRunner(
        model_id="google/vit-base-patch16-224",
        task="image-classification",
        category="Vision",
        short_description="ImageNet-style image classification using ViT base.",
    )

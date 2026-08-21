"""Abstract base interface for all statistical and ML-based imputers."""

from abc import ABC, abstractmethod
from typing import Any, Self

import pandas as pd


class BaseImputer(ABC):
    """Abstract base class for modular dataset imputation algorithms."""

    def __init__(self) -> None:
        self.is_fitted: bool = False
        self.imputation_parameters: dict[str, Any] = {}

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> Self:
        """Fit imputation parameters (statistics, models, embeddings) from observed data.

        Args:
            df: Input training/reference DataFrame with observed and missing values.

        Returns:
            Self: The fitted imputer instance.
        """
        ...

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in the target dataset using pre-fitted parameters.

        Args:
            df: Input DataFrame containing missing values to impute.

        Returns:
            pd.DataFrame: Imputed DataFrame with missing values filled.
        """
        ...

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit parameters on dataset and return the imputed DataFrame."""
        return self.fit(df).transform(df)

"""
Base class for expert policy wrappers.

This module provides a unified interface for different expert policies.
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Optional


class ExpertWrapper(ABC):
    """
    Base class for wrapping expert policies.
    
    Provides a unified interface for getting actions from different expert models.
    """
    
    def __init__(self, checkpoint_path: str, device: Optional[torch.device] = None):
        """
        Initialize the expert wrapper.
        
        Args:
            checkpoint_path: Path to the expert model checkpoint
            device: Torch device (cpu/cuda). If None, uses CPU.
        """
        self.checkpoint_path = checkpoint_path
        self.device = device if device is not None else torch.device("cpu")
        self._policy_net = None
        self._loaded = False
        
    @abstractmethod
    def load(self) -> None:
        """
        Load the expert model from checkpoint.
        
        Must be implemented by subclasses.
        """
        raise NotImplementedError
        
    @abstractmethod
    def get_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Get action from expert for given observation.
        
        Args:
            obs: Observation array (1D or 2D)
            deterministic: If True, return mean action. If False, sample from distribution.
            
        Returns:
            Action array (1D)
        """
        raise NotImplementedError
        
    def get_action_batch(self, obs_batch: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Get actions from expert for a batch of observations.
        
        Args:
            obs_batch: Batch of observations (2D array: [batch_size, obs_dim])
            deterministic: If True, return mean actions. If False, sample from distribution.
            
        Returns:
            Batch of actions (2D array: [batch_size, action_dim])
        """
        if not self._loaded:
            self.load()
            
        actions = []
        for obs in obs_batch:
            action = self.get_action(obs, deterministic)
            actions.append(action)
        return np.array(actions)
    
    @property
    def action_dim(self) -> int:
        """Get action dimension."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._action_dim
    
    @property
    def obs_dim(self) -> int:
        """Get observation dimension."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._obs_dim


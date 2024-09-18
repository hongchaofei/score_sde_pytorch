# -*- coding: utf-8 -*-
"""
Created on 2024/9/18
@project: score_sde_pytorch
@filename: utils
@author: Hong Chaofei
@contact: hongchf@gmail.com

@description:
"""
import torch


def get_data_normalizer(config):
	return lambda x: (x - config.data.mean) / config.data.std


def get_data_inverse_normalizer(config):
	return lambda x: x * config.data.std + config.data.mean


def get_model_fn(model, train=False):
	"""Create a function to give the output of the score-based model.

	Args:
	  model: The score model.
	  train: `True` for training and `False` for evaluation.

	Returns:
	  A model function.
	"""

	def model_fn(x, labels):
		"""Compute the output of the score-based model.

		Args:
		  x: A mini-batch of input data.
		  labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
			for different models.

		Returns:
		  A tuple of (model output, new mutable states)
		"""
		if not train:
			model.eval()
			return model(x, labels)
		else:
			model.train()
			return model(x, labels)

	return model_fn


def get_score_fn(sde, model, train=False):
	"""Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

	Args:
	  sde: An `sde_lib.SDE` object that represents the forward SDE.
	  model: A score model.
	  train: `True` for training and `False` for evaluation.
	  continuous: If `True`, the score-based model is expected to directly take continuous time steps.

	Returns:
	  A score function.
	"""
	model_fn = get_model_fn(model, train=train)

	# assert isinstance(sde, sde_lib.VESDE)
	def score_fn(x, t):
		labels = sde.marginal_prob(torch.zeros_like(x), t)[1]

		score = model_fn(x, labels)
		return score

	return score_fn

def restore_checkpoint(ckpt_dir, state, device):
	loaded_state = torch.load(ckpt_dir, map_location=device)
	state['optimizer'].load_state_dict(loaded_state['optimizer'])
	state['model'].load_state_dict(loaded_state['model'], strict=False)
	state['ema'].load_state_dict(loaded_state['ema'])
	state['step'] = loaded_state['step']
	return state


def save_checkpoint(ckpt_dir, state):
	saved_state = {
	'optimizer': state['optimizer'].state_dict(),
	'model': state['model'].state_dict(),
	'ema': state['ema'].state_dict(),
	'step': state['step']
	}
	torch.save(saved_state, ckpt_dir)

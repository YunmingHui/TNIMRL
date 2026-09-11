import torch
import numpy as np


def message_aggregator(node_ids, messages):
  """Only keep the last message for each node"""
  unique_node_ids = np.unique(node_ids)
  unique_messages = []
  unique_timestamps = []

  to_update_node_ids = []

  for node_id in unique_node_ids:
      if len(messages[node_id]) > 0:
          to_update_node_ids.append(node_id)
          unique_messages.append(messages[node_id][-1][0])
          unique_timestamps.append(messages[node_id][-1][1])

  unique_messages = torch.stack(unique_messages) if len(
      to_update_node_ids) > 0 else []
  unique_timestamps = torch.stack(unique_timestamps) if len(
      to_update_node_ids) > 0 else []

  return to_update_node_ids, unique_messages, unique_timestamps

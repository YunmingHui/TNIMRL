from torch import nn


class MemoryUpdater(nn.Module):
  def __init__(self, message_dimension, memory_dimension, device):
    super(MemoryUpdater, self).__init__()
    self.message_dimension = message_dimension
    self.device = device
    self.memory_updater = nn.GRUCell(input_size=message_dimension,
                                     hidden_size=memory_dimension).to(device)

  def update_memory(self, memory_original, unique_node_ids, unique_messages, timestamps):
    if len(unique_node_ids) <= 0:
      return

    assert (memory_original.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
        "update memory to time in the past"

    memory = memory_original.get_memory(unique_node_ids)
    memory_original.last_update[unique_node_ids] = timestamps

    updated_memory = self.memory_updater(unique_messages, memory)

    memory_original.set_memory(unique_node_ids, updated_memory)

  def get_updated_memory(self, memory_original, unique_node_ids, unique_messages, timestamps):
    if len(unique_node_ids) <= 0:
      return memory_original.memory.data.clone(), memory_original.last_update.data.clone()

    assert (memory_original.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
        "update memory to time in the past"

    updated_memory = memory_original.memory.data.clone()
    updated_memory[unique_node_ids] = self.memory_updater(
        unique_messages, updated_memory[unique_node_ids])

    updated_last_update = memory_original.last_update.data.clone()
    updated_last_update[unique_node_ids] = timestamps

    return updated_memory, updated_last_update


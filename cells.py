import tensorflow as tf

class DeviceWrapper(tf.contrib.rnn.RNNCell):
  def __init__(self, cell, device_id):
    self._cell = cell
    self._device_id = device_id
    print "Assign to device: %s" % device_id

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, scope=None):
    with tf.device(self._device_id):
      return self._cell(inputs, state, scope)

class ResidualMultiRNNCell(tf.contrib.rnn.RNNCell):
  def __init__(self, cells):
    self._cells = cells

  @property
  def state_size(self):
    return tuple(cell.state_size for cell in self._cells)

  @property
  def output_size(self):
    return self._cells[-1].output_size

  def __call__(self, inputs, state, scope=None):
    with tf.variable_scope(scope or "multi_rnn_cell"):
      cur_state_pos = 0
      cur_inp = inputs
      new_states = []
      for i, cell in enumerate(self._cells):
        with tf.variable_scope("cell_%d" % i):
          cur_state = state[i]
          cur_outp, new_state = cell(cur_inp, cur_state)
          if i > 0:
            cur_inp = tf.add(cur_inp, cur_outp)
          else:
            cur_inp = cur_outp
          new_states.append(new_state)
    new_states = tuple(new_states)
    return cur_inp, new_states

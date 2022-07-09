Restoring a Keras model from a checkpoint can be done using a code similar to the [`initialize`](https://github.com/tensorflow/models/blob/7d45e7b925a0d5c360bada2f5b577694031e5478/official/core/base_task.py#L111) function in the tensorflow official models repository:
```python
  def initialize(self, model: tf.keras.Model):
    """[Optional] A callback function used as CheckpointManager's init_fn.

    This function will be called when no checkpoint is found for the model.
    If there is a checkpoint, the checkpoint will be loaded and this function
    will not be called. You can use this callback function to load a pretrained
    checkpoint, saved under a directory other than the model_dir.

    Args:
      model: The keras.Model built or used by this task.
    """
    ckpt_dir_or_file = self.task_config.init_checkpoint
    logging.info("Trying to load pretrained checkpoint from %s",
                 ckpt_dir_or_file)
    if tf.io.gfile.isdir(ckpt_dir_or_file):
      ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)
    if not ckpt_dir_or_file:
      return

    if hasattr(model, "checkpoint_items"):
      checkpoint_items = model.checkpoint_items
    else:
      checkpoint_items = dict(model=model)
    ckpt = tf.train.Checkpoint(**checkpoint_items)
    status = ckpt.read(ckpt_dir_or_file)
    status.expect_partial().assert_existing_objects_matched()
    logging.info("Finished loading pretrained checkpoint from %s",
                 ckpt_dir_or_file)
```

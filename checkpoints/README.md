# Checkpoints

Saved policy and value network weights from training runs. Files containing
`Thesis` in the name are the policies used for the final thesis results. Files
containing `CNN` use the camera based vision pipeline. Load a checkpoint in PyTorch
with `ac.load_state_dict(torch.load(<path>))`.

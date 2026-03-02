## PPO Shared Transformer Model Card



- **`seq_feature_dim`**: Number of input features per day in the 60‑day time window.
- **`portfolio_dim`**: Dimension of the portfolio state vector (e.g., positions, cash, risk stats).
- **`action_dim`**: Number of discrete actions for the trading policy (size of the action space).
- **`max_seq_len`**: Length of the historical window in days; set to 60 for a 60‑day history.
- **`d_model`**: Hidden size of the Transformer encoder (embedding dimension of each token).
- **`nhead`**: Number of attention heads in each Transformer layer.
- **`num_layers`**: Number of stacked Transformer encoder layers.
- **`dim_feedforward`**: Hidden size of the feedforward network inside each Transformer layer.
- **`dropout`**: Dropout rate used in the encoder and MLPs for regularization.
- **`latent_dim`**: Dimension of the shared latent state passed to the actor and critic heads.


"""Configuration classes for Hebbian experiments."""

class NetworkConfig:
    """Network architecture and learning parameters."""
    def __init__(
        self,
        hidden_sizes,  # List[int] or int
        n_colors=None,  # Optional[int]
        p=2.0,
        delta=0.4,
        k=2,
        allow_pathway_interaction=False,
        signific_p_multiplier=5.0
    ):
        self.hidden_sizes = [hidden_sizes] if isinstance(hidden_sizes, int) else hidden_sizes
        self.n_colors = n_colors
        self.p = p
        self.delta = delta
        self.k = k
        self.allow_pathway_interaction = allow_pathway_interaction
        self.signific_p_multiplier = signific_p_multiplier
        
    @property
    def layer_sizes(self):
        """Get layer sizes including input layer."""
        input_size = 784 * (1 + self.n_colors if self.n_colors else 1)
        return [input_size] + self.hidden_sizes
        
    @property
    def signific_size(self):
        """Get size of significance layer."""
        return 10 + (self.n_colors if self.n_colors else 0)
        
    @property
    def hidden_signific_connections(self):
        """Get list of which layers have signific connections."""
        return [True] * len(self.hidden_sizes)

class TrainingConfig:
    """Training hyperparameters."""
    def __init__(
        self,
        n_epochs=20,
        batch_size=100,
        eval_batch_size=1000,
        learning_rate=0.02,
        train_ratio=0.8,  # For compositional holdout
        save_dir='results'
    ):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.learning_rate = learning_rate
        self.train_ratio = train_ratio
        self.save_dir = save_dir 
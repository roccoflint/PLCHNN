import numpy as np
from tqdm import tqdm

class Decoder:
    """
    Decoder employing the Krotov & Hopfield approach with:
        - Power of activation function n = 4.5
        - Loss function parameter m = 6
        - β = 0.1
    Uses an Adam optimizer with a stepped learning rate schedule:
        - 0.001 for first 100 epochs
        - 0.0005 for next 50 epochs
        - 0.0001 for next 50 epochs
        - 0.00005 for next 50 epochs
        - 0.00001 for final 50 epochs
    Total epochs = 300, mini-batch size = 100.
    """

    def __init__(self, input_dim, output_dim=10, n=4.5, m=6.0, beta=0.1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n = n
        self.m = m
        self.beta = beta

        # Initialize weights: shape [input_dim, output_dim]
        # Using small random normal init (could tweak scale as desired)
        self.W = 0.01 * np.random.randn(self.input_dim, self.output_dim)

        # Adam state
        self.m_adam = np.zeros_like(self.W)
        self.v_adam = np.zeros_like(self.W)
        self.t = 0  # Adam time step

    def _phi(self, u):
        """
        Krotov-Hopfield activation:
        phi(u) = sign(u) * |u|^(n-1).
        """
        return np.sign(u) * np.abs(u)**(self.n - 1)

    def _forward(self, X):
        """
        Forward pass with nonlinearity applied to outputs, exactly as in KH.
        Here, X shape is [batch_size, input_dim].
        Returns shape [batch_size, output_dim].
        """
        # Linear combination first
        logits = X @ self.W
        # Apply nonlinearity to outputs
        return np.sign(logits) * np.abs(logits)**(self.n - 1)

    def _loss_and_grad(self, X_batch, y_batch):
        """
        Implements the Krotov & Hopfield classification loss (Eqn. 12 style),
        with label vectors in {+1, -1}, margin exponent 'm', and L2 penalty 'beta'.
        """
        batch_size = X_batch.shape[0]

        # 1) Forward pass - get nonlinear outputs
        outputs = self._forward(X_batch)  # [batch_size, output_dim]

        # 2) Build {+1, -1} targets, per Krotov & Hopfield
        target_pm1 = -np.ones((batch_size, self.output_dim), dtype=np.float32)
        target_pm1[np.arange(batch_size), y_batch] = 1.0

        # 3) Compute the Eqn. (12) loss: sum(|c_alpha - t_alpha|^m)
        diffs = outputs - target_pm1
        abs_diffs = np.abs(diffs) ** self.m
        loss = np.sum(abs_diffs) / batch_size  # Average over batch

        # 4) Add L2-regularization term
        loss += 0.5 * self.beta * np.sum(self.W**2)

        # 5) Compute gradient through the nonlinearity
        # First get raw logits for gradient computation
        logits = X_batch @ self.W
        # Gradient of nonlinearity: d/dx(sign(x)|x|^(n-1)) = (n-1)|x|^(n-2)
        nonlin_grad = (self.n - 1) * np.abs(logits)**(self.n - 2)
        
        # Gradient of margin loss through nonlinearity
        signs = np.sign(diffs)
        mags = np.abs(diffs) ** (self.m - 1)
        doutputs = (self.m * mags * signs) / batch_size
        
        # Chain rule through nonlinearity
        dlogits = doutputs * nonlin_grad

        # 6) Final gradient computation
        dW = X_batch.T @ dlogits + self.beta * self.W

        return loss, dW

    def _adam_update(self, dW, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Perform one step of Adam on self.W using dW, updating internal states.
        """
        self.t += 1
        self.m_adam = beta1 * self.m_adam + (1 - beta1) * dW
        self.v_adam = beta2 * self.v_adam + (1 - beta2) * (dW**2)

        m_hat = self.m_adam / (1 - beta1**self.t)
        v_hat = self.v_adam / (1 - beta2**self.t)

        self.W -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def fit(self, X, y, epochs=300, batch_size=100):
        """
        Train the decoder for 'epochs' using mini-batches of 'batch_size.'
        Learning rate schedule:
            - 0.001 for first 100 epochs
            - 0.0005 for next 50
            - 0.0001 for next 50
            - 0.00005 for next 50
            - 0.00001 for final 50
        """
        idxs = np.arange(X.shape[0])

        for epoch in tqdm(range(epochs), desc='Training decoder'):
            # Determine current LR
            if epoch < 100:
                lr = 0.001
            elif epoch < 150:
                lr = 0.0005
            elif epoch < 200:
                lr = 0.0001
            elif epoch < 250:
                lr = 0.00005
            else:
                lr = 0.00001

            np.random.shuffle(idxs)
            X = X[idxs]
            y = y[idxs]

            for start in range(0, X.shape[0], batch_size):
                end = start + batch_size
                X_batch = X[start:end]
                y_batch = y[start:end]

                loss, dW = self._loss_and_grad(X_batch, y_batch)
                self._adam_update(dW, lr=lr)

    def predict(self, X):
        """
        Predict class index by taking argmax of logits.
        """
        logits = self._forward(X)
        preds = np.argmax(logits, axis=1)
        return preds

    def score(self, X, y):
        """
        Return classification accuracy on (X, y).
        """
        preds = self.predict(X)
        return np.mean(preds == y)
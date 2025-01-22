import numpy as np
try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    cp = np

from .utils import cosine_similarity, to_device

class HebbianLayer:
    """Single layer implementing KH's Hebbian/anti-Hebbian learning exactly."""
    def __init__(self, input_size, output_size, p=2.0, delta=0.4, k=2, mu=0.0, sigma=1.0):
        self.input_size = input_size
        self.output_size = output_size
        self.p = p
        self.delta = delta
        self.k = k
        
        # Initialize weights - no normalization yet, exactly as in KH
        self.W = None
        self.xp = None
        
    def _initialize_weights(self):
        """Initialize weights exactly as in KH."""
        return self.xp.random.normal(0.0, 1.0, (self.output_size, self.input_size))
    
    def forward(self, x, signific_input=None, allow_pathway_interaction=False):
        """Forward pass exactly matching KH's implementation."""
        sig = self.xp.sign(self.W)
        total_input = self.xp.dot(sig * self.xp.abs(self.W)**(self.p - 1), x.T)
        
        # If pathway interaction allowed and signific input provided, add it before nonlinearity
        if allow_pathway_interaction and signific_input is not None:
            total_input += signific_input
            
        return total_input
    
    def update(self, x, learning_rate):
        """Update weights using KH's exact learning rule."""
        # Forward pass
        tot_input = self.forward(x)
        
        # Learning rule exactly as in KH
        y = self.xp.argsort(tot_input, axis=0)
        yl = self.xp.zeros_like(tot_input)
        yl[y[-1, :], self.xp.arange(x.shape[0])] = 1.0
        yl[y[-self.k, :], self.xp.arange(x.shape[0])] = -self.delta
        
        # Weight updates exactly as in KH
        xx = self.xp.sum(yl * tot_input, axis=1)
        ds = self.xp.dot(yl, x) - xx[:, None] * self.W
        
        # Normalize updates exactly as in KH
        nc = self.xp.amax(self.xp.abs(ds))
        if nc < 1e-30:
            nc = 1e-30
        
        # Update weights - no extra normalization
        self.W += learning_rate * (ds / nc)
    
    def to_device(self, device, xp):
        """Move layer to specified device."""
        self.xp = xp
        if self.W is None:
            self.W = self._initialize_weights()
        else:
            self.W = to_device(self.W, device)

class HebbianNetwork:
    """Multi-layer Hebbian network with signific pathway, implementing KH's learning rules exactly."""
    def __init__(self, layer_sizes, signific_size=None, hidden_signific_connections=None, 
                 p=2.0, delta=0.4, k=2, device='gpu', allow_pathway_interaction=False,
                 signific_p_multiplier=5):
        self.device = 'cpu' if not HAS_CUDA or device == 'cpu' else 'gpu'
        self.xp = np if self.device == 'cpu' else cp
        
        # Core parameters
        self.p = p
        self.delta = delta
        self.k = k
        self.allow_pathway_interaction = allow_pathway_interaction
        self.signific_p_multiplier = signific_p_multiplier
        
        # Create layers
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = HebbianLayer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                p=p,
                delta=delta,
                k=k
            )
            layer.to_device(self.device, self.xp)
            self.layers.append(layer)
            
        # Initialize signific pathway if specified
        self.signific_size = signific_size
        if signific_size is not None:
            if hidden_signific_connections is None:
                hidden_signific_connections = [True] * (len(layer_sizes) - 1)
            elif len(hidden_signific_connections) != len(layer_sizes) - 1:
                raise ValueError("hidden_signific_connections must match number of hidden layers")
                
            self.hidden_signific_connections = hidden_signific_connections
            self.signific_weights = []
            
            for i, is_connected in enumerate(hidden_signific_connections):
                if is_connected:
                    # Initialize exactly as in KH
                    self.signific_weights.append(
                        self.xp.random.normal(0.0, 1.0, (signific_size, layer_sizes[i+1]))
                    )
                else:
                    self.signific_weights.append(None)
        else:
            self.hidden_signific_connections = None
            self.signific_weights = None
    
    def forward(self, x, signific_input=None, allow_pathway_interaction=False):
        """Forward pass through network, implementing KH's activation exactly."""
        x = to_device(x, self.device)
        
        # Forward through layers
        h = x
        activations = [h]
        for layer in self.layers:
            h = layer.forward(h, signific_input, allow_pathway_interaction).T
            activations.append(h)
            
        if not allow_pathway_interaction and self.signific_weights is not None:
            # Project hidden activations to signific layer
            signific_inputs = []
            num_connected = sum(1 for is_connected in self.hidden_signific_connections if is_connected)
            
            if num_connected > 0:
                for i, (is_connected, W) in enumerate(zip(self.hidden_signific_connections, self.signific_weights)):
                    if is_connected:
                        # Apply KH's exact activation function with increased power
                        sig = self.xp.sign(W)
                        abs_weights = self.xp.abs(W)**(self.p * self.signific_p_multiplier - 1)
                        signific_input = self.xp.dot(sig * abs_weights, activations[i+1].T)
                        signific_inputs.append(signific_input)
                
                # Stack and combine inputs without scaling
                h = self.xp.sum(self.xp.stack(signific_inputs, axis=0), axis=0)
            else:
                h = self.xp.zeros((self.signific_size, x.shape[0]), dtype=x.dtype)
        else:
            h = h.T
            
        return h if allow_pathway_interaction else to_device(h, self.device)
    
    def update(self, x, signific_pattern=None, learning_rate=None):
        """Update weights using KH's exact learning rule."""
        x = to_device(x, self.device)
        if signific_pattern is not None:
            signific_pattern = to_device(signific_pattern, self.device)
        
        # Forward and collect activations
        activations = [x]
        h = x
        for layer in self.layers:
            # If pathway interaction is allowed, get signific input for this layer
            signific_input = None
            if self.allow_pathway_interaction and signific_pattern is not None:
                signific_acts = self._get_hidden_activations(signific_pattern, is_signific=True)
                if len(signific_acts) > len(activations)-1:  # If this layer has signific connections
                    signific_input = signific_acts[len(activations)-1].T
            
            # Forward pass with potential signific input
            h = layer.forward(h, signific_input, self.allow_pathway_interaction).T
            activations.append(h)
        
        # Update each layer using KH's rule
        for i, layer in enumerate(self.layers):
            layer.update(activations[i], learning_rate)
            
        # Update signific pathway using KH's rule with increased power
        if signific_pattern is not None and self.signific_weights is not None:
            for i, (is_connected, W) in enumerate(zip(self.hidden_signific_connections, self.signific_weights)):
                if is_connected:
                    hidden_activations = activations[i+1]
                    
                    # Forward pass with increased power
                    sig = self.xp.sign(W)
                    tot_input = self.xp.dot(sig * self.xp.abs(W)**(self.p * self.signific_p_multiplier - 1), hidden_activations.T)
                    
                    # Learning rule treating digit-color pairs as atomic units
                    yl = self.xp.zeros_like(tot_input)
                    
                    # Split signific pattern into components (digit and color)
                    n_digits = 10  # First 10 positions are digits
                    digit_pattern = signific_pattern[:, :n_digits]
                    color_pattern = signific_pattern[:, n_digits:]
                    
                    # Get target indices for each component
                    digit_targets = self.xp.argmax(digit_pattern, axis=1)
                    color_targets = self.xp.argmax(color_pattern, axis=1) + n_digits
                    batch_indices = self.xp.arange(x.shape[0])
                    
                    # Single Hebbian update for the digit-color pair
                    yl[digit_targets, batch_indices] = 0.5  # Split +1.0 between digit and color
                    yl[color_targets, batch_indices] = 0.5
                    
                    # Anti-Hebbian updates treating digit-color as a unit
                    # Create mask for all positions except the target digit-color pair
                    mask = self.xp.ones_like(tot_input, dtype=bool)
                    mask[digit_targets, batch_indices] = False
                    mask[color_targets, batch_indices] = False
                    
                    # Find k strongest competitors among all positions
                    inputs = self.xp.where(mask, tot_input, -self.xp.inf)
                    competitors = self.xp.argsort(inputs, axis=0)[-self.k:]
                    yl[competitors, batch_indices] = -self.delta / self.k  # Distribute delta across k competitors
                    
                    # Update weights exactly as KH
                    xx = self.xp.sum(yl * tot_input, axis=1)
                    ds = self.xp.dot(yl, hidden_activations) - xx[:, None] * W
                    nc = self.xp.amax(self.xp.abs(ds))
                    if nc < 1e-30:
                        nc = 1e-30
                    W += learning_rate * (ds / nc)
                    
                    # Normalize weights to unit Lp norm with increased power
                    norms = self.xp.sum(self.xp.abs(W) ** (self.p * self.signific_p_multiplier), axis=1) ** (1.0 / (self.p * self.signific_p_multiplier))
                    W /= norms[:, None]
    
    def rts_classify(self, x, component_sizes):
        """Classify inputs using RtS pathway with separate handling for digits and colors."""
        if self.signific_weights is None:
            raise ValueError("Network does not have signific pathway")
            
        # Get signific activations
        signific_activations = self.forward(x)
        
        # Split activations into digit and color components
        n_digits = 10  # First 10 positions are digits
        digit_activations = signific_activations[:n_digits]
        color_activations = signific_activations[n_digits:]
        
        # Apply nonlinearity with same power for both
        power = self.p * self.signific_p_multiplier - 1
        
        # Process digits
        sig_digits = self.xp.sign(digit_activations)
        act_digits = sig_digits * self.xp.abs(digit_activations)**power
        act_digits = act_digits / (self.xp.sum(self.xp.abs(act_digits), axis=0, keepdims=True) + 1e-8)
        digit_classes = self.xp.argmax(act_digits, axis=0)
        
        # Process colors
        sig_colors = self.xp.sign(color_activations)
        act_colors = sig_colors * self.xp.abs(color_activations)**power
        act_colors = act_colors / (self.xp.sum(self.xp.abs(act_colors), axis=0, keepdims=True) + 1e-8)
        color_classes = self.xp.argmax(act_colors, axis=0)
        
        return [digit_classes, color_classes]
    
    def to_device(self, device):
        """Move network to specified device."""
        if device not in ['cpu', 'gpu']:
            raise ValueError("device must be 'cpu' or 'gpu'")
        if device == self.device:
            return
        
        self.device = device
        self.xp = np if device == 'cpu' else cp
        for layer in self.layers:
            layer.to_device(device, self.xp)

    def to_cpu(self):
        """Transfer network to CPU."""
        self.to_device('cpu')
            
    def to_gpu(self):
        """Transfer network to GPU if available."""
        if HAS_CUDA:
            self.to_device('gpu')

    def _get_hidden_activations(self, x, is_signific=False):
        """Get activations for hidden layers connected to signific pathway."""
        if is_signific:
            # Project signific input to connected hidden layers with increased power
            activations = []
            for i, (is_connected, W) in enumerate(zip(self.hidden_signific_connections, self.signific_weights)):
                if is_connected:
                    sig = self.xp.sign(W)
                    # Split signific input into digit and color components
                    n_digits = 10
                    digit_input = x[:, :n_digits]
                    color_input = x[:, n_digits:]
                    
                    # Project each component separately with increased power
                    W_digits = W[:n_digits]
                    W_colors = W[n_digits:]
                    
                    # Use increased power for both components
                    power = self.p * self.signific_p_multiplier - 1
                    
                    # Project digits
                    sig_digits = self.xp.sign(W_digits)
                    act_digits = self.xp.dot(digit_input, sig_digits * self.xp.abs(W_digits)**power)
                    
                    # Project colors
                    sig_colors = self.xp.sign(W_colors)
                    act_colors = self.xp.dot(color_input, sig_colors * self.xp.abs(W_colors)**power)
                    
                    # Combine activations (sum since they target same hidden units)
                    act = act_digits + act_colors
                    activations.append(act)
            return activations
        else:
            # Forward pass through network, keeping only connected hidden layers
            h = x
            activations = []
            for i, layer in enumerate(self.layers):
                h = layer.forward(h).T
                if self.hidden_signific_connections[i]:
                    activations.append(h)
            return activations

    def str_rec(self, signific_input, max_iter=500, init_lr=5.0, tol=1e-5, patience=10, min_steps=50, component_sizes=None):
        """Reconstruct input from signific input using either direct weight reversal or gradient descent.
        
        Args:
            signific_input: Input to signific pathway
            max_iter: Maximum iterations for gradient descent
            init_lr: Initial learning rate
            tol: Tolerance for improvement in loss
            patience: Number of iterations without improvement before early stopping
            min_steps: Minimum number of steps before early stopping
            component_sizes: List of sizes for each component (e.g. [10, 3] for digits and colors)
                           If provided, optimizes each component separately
        """
        # First compute target activations from signific input
        target_acts = self._get_hidden_activations(signific_input, is_signific=True)
        
        # For single hidden layer, use direct weight reversal
        if len(self.layers) == 1 and len(self.signific_weights) == 1:
            # Step 1: Project signific input to hidden layer using signific weights with increased power
            W_sig = self.signific_weights[0]  # Shape: (signific_size, hidden_size)
            
            # Split signific input and weights into digit and color components
            n_digits = 10
            digit_input = signific_input[:, :n_digits]
            color_input = signific_input[:, n_digits:]
            W_digits = W_sig[:n_digits]
            W_colors = W_sig[n_digits:]
            
            # Get number of colors from signific input size
            n_colors = color_input.shape[1]
            
            # Project each component separately with increased power
            power = self.p * self.signific_p_multiplier - 1
            
            # Project digits to get digit-specific hidden activations
            sig_digits = self.xp.sign(W_digits)
            hidden_digits = self.xp.dot(digit_input, sig_digits * self.xp.abs(W_digits)**power)
            
            # Project colors to get color-specific hidden activations
            sig_colors = self.xp.sign(W_colors)
            hidden_colors = self.xp.dot(color_input, sig_colors * self.xp.abs(W_colors)**power)
            
            # Step 2: Project hidden activations back to input using referential weights
            W_ref = self.layers[0].W  # Shape: (hidden_size, input_size)
            sig = self.xp.sign(W_ref)
            W_ref_nonlinear = sig * self.xp.abs(W_ref)**(self.p - 1)
            
            # First reconstruct the digit pattern using digit hidden activations
            x_digits = self.xp.dot(hidden_digits, W_ref_nonlinear[:, :784])  # Only project to intensity channel
            
            # Normalize digit pattern
            x_digits = x_digits - self.xp.min(x_digits, axis=1, keepdims=True)
            x_digits = x_digits / (self.xp.max(x_digits, axis=1, keepdims=True) + 1e-8)
            
            # Initialize output with zeros
            x = self.xp.zeros((signific_input.shape[0], 784 * (1 + n_colors)), dtype=x_digits.dtype)
            
            # Copy normalized digit pattern to intensity channel
            x[:, :784] = x_digits
            
            # Get color weights
            color_weights = self.xp.abs(color_input)  # Shape: (batch, n_colors)
            
            # For each color channel, copy the digit pattern scaled by color weight
            for c in range(n_colors):
                if color_weights[0, c] > 0:  # If this color is active
                    start_idx = 784 * (c + 1)
                    end_idx = start_idx + 784
                    x[:, start_idx:end_idx] = x_digits * float(color_weights[0, c])
            
            return x
        
        # For multiple hidden layers or compositional inputs, optimize referential input
        print("\nUsing gradient descent for reconstruction...")
        target_vector = self.xp.concatenate([act.flatten() for act in target_acts])
        
        # Initialize with random values in [0, 1]
        x = self.xp.random.uniform(0, 1, (1, self.layers[0].input_size))
        best_x = x.copy()
        best_loss = float('inf')
        no_improvement = 0
        momentum = self.xp.zeros_like(x)
        beta = 0.9  # Momentum coefficient
        
        # If using component-wise optimization, get component masks
        if component_sizes is not None:
            start_idx = 0
            component_masks = []
            for size in component_sizes:
                mask = self.xp.zeros(self.signific_size, dtype=bool)
                mask[start_idx:start_idx + size] = True
                component_masks.append(mask)
                start_idx += size
        
        print(f"Optimizing reconstruction:")
        for i in range(max_iter):
            # Compute current learning rate
            progress = (i - min_steps) / (max_iter - min_steps) if i >= min_steps else 0
            lr = init_lr * (1 - 0.7 * progress)
            
            # Forward pass to get current hidden representations
            current_acts = self._get_hidden_activations(x, is_signific=False)
            current_vector = self.xp.concatenate([act.flatten() for act in current_acts])
            
            # Compute loss - either overall or component-wise
            if component_sizes is None:
                # Overall cosine similarity
                loss = 1 - float(cosine_similarity([current_vector], [target_vector], xp=self.xp))
            else:
                # Component-wise loss
                loss = 0
                for mask in component_masks:
                    # Get masked vectors for this component
                    current_comp = current_vector[mask]
                    target_comp = target_vector[mask]
                    loss += 1 - float(cosine_similarity([current_comp], [target_comp], xp=self.xp))
                loss /= len(component_masks)  # Average across components
            
            # Update best solution
            if loss < best_loss - tol:
                best_loss = loss
                best_x = x.copy()
                no_improvement = 0
            else:
                no_improvement += 1
            
            # Early stopping check
            if i >= min_steps and no_improvement >= patience:
                print(f"Early stopping at iteration {i} - Loss: {loss:.4f}")
                break
            
            # Compute gradient using finite differences
            eps = max(1e-4, min(1e-2, loss))  # Adaptive epsilon based on loss
            perturbations = eps * self.xp.random.randn(self.layers[0].input_size)
            x_perturbed = x + perturbations
            
            # Forward pass on perturbation
            perturbed_acts = self._get_hidden_activations(x_perturbed, is_signific=False)
            perturbed_vector = self.xp.concatenate([act.flatten() for act in perturbed_acts])
            
            # Compute perturbed loss
            if component_sizes is None:
                perturbed_loss = 1 - float(cosine_similarity([perturbed_vector], [target_vector], xp=self.xp))
            else:
                perturbed_loss = 0
                for mask in component_masks:
                    perturbed_comp = perturbed_vector[mask]
                    target_comp = target_vector[mask]
                    perturbed_loss += 1 - float(cosine_similarity([perturbed_comp], [target_comp], xp=self.xp))
                perturbed_loss /= len(component_masks)
            
            # Compute gradient
            grad = (perturbed_loss - loss) / eps * perturbations
            
            # Update with momentum and gradient clipping
            momentum = beta * momentum + (1 - beta) * grad
            update = lr * momentum
            
            if i % 20 == 0:
                grad_norm = float(self.xp.sqrt(self.xp.sum(grad * grad)))
                print(f"Iteration {i}: Loss = {loss:.4f}, Gradient norm = {grad_norm:.4f}")
            
            x = self.xp.clip(x - update, 0, 1)
        
        return best_x

    def str_rep(self, signific_input, class_avg_input):
        """Compute cosine similarity between signific and referential pathways."""
        # Get activations from both pathways for connected layers only
        signific_activations = self._get_hidden_activations(signific_input, is_signific=True)
        referential_activations = self._get_hidden_activations(class_avg_input, is_signific=False)
        
        # Concatenate activations into single vectors
        signific_vector = self.xp.concatenate([act.flatten() for act in signific_activations])
        referential_vector = self.xp.concatenate([act.flatten() for act in referential_activations])
        
        # Use utility function for cosine similarity
        return float(cosine_similarity([signific_vector], [referential_vector], xp=self.xp)) 
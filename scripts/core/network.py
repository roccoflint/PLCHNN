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
                    
                    # Split signific pattern into components (digit and color if present)
                    n_digits = 10  # First 10 positions are digits
                    digit_pattern = signific_pattern[:, :n_digits]
                    has_colors = signific_pattern.shape[1] > n_digits
                    
                    # Get target indices for each component
                    digit_targets = self.xp.argmax(digit_pattern, axis=1)
                    batch_indices = self.xp.arange(x.shape[0])
                    
                    if has_colors:
                        # Handle colored MNIST case
                        color_pattern = signific_pattern[:, n_digits:]
                        color_targets = self.xp.argmax(color_pattern, axis=1) + n_digits
                        # Split +1.0 between digit and color
                        yl[digit_targets, batch_indices] = 0.5
                        yl[color_targets, batch_indices] = 0.5
                        # Create mask for all positions except the target digit-color pair
                        mask = self.xp.ones_like(tot_input, dtype=bool)
                        mask[digit_targets, batch_indices] = False
                        mask[color_targets, batch_indices] = False
                    else:
                        # Handle regular MNIST case
                        yl[digit_targets, batch_indices] = 1.0  # Full +1.0 for digit
                        # Create mask for all positions except the target digit
                        mask = self.xp.ones_like(tot_input, dtype=bool)
                        mask[digit_targets, batch_indices] = False
                    
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
        
        # Split activations into digit and color components if present
        n_digits = 10  # First 10 positions are digits
        has_colors = signific_activations.shape[0] > n_digits
        
        # Apply nonlinearity with same power
        power = self.p * self.signific_p_multiplier - 1
        
        # Process all activations at once
        sig = self.xp.sign(signific_activations)
        act = sig * self.xp.abs(signific_activations)**power
        
        # Normalize each component separately
        digit_act = act[:n_digits]
        digit_act = digit_act / (self.xp.sum(self.xp.abs(digit_act), axis=0, keepdims=True) + 1e-8)
        digit_classes = self.xp.argmax(digit_act, axis=0)
        
        if has_colors:
            color_act = act[n_digits:]
            color_act = color_act / (self.xp.sum(self.xp.abs(color_act), axis=0, keepdims=True) + 1e-8)
            color_classes = self.xp.argmax(color_act, axis=0)
            return [digit_classes, color_classes]
        else:
            return [digit_classes]
    
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
                    # Check if we have color components
                    n_digits = 10
                    has_colors = x.shape[1] > n_digits
                    
                    if has_colors:
                        # Handle colored MNIST case
                        digit_input = x[:, :n_digits]
                        color_input = x[:, n_digits:]
                        
                        # Split weights for digits and colors
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
                    else:
                        # Handle regular MNIST case - project digits only
                        power = self.p * self.signific_p_multiplier - 1
                        sig = self.xp.sign(W)
                        act = self.xp.dot(x, sig * self.xp.abs(W)**power)
                    
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

    def str_rec(self, signific_input):
        """Reconstruct input from signific input using direct weight reversal for single layer networks."""
        # Only support single layer networks
        if len(self.layers) != 1 or len(self.signific_weights) != 1:
            raise ValueError("str_rec currently only supports single layer networks")

        # Project signific input to hidden layer using signific weights with increased power
        W_sig = self.signific_weights[0]  # Shape: (signific_size, hidden_size)
        
        # Check if we have color components
        n_digits = 10
        has_colors = signific_input.shape[1] > n_digits
        
        if has_colors:
            # Handle colored MNIST case
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
            
            # Combine hidden activations
            hidden = hidden_digits + hidden_colors
            
            # Project hidden activations back to input using referential weights
            W_ref = self.layers[0].W  # Shape: (hidden_size, input_size)
            
            # Split referential weights into intensity and color channels
            W_intensity = W_ref[:, :784]  # Weights for intensity channel
            W_colors = [W_ref[:, 784*(i+1):784*(i+2)] for i in range(n_colors)]  # Weights for each color channel
            
            # First reconstruct the intensity pattern
            sig_intensity = self.xp.sign(W_intensity)
            W_intensity_nonlinear = sig_intensity * self.xp.abs(W_intensity)**(self.p - 1)
            x_intensity = self.xp.dot(hidden, W_intensity_nonlinear)
            
            # Initialize output with zeros
            x = self.xp.zeros((signific_input.shape[0], 784 * (1 + n_colors)), dtype=x_intensity.dtype)
            
            # Normalize intensity pattern to [0, 1]
            x_intensity = x_intensity - self.xp.min(x_intensity, axis=1, keepdims=True)
            x_intensity = x_intensity / (self.xp.max(x_intensity, axis=1, keepdims=True) + 1e-8)
            x[:, :784] = x_intensity
            
            # Get color weights from signific input
            color_weights = self.xp.abs(color_input)  # Shape: (batch, n_colors)
            
            # For each color channel, reconstruct using color-specific weights
            for c in range(n_colors):
                if color_weights[0, c] > 0:  # If this color is active
                    # Get color-specific weights
                    sig_color = self.xp.sign(W_colors[c])
                    W_color_nonlinear = sig_color * self.xp.abs(W_colors[c])**(self.p - 1)
                    
                    # Project to color channel
                    x_color = self.xp.dot(hidden, W_color_nonlinear)
                    
                    # Normalize color pattern to [0, 1]
                    x_color = x_color - self.xp.min(x_color, axis=1, keepdims=True)
                    x_color = x_color / (self.xp.max(x_color, axis=1, keepdims=True) + 1e-8)
                    
                    # Scale by color weight and copy to appropriate channel
                    start_idx = 784 * (c + 1)
                    end_idx = start_idx + 784
                    x[:, start_idx:end_idx] = x_color * float(color_weights[0, c])
        else:
            # Handle regular MNIST case
            power = self.p * self.signific_p_multiplier - 1
            
            # Project digits to get hidden activations
            sig = self.xp.sign(W_sig)
            hidden = self.xp.dot(signific_input, sig * self.xp.abs(W_sig)**power)
            
            # Project hidden activations back to input using referential weights
            W_ref = self.layers[0].W  # Shape: (hidden_size, input_size)
            sig = self.xp.sign(W_ref)
            W_nonlinear = sig * self.xp.abs(W_ref)**(self.p - 1)
            x = self.xp.dot(hidden, W_nonlinear)
            
            # Normalize to [0, 1] per sample
            x = x - self.xp.min(x, axis=1, keepdims=True)
            x = x / (self.xp.max(x, axis=1, keepdims=True) + 1e-8)
        
        return x

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
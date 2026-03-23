import torch
import torch.nn as nn
import torch.nn.functional as F

from .CSBrain import CSBrain
from EEG_Encoder.Model.CommonBlock import Config


class CSBrain_Encoder(nn.Module):
    """
    CSBrain Encoder wrapper for WaveMind framework
    
    Handles input format conversion and integrates with CLIP alignment training
    Input: (batch, 32, 200) - 32 channels, 200 time points (1 second @ 200Hz)
    Output: CLIP-aligned embeddings compatible with WaveMind framework
    """
    
    def __init__(self, config=Config()):
        super().__init__()
        
        # Configuration for standard 32-channel EEG
        self.num_channels = config.num_channels
        self.seq_len = config.seq_len
        self.proj_dim = config.proj_dim
        
        # CSBrain parameters - adapted for 200Hz 1-second input
        self.patch_size = 20  # Time points per patch (20 time points)
        self.num_patches = 10  # Number of patches (10 patches × 20 time points = 200 total)
        self.in_dim = self.patch_size  # Input dimension (time points per patch)
        self.out_dim = 200  # Output dimension  
        self.d_model = 200  # Model hidden dimension
        self.dim_feedforward = 800  # Feedforward dimension
        self.n_layer = 6  # Number of transformer layers (reduced for efficiency)
        self.nhead = 8  # Number of attention heads
        
        # Generate standard 32-channel brain region configuration
        brain_regions = self._generate_standard_32ch_regions()
        sorted_indices = list(range(32))
        
        # Initialize CSBrain model - use correct seq_len for patches
        self.csbrain = CSBrain(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            d_model=self.d_model,
            dim_feedforward=self.dim_feedforward,
            seq_len=self.num_patches,  # Number of patches (10 patches)
            n_layer=self.n_layer,
            nhead=self.nhead,
            brain_regions=brain_regions,
            sorted_indices=sorted_indices
        )
        
        # Output projection to CLIP space
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model, self.proj_dim),
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _generate_standard_32ch_regions(self):
        """
        Generate standard 32-channel EEG brain regions based on 10-20 system
        """
        brain_regions = []
        
        # Define 32 channels according to international 10-20 system
        channels = [
            'Fp1', 'Fp2',  # Frontal polar
            'F3', 'Fz', 'F4',  # Frontal
            'Fc3', 'Fcz', 'Fc4',  # Fronto-central
            'C3', 'Cz', 'C4',  # Central
            'Cp3', 'Cpz', 'Cp4',  # Centro-parietal
            'P3', 'Pz', 'P4',  # Parietal
            'Po3', 'Poz', 'Po4',  # Parieto-occipital
            'O1', 'Oz', 'O2',  # Occipital
            'F7', 'F8',  # Lateral frontal
            'T7', 'T8',  # Temporal
            'P7', 'P8',  # Lateral parietal
            'Oz', 'POz'   # Additional occipital/parietal
        ]
        
        # Assign regions based on channel names
        for i, channel in enumerate(channels):
            if i < 32:  # Ensure we only use first 32 channels
                if channel.startswith('Fp'):
                    brain_regions.append('frontal_polar')
                elif channel.startswith('F') and channel != 'Fz':
                    brain_regions.append('frontal')
                elif channel.startswith('Fc'):
                    brain_regions.append('fronto_central')
                elif channel.startswith('C') and channel != 'Cz':
                    brain_regions.append('central')
                elif channel.startswith('Cp'):
                    brain_regions.append('centro_parietal')
                elif channel.startswith('P') and channel != 'Pz':
                    brain_regions.append('parietal')
                elif channel.startswith('Po') or channel == 'Oz':
                    brain_regions.append('parieto_occipital')
                elif channel.startswith('O'):
                    brain_regions.append('occipital')
                elif channel.startswith('T'):
                    brain_regions.append('temporal')
                else:
                    brain_regions.append('other')
        
        # Ensure we have exactly 32 regions
        while len(brain_regions) < 32:
            brain_regions.append('other')
            
        return brain_regions[:32]
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input EEG data (batch_size, 32, 200)
            
        Returns:
            dict with 'pooler_output' and 'last_hidden_state'
        """
        batch_size, channels, time_points = x.shape
        
        # Validate input shape
        assert channels == 32, f"Expected 32 channels, got {channels}"
        assert time_points == 200, f"Expected 200 time points, got {time_points}"
        
        # Convert input format: (batch, 32, 200) -> (batch, 32, 20, 10)
        # CSBrain expects (batch, channels, patch_size, num_patches)
        # Split 200 time points into 10 patches of 20 time points each
        x_patches = x.view(batch_size, channels, self.num_patches, self.patch_size).permute(0, 1, 3, 2)
        # Step 1: reshape to (B, 32, 10, 20) - 10 patches, 20 time points each
        # Step 2: permute to (B, 32, 20, 10) - patch_size=20, num_patches=10 (CSBrain format)
        
        # Process through CSBrain
        csbrain_output = self.csbrain(x_patches)
        
        # csbrain_output shape depends on CSBrain's internal architecture
        # Let's take mean across spatial dimensions to get a feature vector
        if len(csbrain_output.shape) == 4:  # (batch, channels, patch_size, features)
            # Average over channels and patch_size
            pooled_output = csbrain_output.mean(dim=(1, 2))
        else:
            # For other shapes, adapt accordingly
            # Take mean across all spatial dimensions
            spatial_dims = list(range(1, len(csbrain_output.shape) - 1))
            pooled_output = csbrain_output.mean(dim=spatial_dims)
        
        # Project to CLIP space
        pooled_output = self.output_projection(pooled_output)
        
        # Normalize for CLIP alignment
        pooled_output = F.normalize(pooled_output, dim=-1)
        
        return {
            'pooler_output': pooled_output,
            'last_hidden_state': csbrain_output
        }
    
    def get_config(self):
        """Return model configuration"""
        return {
            'model_name': 'CSBrain',
            'input_shape': (32, 200),
            'output_dim': self.proj_dim,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'brain_regions': self._generate_standard_32ch_regions(),
            'sampling_rate': 200,
            'segment_length': 1.0  # seconds
        }


def create_csbrain_encoder(
    num_channels=32,
    d_model=512,
    num_heads=8,
    num_layers=6,
    patch_size=20,
    proj_dim=768,
    **kwargs
):
    """
    Factory function to create CSBrain encoder
    
    Args:
        num_channels: Number of EEG channels (default: 32)
        d_model: Model hidden dimension (default: 512)
        num_heads: Number of attention heads (default: 8)
        num_layers: Number of transformer layers (default: 6)
        patch_size: Time points per patch (default: 20)
        proj_dim: Output projection dimension (default: 768)
        **kwargs: Additional configuration parameters
        
    Returns:
        CSBrain_Encoder instance
    """
    # Create configuration object
    config = Config(
        num_channels=num_channels,
        seq_len=patch_size * 10,  # Total sequence length = patch_size * num_patches
        proj_dim=proj_dim
    )
    
    return CSBrain_Encoder(config)


if __name__ == '__main__':
    # Test the encoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config(num_channels=32, seq_len=200, proj_dim=768)
    
    # Create model
    model = CSBrain_Encoder(config).to(device)
    
    # Test forward pass
    batch_size = 4
    test_input = torch.randn(batch_size, 32, 200).to(device)
    
    print(f"Input shape: {test_input.shape}")
    
    with torch.no_grad():
        output = model(test_input)
    
    print(f"Output pooler shape: {output['pooler_output'].shape}")
    print(f"Output last_hidden_state shape: {output['last_hidden_state'].shape}")
    
    # Print model configuration
    config_info = model.get_config()
    print(f"Model configuration: {config_info}")
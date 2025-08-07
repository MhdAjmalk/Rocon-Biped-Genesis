# Biped Environment - Modular Architecture

This repository contains a high-performance biped robot environment that has been refactored into a modular architecture for better organization and maintainability.

## File Structure

### Original File
- `biped_env.py` - Original monolithic implementation (now serves as legacy compatibility layer)

### New Modular Files
1. `biped_env_main.py` - **Main environment class** with core simulation logic
2. `domain_randomization.py` - **Domain randomization module** with all randomization functionality
3. `reward_functions.py` - **Reward functions module** with all reward calculations

## Usage

### Using the New Modular Version (Recommended)

```python
from biped_env_main import BipedEnv

# Create environment instance
env = BipedEnv(
    num_envs=1024,
    env_cfg=env_config,
    obs_cfg=obs_config, 
    reward_cfg=reward_config,
    command_cfg=command_config,
    show_viewer=False
)
```

### Using the Legacy Version (Compatibility)

The original `biped_env.py` file now automatically detects if the modular components are available and uses them. If not, it falls back to the legacy implementation:

```python
from biped_env import BipedEnv

# Works with both modular and legacy implementations
env = BipedEnv(num_envs=1024, ...)
```

## Modular Components

### 1. Domain Randomization (`domain_randomization.py`)

**Features:**
- Vectorized motor strength randomization
- Friction coefficient randomization  
- Mass randomization
- Motor backlash simulation
- Foot contact sensor randomization
- Observation noise injection

**Key Methods:**
- `randomize_motor_strength()` - Apply motor strength variations
- `randomize_friction()` - Randomize surface friction
- `randomize_mass()` - Add mass variations to robot links
- `apply_motor_backlash()` - Simulate motor backlash effects
- `apply_foot_contact_randomization_optimized()` - Randomize contact sensors

### 2. Reward Functions (`reward_functions.py`)

**Available Rewards:**
- `reward_lin_vel_z()` - Penalize vertical velocity
- `reward_action_rate()` - Penalize rapid action changes
- `reward_similar_to_default()` - Encourage default joint positions
- `reward_forward_velocity()` - Reward forward movement
- `reward_tracking_lin_vel_x/y()` - Track commanded velocities
- `reward_alive_bonus()` - Constant alive bonus
- `reward_fall_penalty()` - Penalize falling
- `reward_torso_stability()` - Reward stable orientation
- `reward_height_maintenance()` - Maintain target height
- `reward_joint_movement()` - Reward appropriate joint movement
- `reward_sinusoidal_gait()` - Encourage gait patterns
- `reward_actuator_constraint()` - Enforce actuator limits

### 3. Main Environment (`biped_env_main.py`)

**Core Features:**
- Genesis physics simulation setup
- Robot URDF loading and configuration
- Contact sensor management
- PD control implementation
- Observation processing with pre-allocated buffers
- Episode management and reset logic

## Performance Optimizations

The modular architecture maintains all performance optimizations from the original implementation:

### 1. Pre-allocated Tensor Buffers
- **obs_components**: Eliminates torch.cat overhead in observation creation
- **noise_buffers**: Pre-allocated noise tensors for domain randomization
- **reward_buffers**: Buffers for reward computation intermediate values
- **randomization_buffers**: Vectorized domain randomization parameters

### 2. Vectorized Operations
- Motor strength: Batch updates using `tensor.uniform_()` instead of loops
- Friction: Vectorized friction coefficient generation
- Mass: Batch mass randomization where API supports it
- Motor backlash: Vectorized backlash value generation

### 3. Optimized Observation Creation
- Single `torch.cat` call instead of multiple concatenations
- In-place operations using `out=` parameter
- Pre-allocated component buffers to avoid memory allocation

### 4. Efficient Reward Computation  
- Pre-allocated buffers for intermediate calculations
- In-place operations (`torch.sub`, `torch.square`, `torch.abs` with `out=`)
- Reduced temporary tensor creation

### 5. Optimized Noise Generation
- Batch noise generation with in-place scaling
- Single `randn` call per noise type using `out=` parameter

## Expected Performance Improvements

- **15-25% faster environment step times**
- **Reduced memory fragmentation**
- **Better GPU utilization through vectorized operations**
- **Improved code maintainability and extensibility**

## Migration Guide

### From Original Implementation

If you're currently using the original `biped_env.py`:

1. **No changes required** - The original file now automatically uses modular components when available
2. **Optional**: Switch to `biped_env_main.py` for cleaner imports and explicit modular usage

### Extending the Environment

To add new reward functions:

1. Add the reward method to `reward_functions.py`
2. Update the `reward_function_map` in both `biped_env_main.py` and `biped_env.py`

To add new domain randomization:

1. Add the randomization method to `domain_randomization.py`  
2. Call it from the appropriate place in the main environment

## Benefits of Modular Architecture

1. **Better Organization**: Related functionality is grouped together
2. **Easier Testing**: Components can be tested in isolation
3. **Improved Maintainability**: Changes to rewards/randomization don't affect core environment
4. **Enhanced Extensibility**: New features can be added without modifying core code
5. **Code Reusability**: Components can be reused across different environments

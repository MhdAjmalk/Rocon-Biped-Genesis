# Updated Biped Inference Script with Keyboard Control

## Overview
The updated `biped_inference.py` script now includes real-time keyboard control for forward velocity commands, allowing you to interactively control the biped robot during inference.

## Key Features

### ✅ **Real-Time Velocity Control**
- Control forward velocity with keyboard input
- Smooth command transitions
- Range: -0.5 to +1.0 m/s (backward to forward)
- Step size: 0.1 m/s increments

### ✅ **Interactive Commands**
- **W / ↑**: Increase forward velocity (+0.1 m/s)
- **S / ↓**: Decrease forward velocity (-0.1 m/s)
- **SPACE**: Stop robot (velocity = 0.0 m/s)
- **Q**: Quit inference

### ✅ **Real-Time Feedback**
- Live display of commanded vs actual velocity
- Step counter and status updates
- Clean console interface

### ✅ **Optimized for Inference**
- Domain randomization disabled for clean simulation
- No reward calculations for faster execution
- Single environment for focused observation

## Usage

### **Basic Command**
```bash
python biped_inference.py -e biped-walking --ckpt 100
```

### **With Custom Experiment**
```bash
python biped_inference.py -e my-experiment --ckpt 200
```

### **Command Line Arguments**
- `-e, --exp_name`: Experiment name (default: "biped-walking")
- `--ckpt`: Checkpoint number to load (default: 100)

## How It Works

### **1. Initialization**
- Loads trained model and configurations
- Disables domain randomization for clean simulation
- Sets up keyboard controller in separate thread

### **2. Real-Time Control Loop**
```python
# Keyboard input updates velocity command
forward_vel_cmd = keyboard_controller.get_velocity_command()

# Environment receives the command
env.commands[:, 0] = forward_vel_cmd  # Forward velocity
env.commands[:, 1] = 0.0             # No sideways velocity  
env.commands[:, 2] = 0.0             # No angular velocity

# Policy responds to updated observations (including commands)
actions = policy(obs)
obs, _, _, _ = env.step(actions)
```

### **3. Status Display**
```
📊 Step:   1234 | Command: +0.50 m/s | Actual: +0.47 m/s
```

## Example Session

```bash
$ python biped_inference.py -e biped-walking --ckpt 100

Loading configurations from: logs/biped-walking/cfgs.pkl
Loading model checkpoint from: logs/biped-walking/model_100.pt

============================================================
🎮 KEYBOARD CONTROLS:
============================================================
W / ↑  : Increase forward velocity (+0.1 m/s)
S / ↓  : Decrease forward velocity (-0.1 m/s)
SPACE  : Stop (set velocity to 0.0 m/s)
Q      : Quit inference
============================================================
Current velocity: 0.00 m/s
Range: [-0.5, 1.0] m/s
============================================================

🚀 Inference started with keyboard control!
💡 The robot will follow your velocity commands.
🔄 Commands are updated in real-time.

Enter command (w/s/space/q) or press Enter to continue: w
🚀 Forward velocity: 0.10 m/s

📊 Step:    120 | Command: +0.10 m/s | Actual: +0.08 m/s

Enter command (w/s/space/q) or press Enter to continue: w
🚀 Forward velocity: 0.20 m/s

Enter command (w/s/space/q) or press Enter to continue: space
⏸️  Stopped: 0.00 m/s

Enter command (w/s/space/q) or press Enter to continue: q
🛑 Quitting inference...

✅ Inference completed. Goodbye!
```

## Technical Details

### **Keyboard Controller Class**
- Runs in separate thread to avoid blocking simulation
- Thread-safe velocity command updates
- Graceful shutdown handling

### **Command Integration**
- Commands are part of the 38-dimensional observation vector
- Policy was trained to follow these velocity commands
- Real-time command updates change robot behavior instantly

### **Performance**
- 50Hz simulation step rate (0.02s timesteps)
- Minimal overhead from keyboard input
- Status updates every 2 seconds to avoid console spam

## Requirements

### **Trained Model**
- Model must be trained with command following (tracking_lin_vel_x reward)
- 38-dimensional observation space (including 3 command observations)
- Compatible with the current environment configuration

### **System**
- Terminal/console access for keyboard input
- Graphics support for Genesis viewer
- Python threading support

## Troubleshooting

### **Model Not Found**
```bash
Error: Model checkpoint not found at logs/experiment-name/model_100.pt
```
**Solution**: Check experiment name and checkpoint number

### **Configuration Mismatch**
```bash
Error: tensor dimension mismatch
```
**Solution**: Ensure model was trained with 38 observations (including commands)

### **Keyboard Input Issues**
- Use standard terminal/console
- Input commands are case-insensitive
- Press Enter after each command

## Next Steps

1. **Train Your Model**: Use `biped_train.py` with command following enabled
2. **Test Inference**: Run with a known good checkpoint
3. **Experiment**: Try different velocity commands and observe robot behavior
4. **Advanced Control**: Consider extending to 2D velocity control (x, y)

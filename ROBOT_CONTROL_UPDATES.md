# Robot Control Updates - Pause & Status Overlay

## Summary
Added comprehensive pause/resume control with visual status feedback for the Hades RL agent.

## Changes Made

### 1. Configuration (`config/game_config.json`)
- **Startup delay**: Reduced from 60s to **15 seconds** (configurable)
- **Emergency stop key**: Changed to **F12** (toggle pause/resume)
- **Status overlay**: Added `show_status_overlay: true` option

### 2. New File: `scripts/status_overlay.py`
On-screen display showing robot state in bottom-right corner:
- 🟢 **RUNNING** (green) - Agent actively playing
- ⏸️ **PAUSED** (orange) - Agent frozen, waiting for resume
- 🔴 **STOPPED** (red) - Emergency stop triggered

### 3. Updated: `scripts/main.py`
- Added global pause control with thread-safe event signaling
- F12 hotkey toggles pause/resume **instantly** (even during startup countdown)
- Pause works during the 15-second startup delay
- Countdown timer shows remaining startup time in debug mode
- Emergency stop (Ctrl+C) sets flag to halt all operations
- Status overlay lifecycle management (start/stop/update)

### 4. Updated: `config/requirements.txt`
- Added `keyboard>=0.13.5` for global hotkey listening

## How It Works

### Startup Sequence
1. Game launches
2. **15-second countdown begins** (agent waits, no actions sent)
3. During countdown: Press **F12** to pause → countdown freezes
4. Press **F12** again → countdown resumes
5. After countdown: Agent enters gameplay

### During Gameplay
- Press **F12** → Agent immediately pauses (no actions sent)
- Overlay shows ⏸️ PAUSED
- Press **F12** again → Agent resumes from exact state
- Overlay shows 🟢 RUNNING

### Emergency Stop
- Press **Ctrl+C** in terminal → All keys released, overlay shows 🔴 STOPPED
- Or close the overlay window

## Key Features

✅ **Pause works during startup** - Even in the 15-second load delay  
✅ **Instant response** - No lag between keypress and pause  
✅ **Visual confirmation** - Always know the robot's state  
✅ **Thread-safe** - Pause listener runs in background thread  
✅ **Resume from exact state** - No lost progress when unpausing  

## Usage

```bash
# Normal run with overlay and pause control
python scripts/main.py --mode demo --episodes 3 --verbose

# Custom pause key (if F12 conflicts)
python scripts/main.py --mode demo --pause-key p
```

## Important Notes

⚠️ **Windows Defender**: The `keyboard` module requires administrator privileges on Windows. Run the terminal as Administrator if the pause hotkey doesn't register.

⚠️ **Overlay requires tkinter**: Standard Python GUI library. If missing, install with:
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# Windows: Included with Python installer
```

⚠️ **Does NOT modify Hades**: The robot only sends keyboard inputs and reads screen pixels - it never touches game files. Windows Defender may flag it due to automation behavior, not file modification.

## Testing Checklist

- [ ] Overlay appears in bottom-right corner on startup
- [ ] Shows 🟢 RUNNING initially
- [ ] Press F12 → changes to ⏸️ PAUSED
- [ ] Press F12 → returns to 🟢 RUNNING
- [ ] Pause during 15-second startup works
- [ ] Ctrl+C → overlay shows 🔴 STOPPED briefly before closing
- [ ] No actions sent while paused (watch action counter)

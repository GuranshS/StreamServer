#!/usr/bin/env python3
"""
Module for managing virtual displays using xrandr.
"""

import subprocess
import sys
from typing import Optional, Dict, Any


def run_command(cmd: str, check: bool = True) -> Optional[str]:
    """Execute a shell command and return its output."""
    try:
        result = subprocess.run(cmd, shell=True, check=check,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {cmd}")
        print(e.stderr)
        return None


def get_current_screens() -> Optional[Dict[str, Any]]:
    """Get information about currently connected screens."""
    output = run_command("xrandr")
    if not output:
        return None
    
    screens = {}
    current_screen = None
    
    for line in output.splitlines():
        # Check for connected/disconnected screens
        if "connected" in line:
            parts = line.split()
            screen_name = parts[0]
            status = parts[1]
            screens[screen_name] = {
                'status': status,
                'modes': [],
                'current_mode': None,
                'position': None
            }
            current_screen = screen_name
            if len(parts) > 3 and parts[2] == "primary":
                screens[screen_name]['primary'] = True
                screens[screen_name]['current_mode'] = parts[3]
            elif len(parts) > 2 and "x" in parts[2]:
                screens[screen_name]['current_mode'] = parts[2]
            
            # Check for position information
            for part in parts:
                if part in ["left-of", "right-of", "above", "below"]:
                    screens[screen_name]['position'] = part
        elif current_screen and "x" in line:
            # Add available modes
            mode = line.strip().split()[0]
            screens[current_screen]['modes'].append(mode)
    
    return screens


def check_mode_exists(mode_name: str) -> bool:
    """Check if a display mode already exists."""
    output = run_command("xrandr")
    return mode_name in output


def create_mode(width: int, height: int, refresh_rate: float) -> Optional[str]:
    """Create a new display mode using cvt and xrandr."""
    mode_name = f"{width}x{height}_{refresh_rate:.2f}"
    
    if check_mode_exists(mode_name):
        print(f"Mode {mode_name} already exists.")
        return mode_name
    
    # Generate modeline with cvt
    cvt_output = run_command(f"cvt {width} {height} {refresh_rate}")
    if not cvt_output:
        print("Failed to generate modeline with cvt")
        return None
    
    # Extract modeline from cvt output
    modeline = None
    for line in cvt_output.splitlines():
        if line.startswith('Modeline'):
            modeline = line.split('Modeline')[1].strip().split('"')[2].strip()
            break
    
    if not modeline:
        print("Could not extract modeline from cvt output")
        return None
    
    # Create new mode with xrandr
    newmode_cmd = f'xrandr --newmode "{mode_name}" {modeline}'
    if run_command(newmode_cmd) is None:
        return None
    
    print(f"Created new mode: {mode_name}")
    return mode_name


def setup_extended_screen(
        mode_name: str, 
        output_name: str = "HDMI-1", 
        position: str = "left-of", 
        relative_to: str = "eDP-1") -> bool:
    """Set up the extended screen with the specified mode."""
    # Add mode to output
    addmode_cmd = f'xrandr --addmode {output_name} {mode_name}'
    if run_command(addmode_cmd) is None:
        return False
    
    # Position the screen
    position_cmd = f'xrandr --output {output_name} --mode {mode_name} --{position} {relative_to}'
    if run_command(position_cmd) is None:
        return False
    
    print(f"Successfully set up {output_name} with mode {mode_name} {position} {relative_to}")
    return True


def cleanup(output_name: str = "HDMI-1", mode_name: Optional[str] = None) -> None:
    """Clean up by turning off the output and deleting the mode."""
    if mode_name:
        # Remove the mode from the output first
        run_command(f'xrandr --delmode {output_name} {mode_name}', check=False)
        # Delete the mode
        run_command(f'xrandr --rmmode {mode_name}', check=False)
    
    # Turn off the output
    run_command(f'xrandr --output {output_name} --off', check=False)
    print("Display cleanup complete")


if __name__ == "__main__":
    # This functionality is now delegated to screen_manager.py
    print("This module should be imported by screen_manager.py.")
    print("For standalone usage, please use screen_manager.py with appropriate arguments.")
    sys.exit(0)
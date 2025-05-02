#!/usr/bin/env python3
import subprocess
import argparse
import sys
import re

def run_command(cmd, check=True):
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

def get_current_screens():
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

def check_mode_exists(mode_name):
    """Check if a display mode already exists."""
    output = run_command("xrandr")
    return mode_name in output

def create_mode(width, height, refresh_rate):
    """Create a new display mode using cvt and xrandr."""
    mode_name = f"{width}x{height}_{refresh_rate:.2f}"
    
    if check_mode_exists(mode_name):
        print(f"Mode {mode_name} already exists.")
        return mode_name
    
    # Step 1: Generate modeline with cvt
    cvt_output = run_command(f"cvt {width} {height} {refresh_rate}")
    if not cvt_output:
        print("Failed to generate modeline with cvt")
        return None
    
    # Extract modeline from cvt output
    modeline = None
    for line in cvt_output.splitlines():
        if line.startswith('Modeline'):
            modeline = line.split('Modeline ')[1].strip('"')
            break
    
    if not modeline:
        print("Could not extract modeline from cvt output")
        return None
    
    # Step 2: Create new mode with xrandr
    newmode_cmd = f'xrandr --newmode "{mode_name}" {modeline}'
    if run_command(newmode_cmd) is None:
        return None
    
    print(f"Created new mode: {mode_name}")
    return mode_name

def setup_extended_screen(mode_name, output_name="HDMI-1", position="left-of", relative_to="eDP-1"):
    """Set up the extended screen with the specified mode."""
    # Step 4: Add mode to output
    addmode_cmd = f'xrandr --addmode {output_name} {mode_name}'
    if run_command(addmode_cmd) is None:
        return False
    
    # Step 5: Position the screen
    position_cmd = f'xrandr --output {output_name} --mode {mode_name} --{position} {relative_to}'
    if run_command(position_cmd) is None:
        return False
    
    print(f"Successfully set up {output_name} with mode {mode_name} {position} {relative_to}")
    return True

def cleanup(output_name="HDMI-1", mode_name=None):
    """Clean up by turning off the output and deleting the mode."""
    if mode_name:
        # Remove the mode from the output first
        run_command(f'xrandr --delmode {output_name} {mode_name}', check=False)
        # Delete the mode
        run_command(f'xrandr --rmmode {mode_name}', check=False)
    
    # Turn off the output
    run_command(f'xrandr --output {output_name} --off', check=False)
    print("Cleanup complete")

def main():
    parser = argparse.ArgumentParser(
        description="Manage custom display resolutions for extended screens",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("width", type=int, help="Width of the display in pixels")
    parser.add_argument("height", type=int, help="Height of the display in pixels")
    parser.add_argument("refresh_rate", type=float, help="Refresh rate in Hz")
    
    parser.add_argument("--output", default="HDMI-1",
                       help="Output name for the extended screen")
    parser.add_argument("--relative-to", default="eDP-1",
                       help="Primary screen to position relative to")
    parser.add_argument("--position", default="left-of",
                       choices=["left-of", "right-of", "above", "below"],
                       help="Position relative to primary screen")
    parser.add_argument("--no-cleanup", action="store_true",
                       help="Don't automatically clean up after setup")
    
    args = parser.parse_args()
    
    # Check current screens
    screens = get_current_screens()
    if not screens:
        print("Could not get screen information")
        return 1
    
    # Check if the specified output is already in use
    if args.output in screens and screens[args.output]['status'] == "connected":
        print(f"Output {args.output} is already connected and in use")
        return 1
    
    # Create the new mode
    mode_name = create_mode(args.width, args.height, args.refresh_rate)
    if not mode_name:
        return 1
    
    # Set up the extended screen
    if not setup_extended_screen(mode_name, args.output, args.position, args.relative_to):
        cleanup(args.output, mode_name)
        return 1
    
    if not args.no_cleanup:
        try:
            print("\nPress Enter to clean up and exit...")
            input()
        finally:
            cleanup(args.output, mode_name)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Main module for managing virtual displays and streaming screen content.

This program creates custom display resolutions for extended screens and
streams the content over UDP with low latency using FFmpeg.
"""

import argparse
import sys
import time
import signal

# Import modules
import virtual_display as vd
from screen_streamer import ScreenStreamer, create_streamer_from_config


class ScreenManagerStreamer:
    """Main class that combines display management and streaming functionality."""
    
    def __init__(self):
        """Initialize the screen manager."""
        self.streamer = None
        self.cleanup_needed = False
        self.output_name = None
        self.mode_name = None

    def setup_and_stream(self, config):
        """Main function that handles both screen setup and streaming."""
        # Store important values for cleanup
        self.output_name = config['output']
        self.mode_name = f"{config['width']}x{config['height']}_{config['refresh_rate']:.2f}"
        
        # Check current screens 
        screens = vd.get_current_screens()
        if not screens:
            print("Could not get screen information")
            return False

        # Check if the specified output is already in use
        if config['output'] in screens and screens[config['output']]['status'] == "connected":
            print(f"Output {config['output']} is already connected and in use")
            return False

        # Create the new mode if needed
        if not vd.create_mode(config['width'], config['height'], config['refresh_rate']):
            return False

        # Setup the extended screen
        if not vd.setup_extended_screen(
                self.mode_name, config['output'], config['position'], config['relative_to']):
            self.cleanup()
            return False
        
        self.cleanup_needed = True

        # Give the display a moment to stabilize
        time.sleep(2)

        # Now setup the streamer
        self.streamer = create_streamer_from_config(config)

        if not self.streamer.start_streaming():
            self.cleanup()
            return False

        return True

    def setup_only(self, config):
        """Set up the display without streaming."""
        # Store important values for cleanup
        self.output_name = config['output']
        self.mode_name = f"{config['width']}x{config['height']}_{config['refresh_rate']:.2f}"
        
        # Check current screens
        screens = vd.get_current_screens()
        if not screens:
            print("Could not get screen information")
            return False

        # Check if the specified output is already in use
        if config['output'] in screens and screens[config['output']]['status'] == "connected":
            print(f"Output {config['output']} is already connected and in use")
            return False

        # Create the new mode if needed
        if not vd.create_mode(config['width'], config['height'], config['refresh_rate']):
            return False

        # Setup the extended screen
        if not vd.setup_extended_screen(
                self.mode_name, config['output'], config['position'], config['relative_to']):
            self.cleanup()
            return False
        
        self.cleanup_needed = True
        print(f"Successfully set up display {config['output']} with mode {self.mode_name}")
        return True

    def stream_only(self, config):
        """Stream without setting up a new display mode."""
        self.streamer = create_streamer_from_config(config)
        return self.streamer.start_streaming()

    def cleanup(self):
        """Clean up both streaming and screen setup."""
        if self.streamer:
            self.streamer.stop_streaming()
        
        if self.cleanup_needed and self.output_name and self.mode_name:
            vd.cleanup(self.output_name, self.mode_name)
        
        self.cleanup_needed = False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Manage custom display resolutions and stream the extended screen",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Screen resolution parameters
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
    
    # Streaming parameters
    parser.add_argument("--framerate", type=int, default=30,
                       help="Target framerate for streaming")
    parser.add_argument("--udp-target", default="udp://10.1.10.251:5001",
                       help="UDP target address")
    parser.add_argument("--bitrate", default="1M",
                       help="Target bitrate (e.g., 1M, 500K)")
    parser.add_argument("--preset", default="ultrafast",
                       choices=["ultrafast", "superfast", "veryfast", "faster", "fast"],
                       help="Encoding preset")
    parser.add_argument("--tune", default="zerolatency",
                       choices=["zerolatency", "fastdecode"],
                       help="Encoding tuning")
    parser.add_argument("--qp", type=int, default=30,
                       help="Quantization parameter (0-51, lower is better quality)")
    
    # Operation modes
    parser.add_argument("--setup-only", action="store_true",
                       help="Only set up the display without streaming")
    parser.add_argument("--stream-only", action="store_true",
                       help="Only stream without setting up a new display mode")
    parser.add_argument("--no-cleanup", action="store_true",
                       help="Don't automatically clean up after setup")
    
    return parser.parse_args()


def main():
    """Main entry point for the program."""
    args = parse_arguments()
    
    # Convert args to a config dictionary
    config = vars(args)
    
    manager = ScreenManagerStreamer()

    # Handle Ctrl-C gracefully
    def signal_handler(sig, frame):
        print("\nStopping stream and cleaning up...")
        if not args.no_cleanup:
            manager.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        if args.setup_only:
            # Only set up the display
            if not manager.setup_only(config):
                sys.exit(1)
            print("Display setup complete. Press Ctrl+C to clean up...")
                
        elif args.stream_only:
            # Only stream without setting up a display
            if not manager.stream_only(config):
                sys.exit(1)
            print("Streaming started. Press Ctrl+C to stop...")
                
        else:
            # Default: setup and stream
            if not manager.setup_and_stream(config):
                sys.exit(1)
            print("Streaming started. Press Ctrl+C to stop...")
        
        # Main loop
        while True:
            time.sleep(1)
            if manager.streamer and manager.streamer.process and manager.streamer.process.poll() is not None:
                print("Streaming process ended unexpectedly")
                break
                    
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    finally:
        if not args.no_cleanup:
            manager.cleanup()


if __name__ == "__main__":
    main()
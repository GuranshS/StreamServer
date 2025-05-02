#!/usr/bin/env python3
import argparse
import sys
import time
import signal
from virtual_display import setup_extended_screen, cleanup, get_current_screens
from screen_streamer import ScreenStreamer

class ScreenManagerStreamer:
    def __init__(self):
        self.streamer = None
        self.cleanup_needed = False

    def setup_and_stream(self, args):
        """Main function that handles both screen setup and streaming"""
        # First setup the extended screen
        screens = get_current_screens()
        if not screens:
            print("Could not get screen information")
            return False

        # Check if the specified output is already in use
        if args.output in screens and screens[args.output]['status'] == "connected":
            print(f"Output {args.output} is already connected and in use")
            return False

        # Create the new mode and setup the screen
        mode_name = f"{args.width}x{args.height}_{args.refresh_rate:.2f}"
        if not setup_extended_screen(mode_name, args.output, args.position, args.relative_to):
            return False
        self.cleanup_needed = True

        # Give the display a moment to stabilize
        time.sleep(2)

        # Now setup the streamer
        self.streamer = ScreenStreamer(
            width=args.width,
            height=args.height,
            framerate=args.framerate,
            output=args.output,
            udp_target=args.udp_target,
            bitrate=args.bitrate,
            preset=args.preset,
            tune=args.tune,
            qp=args.qp
        )

        if not self.streamer.start_streaming():
            self.cleanup(args.output, mode_name)
            return False

        return True

    def cleanup(self, output_name="HDMI-1", mode_name=None):
        """Clean up both streaming and screen setup"""
        if self.streamer:
            self.streamer.stop_streaming()
        
        if self.cleanup_needed:
            cleanup(output_name, mode_name)
        
        self.cleanup_needed = False

def main():
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
    parser.add_argument("--no-cleanup", action="store_true",
                       help="Don't automatically clean up after setup")
    
    args = parser.parse_args()

    manager = ScreenManagerStreamer()

    # Handle Ctrl-C gracefully
    def signal_handler(sig, frame):
        print("\nStopping stream and cleaning up...")
        manager.cleanup(args.output, f"{args.width}x{args.height}_{args.refresh_rate:.2f}")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    if not manager.setup_and_stream(args):
        sys.exit(1)

    print("Streaming started. Press Ctrl+C to stop...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        if not args.no_cleanup:
            manager.cleanup(args.output, f"{args.width}x{args.height}_{args.refresh_rate:.2f}")

if __name__ == "__main__":
    main()
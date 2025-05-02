#!/usr/bin/env python3
"""
Module for streaming screen content over UDP using FFmpeg.
"""

import subprocess
import sys
import time
from typing import Optional, Tuple, Dict, Any


class ScreenStreamer:
    """Handles streaming screen content over UDP using FFmpeg."""
    
    def __init__(self, 
                 width: int, 
                 height: int, 
                 framerate: int = 30,
                 output: str = "HDMI-1", 
                 offset_x: int = 0, 
                 offset_y: int = 0,
                 udp_target: str = "udp://10.1.10.251:5001",
                 bitrate: str = "1M", 
                 preset: str = "ultrafast",
                 tune: str = "zerolatency", 
                 qp: int = 30):
        """
        Initialize the screen streamer with capture and encoding parameters.
        
        Args:
            width: Width of the capture area
            height: Height of the capture area
            framerate: Target framerate
            output: X11 display output name (e.g., HDMI-1)
            offset_x: X offset for capture (from left of display)
            offset_y: Y offset for capture (from top of display)
            udp_target: UDP target address (e.g., "udp://10.1.10.251:5001")
            bitrate: Target bitrate (e.g., "1M")
            preset: Encoding preset (e.g., "ultrafast")
            tune: Encoding tuning (e.g., "zerolatency")
            qp: Quantization parameter (0-51, lower is better quality)
        """
        self.width = width
        self.height = height
        self.framerate = framerate
        self.output = output
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.udp_target = udp_target
        self.bitrate = bitrate
        self.preset = preset
        self.tune = tune
        self.qp = qp
        self.process = None

    def get_display_position(self) -> Optional[Tuple[int, int]]:
        """Get the position of the specified output relative to the main display."""
        try:
            output = subprocess.check_output(["xrandr"], text=True)
            for line in output.splitlines():
                if self.output in line and "connected" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part in ["left-of", "right-of", "above", "below"]:
                            # Get the coordinates from the next parts
                            coords = parts[i+1].split('+')
                            if len(coords) >= 3:
                                return int(coords[1]), int(coords[2])
            return None
        except subprocess.CalledProcessError:
            return None

    def build_ffmpeg_command(self) -> str:
        """Build the ffmpeg command for low-latency screen capture and streaming."""
        # Get the display position
        position = self.get_display_position()
        if position:
            self.offset_x, self.offset_y = position

        # Base capture parameters
        cmd = [
            "ffmpeg",
            "-f", "x11grab",
            "-video_size", f"{self.width}x{self.height}",
            "-framerate", str(self.framerate),
            "-i", f":0.0+{self.offset_x},{self.offset_y}",
        ]

        # Video encoding parameters for low latency
        cmd.extend([
            "-c:v", "libx264",
            "-preset", self.preset,
            "-tune", self.tune,
            "-qp", str(self.qp),
            "-b:v", self.bitrate,
            "-maxrate", self.bitrate,
            "-bufsize", f"{int(float(self.bitrate[:-1]) * 0.5)}{self.bitrate[-1]}",  # Half of bitrate
            "-g", str(self.framerate),  # GOP size = framerate (1 second)
            "-profile:v", "baseline",
            "-pix_fmt", "yuv420p",
            "-f", "mpegts",
            self.udp_target
        ])

        return " ".join(cmd)

    def start_streaming(self) -> bool:
        """Start the streaming process."""
        cmd = self.build_ffmpeg_command()
        print(f"Starting stream with command:\n{cmd}")
        
        try:
            self.process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            # Give ffmpeg a moment to start
            time.sleep(1)
            if self.process.poll() is not None:
                # Process ended immediately, something went wrong
                print("Failed to start ffmpeg:")
                print(self.process.stderr.read())
                return False
            return True
        except Exception as e:
            print(f"Error starting ffmpeg: {e}")
            return False

    def stop_streaming(self) -> None:
        """Stop the streaming process."""
        if self.process and self.process.poll() is None:
            # Send SIGTERM to ffmpeg for graceful shutdown
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            print("Streaming stopped")


# Import function for the module
def create_streamer_from_config(config: Dict[str, Any]) -> ScreenStreamer:
    """Create a ScreenStreamer instance from a configuration dictionary."""
    return ScreenStreamer(
        width=config['width'],
        height=config['height'],
        framerate=config['framerate'],
        output=config['output'],
        udp_target=config['udp_target'],
        bitrate=config['bitrate'],
        preset=config['preset'],
        tune=config['tune'],
        qp=config['qp']
    )


if __name__ == "__main__":
    # This functionality is now delegated to screen_manager.py
    print("This module should be imported by screen_manager.py.")
    print("For standalone usage, please use screen_manager.py with appropriate arguments.")
    sys.exit(0)
#!/usr/bin/env python3
"""
Screen Manager - A tool for creating and streaming virtual displays.

This program sets up custom display resolutions and can stream display content
over UDP with low latency using FFmpeg.
"""

import argparse
import signal
import subprocess
import sys
import time
from typing import Dict, Any, Optional, Tuple
import socket
import json
import threading

class ControlServer:
    def __init__(self, port=5000):
        self.port = port
        self.server_socket = None
        self.running = False
        self.lock = threading.Lock()
        self.current_config = None
        
    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', self.port))
        self.server_socket.listen(1)
        self.running = True
        
        threading.Thread(target=self._accept_connections, daemon=True).start()
    
    def _accept_connections(self):
        while self.running:
            try:
                client_socket, addr = self.server_socket.accept()
                threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, addr)
                ).start()
            except Exception as e:
                if self.running:
                    print(f"Accept error: {e}")
    
    def _handle_client(self, client_socket, addr):
        try:
            print(f"Client connected: {addr}")
            data = client_socket.recv(1024).decode()
            if not data:
                return
                
            message = json.loads(data)
            print(f"Received: {message}")
            
            if message.get('command') == 'setup':
                with self.lock:
                    self.current_config = {
                        'width': message['width'],
                        'height': message['height'],
                        'refresh_rate': message['refresh_rate'],
                        'udp_target': f"udp://{socket.gethostbyname(socket.gethostname())}:{message['udp_port']}"
                    }
                
                response = {
                    'status': 'ready',
                    'udp_target': self.current_config['udp_target']
                }
                client_socket.send(json.dumps(response).encode())
                
        except Exception as e:
            print(f"Client handling error: {e}")
        finally:
            client_socket.close()
    
    def wait_for_client_config(self):
        """Wait for and return client configuration."""
        while self.running:
            with self.lock:
                if self.current_config:
                    config = self.current_config.copy()
                    self.current_config = None
                    return config
            time.sleep(0.1)
        return None
    
    def stop(self):
        self.running = False
        if self.server_socket:
            self.server_socket.close()
            
class DisplayManager:
    """Manages virtual displays using xrandr."""
    
    @staticmethod
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
    
    @staticmethod
    def get_screens() -> Dict[str, Any]:
        """Get information about currently connected screens."""
        output = DisplayManager.run_command("xrandr")
        if not output:
            return {}
        
        screens = {}
        current_screen = None
        
        for line in output.splitlines():
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
                
                for part in parts:
                    if part in ["left-of", "right-of", "above", "below"]:
                        screens[screen_name]['position'] = part
                        
            elif current_screen and "x" in line:
                mode = line.strip().split()[0]
                screens[current_screen]['modes'].append(mode)
        
        return screens
    
    @staticmethod
    def create_mode(width: int, height: int, refresh_rate: float) -> Optional[str]:
        """Create a new display mode."""
        mode_name = f"{width}x{height}_{refresh_rate:.2f}"
        
        # Check if mode already exists
        output = DisplayManager.run_command("xrandr")
        if output and mode_name in output:
            print(f"Mode {mode_name} already exists.")
            return mode_name
        
        # Generate modeline with cvt
        cvt_output = DisplayManager.run_command(f"cvt {width} {height} {refresh_rate}")
        if not cvt_output:
            print("Failed to generate modeline")
            return None
        
        # Extract modeline
        modeline = None
        for line in cvt_output.splitlines():
            if line.startswith('Modeline'):
                modeline = line.split('Modeline')[1].strip().split('"')[2].strip()
                break
        
        if not modeline:
            print("Could not extract modeline from cvt output")
            return None
        
        # Create new mode
        if DisplayManager.run_command(f'xrandr --newmode "{mode_name}" {modeline}') is None:
            return None
        
        print(f"Created new mode: {mode_name}")
        return mode_name
    
    @staticmethod
    def setup_display(mode_name: str, 
                      output_name: str, 
                      position: str, 
                      relative_to: str) -> bool:
        """Set up a display with the specified mode and position."""
        # Add mode to output
        if DisplayManager.run_command(f'xrandr --addmode {output_name} {mode_name}') is None:
            return False
        
        # Position the screen
        cmd = f'xrandr --output {output_name} --mode {mode_name} --{position} {relative_to}'
        if DisplayManager.run_command(cmd) is None:
            return False
        
        print(f"Successfully set up {output_name} with mode {mode_name} {position} {relative_to}")
        return True
    
    @staticmethod
    def cleanup_display(output_name: str, mode_name: Optional[str] = None) -> None:
        """Clean up by turning off the output and optionally deleting the mode."""
        if mode_name:
            DisplayManager.run_command(f'xrandr --delmode {output_name} {mode_name}', check=False)
            DisplayManager.run_command(f'xrandr --rmmode {mode_name}', check=False)
        
        DisplayManager.run_command(f'xrandr --output {output_name} --off', check=False)
        print("Display cleanup complete")


class ScreenStreamer:
    """Handles streaming screen content over UDP using FFmpeg."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize streamer with configuration parameters."""
        self.width = config['width']
        self.height = config['height']
        self.framerate = config['framerate']
        self.output = config['output']
        self.udp_target = config['udp_target']
        self.bitrate = config['bitrate']
        self.preset = config['preset']
        self.tune = config['tune']
        self.qp = config['qp']
        self.process = None
        # Add explicit offset parameters with defaults
        self.offset_x = config.get('offset_x', None)
        self.offset_y = config.get('offset_y', None)
    
    def get_display_position(self) -> Tuple[int, int]:
        """Get the position of the specified output."""
        try:
            # First try with xrandr --verbose to get detailed geometry info
            print(f"DEBUG: Trying to find position of display: {self.output}")
            output = subprocess.check_output(["xrandr", "--verbose"], text=True)
            lines = output.splitlines()
            
            # Extract and print info about all displays for debugging
            print("DEBUG: Connected displays:")
            for line in lines:
                if "connected" in line:
                    print(f"  {line}")
            
            # Print geometry info
            print("DEBUG: Display geometry info:")
            for line in lines:
                if "geometry" in line:
                    print(f"  {line}")
            
            # Find the specified output
            output_found = False
            for i, line in enumerate(lines):
                if self.output in line and "connected" in line:
                    output_found = True
                    print(f"DEBUG: Found output {self.output} at line {i}")
                    # Look for the geometry line which contains position information
                    for j in range(i, min(i + 20, len(lines))):
                        if "geometry" in lines[j]:
                            geo_line = lines[j].strip()
                            print(f"DEBUG: Found geometry line: {geo_line}")
                            # Parse the geometry line, format is something like:
                            # geometry: 1280x800+1920+0
                            geo_parts = geo_line.split()
                            if len(geo_parts) >= 2 and "+" in geo_parts[1]:
                                coords = geo_parts[1].split("+")
                                if len(coords) >= 3:
                                    x_pos, y_pos = int(coords[1]), int(coords[2])
                                    print(f"DEBUG: Detected position: x={x_pos}, y={y_pos}")
                                    return x_pos, y_pos
            
            # Try alternative method: parse xrandr output for position
            print("DEBUG: Trying alternative method...")
            output = subprocess.check_output(["xrandr"], text=True)
            lines = output.splitlines()
            
            screens = {}
            primary_x, primary_y = 0, 0
            primary_width, primary_height = 0, 0
            
            # Find primary screen position and dimensions
            for line in lines:
                if "connected" in line and "primary" in line:
                    parts = line.split()
                    for part in parts:
                        if "x" in part and "+" in part:
                            # Format: 1920x1080+0+0
                            dims = part.replace('+', 'x').split('x')
                            if len(dims) >= 4:
                                primary_width = int(dims[0])
                                primary_height = int(dims[1])
                                primary_x = int(dims[2])
                                primary_y = int(dims[3])
                                print(f"DEBUG: Primary screen: {primary_width}x{primary_height} at +{primary_x}+{primary_y}")
            
            # Find target screen position
            for line in lines:
                if self.output in line and "connected" in line:
                    parts = line.split()
                    for part in parts:
                        if "x" in part and "+" in part:
                            # Format: 1280x800+1920+0
                            dims = part.replace('+', 'x').split('x')
                            if len(dims) >= 4:
                                x_pos = int(dims[2])
                                y_pos = int(dims[3])
                                print(f"DEBUG: Target display: {self.output} at +{x_pos}+{y_pos}")
                                return x_pos, y_pos
            
            # Fallback: Use relative position logic
            if output_found:
                print(f"DEBUG: Using fallback position logic...")
                screens = DisplayManager.get_screens()
                
                if self.output in screens:
                    position = screens[self.output].get('position')
                    print(f"DEBUG: Display {self.output} position: {position}")
                    
                    if position == "right-of":
                        return primary_width, 0
                    elif position == "left-of":
                        return 0, 0  # Assuming primary is not at 0,0
                    elif position == "above":
                        return 0, 0
                    elif position == "below":
                        return 0, primary_height
            
            print("WARNING: Could not determine display position automatically")
            print("DEBUG: Please specify position with --offset-x and --offset-y")
            return 0, 0  # Default if position not found
            
        except subprocess.CalledProcessError as e:
            print(f"Error getting display position: {e}")
            return 0, 0
    
    def build_ffmpeg_command(self) -> str:
        """Build the ffmpeg command for screen capture and streaming."""
        # Use explicit offsets if provided, otherwise try to detect them
        if self.offset_x is not None and self.offset_y is not None:
            offset_x, offset_y = self.offset_x, self.offset_y
            print(f"Using explicitly provided offsets: x={offset_x}, y={offset_y}")
        else:
            offset_x, offset_y = self.get_display_position()
        
        cmd = [
            "ffmpeg",
            "-f", "x11grab",
            "-video_size", f"{self.width}x{self.height}",
            "-framerate", str(self.framerate),
            "-i", f":0.0+{offset_x},{offset_y}",
            "-c:v", "libx264",
            "-preset", self.preset,
            "-tune", self.tune,
            "-qp", str(self.qp),
            "-b:v", self.bitrate,
            "-maxrate", self.bitrate,
            "-bufsize", f"{int(float(self.bitrate[:-1]) * 0.5)}{self.bitrate[-1]}",
            "-g", str(self.framerate),
            "-profile:v", "baseline",
            "-pix_fmt", "yuv420p",
            "-f", "mpegts",
            self.udp_target
        ]
        
        return " ".join(cmd)
    
    def start(self) -> bool:
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
            time.sleep(1)
            if self.process.poll() is not None:
                print("Failed to start ffmpeg:")
                print(self.process.stderr.read())
                return False
            return True
        except Exception as e:
            print(f"Error starting ffmpeg: {e}")
            return False
    
    def stop(self) -> None:
        """Stop the streaming process."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            print("Streaming stopped")



class ScreenManager:
    def __init__(self):
        self.streamer = None
        self.output_name = None
        self.mode_name = None
        self.cleanup_needed = False
        self.control_server = ControlServer()
        
    def start_control_server(self):
        """Start the TCP control server."""
        self.control_server.start()
        
    def wait_for_client_config(self) -> Optional[Dict[str, Any]]:
        """Wait for client configuration over TCP."""
        return self.control_server.handle_client()
    
    def setup_display(self, config: Dict[str, Any]) -> bool:
        """Setup the display with the given configuration."""
        self.output_name = config['output']
        self.mode_name = f"{config['width']}x{config['height']}_{config['refresh_rate']:.2f}"
        
        # Check if output is already in use
        screens = DisplayManager.get_screens()
        if self.output_name in screens and screens[self.output_name]['status'] == "connected":
            print(f"Output {self.output_name} is already connected and in use")
            return False
        
        # Create mode and setup display
        if not DisplayManager.create_mode(config['width'], config['height'], config['refresh_rate']):
            return False
            
        if not DisplayManager.setup_display(
                self.mode_name, 
                self.output_name, 
                config['position'], 
                config['relative_to']):
            self.cleanup()
            return False
        
        self.cleanup_needed = True
        return True
    
    def start_streaming(self, config: Dict[str, Any]) -> bool:
        """Start streaming with the given configuration."""
        self.streamer = ScreenStreamer(config)
        return self.streamer.start()
    
    def setup_and_stream(self, config: Dict[str, Any]) -> bool:
        """Set up display and start streaming."""
        if not self.setup_display(config):
            return False
        
        # Give the display a moment to stabilize
        time.sleep(2)
        
        return self.start_streaming(config)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.streamer:
            self.streamer.stop()
        
        if self.cleanup_needed and self.output_name and self.mode_name:
            DisplayManager.cleanup_display(self.output_name, self.mode_name)
            self.cleanup_needed = False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Manage custom display resolutions and stream screen content",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Display parameters
    display_group = parser.add_argument_group('Display Options')
    display_group.add_argument("--width", type=int, default=1280, 
                              help="Width of the display in pixels")
    display_group.add_argument("--height", type=int, default=800, 
                              help="Height of the display in pixels")
    display_group.add_argument("--refresh-rate", type=float, default=60.0, 
                              help="Refresh rate in Hz")
    display_group.add_argument("--output", default="HDMI-1",
                              help="Output name for the extended screen")
    display_group.add_argument("--relative-to", default="eDP-1",
                              help="Primary screen to position relative to")
    display_group.add_argument("--position", default="left-of",
                              choices=["left-of", "right-of", "above", "below"],
                              help="Position relative to primary screen")
    
    # Streaming parameters
    stream_group = parser.add_argument_group('Streaming Options')
    stream_group.add_argument("--framerate", type=int, default=30,
                             help="Target framerate for streaming")
    stream_group.add_argument("--udp-target", default="udp://10.1.10.251:5001",
                             help="UDP target address")
    stream_group.add_argument("--bitrate", default="1M",
                             help="Target bitrate (e.g., 1M, 500K)")
    stream_group.add_argument("--preset", default="ultrafast",
                             choices=["ultrafast", "superfast", "veryfast", "faster", "fast"],
                             help="Encoding preset")
    stream_group.add_argument("--tune", default="zerolatency",
                             choices=["zerolatency", "fastdecode"],
                             help="Encoding tuning")
    stream_group.add_argument("--qp", type=int, default=30,
                             help="Quantization parameter (0-51, lower is better quality)")
    stream_group.add_argument("--offset-x", type=int, help="Manual X offset for capture")
    stream_group.add_argument("--offset-y", type=int, help="Manual Y offset for capture")
    
    # Operation modes
    mode_group = parser.add_argument_group('Mode Options')
    mode_group.add_argument("--setup-only", action="store_true",
                           help="Only set up the display without streaming")
    mode_group.add_argument("--stream-only", action="store_true",
                           help="Only stream without setting up a new display mode")
    mode_group.add_argument("--no-cleanup", action="store_true",
                           help="Don't automatically clean up after setup")
    mode_group.add_argument("--debug", action="store_true",
                           help="Show additional debug information")
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = vars(args)
    
    manager = ScreenManager()
    
    def signal_handler(sig, frame):
        print("\nStopping and cleaning up...")
        if not args.no_cleanup:
            manager.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start control server
    control_server = ControlServer()
    control_server.start()
    print("Control server started on port 5000. Waiting for client connection...")
    
    try:
        # Wait for client configuration
        client_config = None
        while not client_config:
            print("Waiting for client to connect...")
            client_config = control_server.wait_for_client_config()
            time.sleep(1)
        
        # Merge client config with command line args
        config.update(client_config)
        
        if args.setup_only:
            if not manager.setup_display(config):
                sys.exit(1)
            print("Display setup complete. Press Ctrl+C to clean up...")
            
        elif args.stream_only:
            if not manager.start_streaming(config):
                sys.exit(1)
            print("Streaming started. Press Ctrl+C to stop...")
            
        else:
            if not manager.setup_and_stream(config):
                sys.exit(1)
            print("Display setup and streaming started. Press Ctrl+C to stop...")
        
        # Main loop
        while True:
            time.sleep(1)
            if (manager.streamer and manager.streamer.process and 
                manager.streamer.process.poll() is not None):
                print("Streaming process ended unexpectedly")
                break
                
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    finally:
        if not args.no_cleanup:
            manager.cleanup()
        control_server.stop()

if __name__ == "__main__":
    main()
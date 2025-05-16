#!/usr/bin/env python3
"""
Screen Manager - A tool for creating and streaming virtual displays.

This program sets up custom display resolutions and can stream display content
over UDP with low latency using FFmpeg.
"""

import argparse
import dataclasses
import enum
import logging
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("screen_manager")


class ScreenManagerError(Exception):
    """Base exception for Screen Manager errors."""
    pass


class DisplayError(ScreenManagerError):
    """Exception for display-related errors."""
    pass


class StreamError(ScreenManagerError):
    """Exception for streaming-related errors."""
    pass


class CommandError(ScreenManagerError):
    """Exception for command execution errors."""
    pass


class Position(enum.Enum):
    """Valid positions for display placement."""
    LEFT_OF = "left-of"
    RIGHT_OF = "right-of"
    ABOVE = "above"
    BELOW = "below"


@dataclass
class DisplayConfig:
    """Configuration for display setup."""
    width: int
    height: int
    refresh_rate: float
    output: str
    position: Position
    relative_to: str


@dataclass
class StreamConfig:
    """Configuration for streaming."""
    width: int
    height: int
    framerate: int
    output: str
    udp_target: str
    bitrate: str
    preset: str
    tune: str
    qp: int
    offset_x: Optional[int] = None
    offset_y: Optional[int] = None
    dimension_wait_time: float = 5.0


@dataclass
class ScreenInfo:
    """Information about a screen."""
    name: str
    status: str
    modes: List[str]
    current_mode: Optional[str] = None
    position: Optional[Position] = None
    primary: bool = False
    geometry: Optional[Tuple[int, int, int, int]] = None  # width, height, x, y


class CommandRunner:
    """Handles execution of shell commands."""
    
    @staticmethod
    def run(cmd: str, check: bool = True) -> Tuple[int, str, str]:
        """
        Execute a shell command and return its output.
        
        Args:
            cmd: Command to execute
            check: Whether to raise exception on non-zero exit code
            
        Returns:
            Tuple of (return_code, stdout, stderr)
            
        Raises:
            CommandError: If command fails and check is True
        """
        logger.debug(f"Running command: {cmd}")
        try:
            result = subprocess.run(
                cmd, 
                shell=True,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if check and result.returncode != 0:
                raise CommandError(f"Command failed with code {result.returncode}: {result.stderr}")
            return result.returncode, result.stdout, result.stderr
        except subprocess.SubprocessError as e:
            raise CommandError(f"Failed to execute command: {cmd}") from e


class DisplayManager:
    """Manages virtual displays using xrandr."""
    
    @staticmethod
    def get_screens() -> Dict[str, ScreenInfo]:
        """
        Get information about currently connected screens.
        
        Returns:
            Dictionary of screen names to ScreenInfo objects
        """
        _, output, _ = CommandRunner.run("xrandr")
        screens = {}
        current_screen = None
        
        for line in output.splitlines():
            if "connected" in line:
                parts = line.split()
                screen_name = parts[0]
                status = parts[1]
                screen_info = ScreenInfo(
                    name=screen_name,
                    status=status,
                    modes=[],
                    primary="primary" in parts
                )
                screens[screen_name] = screen_info
                current_screen = screen_name
                
                # Parse geometry and current mode
                for part in parts:
                    if "x" in part and "+" in part:
                        # Format: 1920x1080+0+0
                        dims = part.replace('+', 'x').split('x')
                        if len(dims) >= 4:
                            width = int(dims[0])
                            height = int(dims[1])
                            x_pos = int(dims[2])
                            y_pos = int(dims[3])
                            screen_info.geometry = (width, height, x_pos, y_pos)
                            screen_info.current_mode = f"{width}x{height}"
                
                # Parse position
                for i, part in enumerate(parts):
                    if part in [pos.value for pos in Position]:
                        screen_info.position = Position(part)
                        
            elif current_screen and line.strip() and "x" in line:
                mode = line.strip().split()[0]
                screens[current_screen].modes.append(mode)
        
        return screens
    
    @staticmethod
    def create_mode(config: DisplayConfig) -> str:
        """
        Create a new display mode.
        
        Args:
            config: Display configuration
            
        Returns:
            Mode name
            
        Raises:
            DisplayError: If mode creation fails
        """
        mode_name = f"{config.width}x{config.height}_{config.refresh_rate:.2f}"
        
        # Check if mode already exists
        _, output, _ = CommandRunner.run("xrandr")
        if mode_name in output:
            logger.info(f"Mode {mode_name} already exists")
            return mode_name
        
        # Generate modeline with cvt
        try:
            _, cvt_output, _ = CommandRunner.run(
                f"cvt {config.width} {config.height} {config.refresh_rate}"
            )
            
            # Extract modeline
            modeline = None
            for line in cvt_output.splitlines():
                if line.startswith('Modeline'):
                    modeline = line.split('Modeline')[1].strip().split('"')[2].strip()
                    break
            
            if not modeline:
                raise DisplayError("Could not extract modeline from cvt output")
            
            # Create new mode
            CommandRunner.run(f'xrandr --newmode "{mode_name}" {modeline}')
            logger.info(f"Created new mode: {mode_name}")
            return mode_name
            
        except CommandError as e:
            raise DisplayError(f"Failed to create display mode: {e}") from e
    
    @staticmethod
    def setup_display(config: DisplayConfig, mode_name: str) -> bool:
        """
        Set up a display with the specified mode and position.
        
        Args:
            config: Display configuration
            mode_name: Name of the mode to use
            
        Returns:
            True if successful
            
        Raises:
            DisplayError: If display setup fails
        """
        try:
            # Add mode to output
            CommandRunner.run(f'xrandr --addmode {config.output} {mode_name}')
            
            # Position the screen
            cmd = f'xrandr --output {config.output} --mode {mode_name} --{config.position.value} {config.relative_to}'
            CommandRunner.run(cmd)
            
            logger.info(f"Successfully set up {config.output} with mode {mode_name} {config.position.value} {config.relative_to}")
            return True
            
        except CommandError as e:
            raise DisplayError(f"Failed to setup display: {e}") from e
    
    @staticmethod
    def cleanup_display(output_name: str, mode_name: Optional[str] = None) -> None:
        """
        Clean up by turning off the output and optionally deleting the mode.
        
        Args:
            output_name: Name of the output to clean up
            mode_name: Name of the mode to delete (optional)
        """
        try:
            if mode_name:
                # Best effort to remove mode from output
                CommandRunner.run(f'xrandr --delmode {output_name} {mode_name}', check=False)
                CommandRunner.run(f'xrandr --rmmode {mode_name}', check=False)
            
            # Turn off output
            CommandRunner.run(f'xrandr --output {output_name} --off', check=False)
            logger.info("Display cleanup complete")
            
        except CommandError as e:
            logger.warning(f"Error during display cleanup: {e}")


class ScreenStreamer:
    """Handles streaming screen content over UDP using FFmpeg."""
    
    def __init__(self, config: StreamConfig):
        """
        Initialize streamer with configuration parameters.
        
        Args:
            config: Streaming configuration
        """
        self.config = config
        self.process = None
    
    def get_display_position(self) -> Tuple[int, int]:
        """
        Get the position of the specified output.
        
        Returns:
            Tuple of (x, y) position
        """
        try:
            screens = DisplayManager.get_screens()
            
            if self.config.output in screens and screens[self.config.output].geometry:
                _, _, x_pos, y_pos = screens[self.config.output].geometry
                logger.debug(f"Found display position: x={x_pos}, y={y_pos}")
                return x_pos, y_pos
            
            # If geometry not found, try to infer from relative position
            primary_screen = None
            for screen in screens.values():
                if screen.primary and screen.geometry:
                    primary_screen = screen
                    break
            
            if primary_screen and self.config.output in screens:
                screen = screens[self.config.output]
                primary_width, primary_height, primary_x, primary_y = primary_screen.geometry
                
                if screen.position == Position.RIGHT_OF:
                    return primary_x + primary_width, primary_y
                elif screen.position == Position.LEFT_OF:
                    return primary_x - screen.geometry[0] if screen.geometry else 0, primary_y
                elif screen.position == Position.ABOVE:
                    return primary_x, primary_y - screen.geometry[1] if screen.geometry else 0
                elif screen.position == Position.BELOW:
                    return primary_x, primary_y + primary_height
            
            logger.warning("Could not determine display position automatically")
            return 0, 0
            
        except Exception as e:
            logger.error(f"Error getting display position: {e}")
            return 0, 0
    
    def build_ffmpeg_command(self) -> List[str]:
        """
        Build the ffmpeg command for screen capture and streaming.
        
        Returns:
            List of command parts
        """
        # Use explicit offsets if provided, otherwise try to detect them
        if self.config.offset_x is not None and self.config.offset_y is not None:
            offset_x, offset_y = self.config.offset_x, self.config.offset_y
            logger.info(f"Using explicitly provided offsets: x={offset_x}, y={offset_y}")
        else:
            offset_x, offset_y = self.get_display_position()
        
        bufsize = f"{int(float(self.config.bitrate[:-1]) * 0.5)}{self.config.bitrate[-1]}"
        
        return [
            "ffmpeg",
            "-f", "x11grab",
            "-video_size", f"{self.config.width}x{self.config.height}",
            "-framerate", str(self.config.framerate),
            "-i", f":0.0+{offset_x},{offset_y}",
            "-c:v", "libx264",
            "-preset", self.config.preset,
            "-tune", self.config.tune,
            "-qp", str(self.config.qp),
            "-b:v", self.config.bitrate,
            "-maxrate", self.config.bitrate,
            "-bufsize", bufsize,
            "-g", str(self.config.framerate),
            "-profile:v", "baseline",
            "-pix_fmt", "yuv420p",
            "-f", "mpegts",
            self.config.udp_target
        ]
   
    def wait_for_client_dimensions(self) -> Optional[Tuple[int, int, str]]:
        """
        Wait briefly for client dimensions packet and return width, height, and client IP.
        
        Returns:
            Tuple of (width, height, client_ip) if received, None otherwise
        """
        try:
            # Extract port from UDP target
            target_parts = self.config.udp_target.split(':')
            if len(target_parts) >= 3:
                port = int(target_parts[2])
                
                # Create a socket to listen for dimensions
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.bind(('0.0.0.0', port))
                sock.settimeout(self.config.dimension_wait_time)
                
                logger.info(f"Waiting for client dimensions on port {port} for {self.config.dimension_wait_time} seconds...")
                data, addr = sock.recvfrom(1024)
                message = data.decode('utf-8')
                
                # Check if this is a dimensions packet
                if message.startswith("DIMS:"):
                    parts = message.split(':')
                    client_ip = addr[0]  # Default to source IP of the packet
                    
                    if len(parts) >= 3:
                        width = int(parts[1])
                        height = int(parts[2])
                        
                        # If client included its IP, use that instead
                        if len(parts) >= 4 and parts[3]:
                            client_ip = parts[3]
                        
                        logger.info(f"Received client dimensions: {width}x{height} from {client_ip}")
                        sock.close()
                        return width, height, client_ip
                
                sock.close()
                
        except socket.timeout:
            logger.info("No client dimensions received within timeout period")
        except Exception as e:
            logger.error(f"Error receiving dimensions: {e}")
        
        logger.info("Using default dimensions and target")
        return None

    def start(self, skip_dimension_check=False) -> bool:
        """
        Start the streaming process.
        
        Args:
            skip_dimension_check: Whether to skip waiting for client dimensions
            
        Returns:
            True if successful
            
        Raises:
            StreamError: If streaming fails to start
        """
        try:
            # Listen for client dimensions first if not skipping dimension check
            if not skip_dimension_check:
                client_info = self.wait_for_client_dimensions()
                if client_info:
                    width, height, client_ip = client_info
                    self.config.width = width
                    self.config.height = height
                    
                    # Update UDP target to use client's IP while keeping port
                    udp_parts = self.config.udp_target.split(':')
                    port = udp_parts[-1] if len(udp_parts) >= 3 else "5001"
                    self.config.udp_target = f"udp://{client_ip}:{port}"
                    logger.info(f"Updated stream dimensions to client size: {width}x{height}")
                    logger.info(f"Updated UDP target to: {self.config.udp_target}")
            
            cmd = self.build_ffmpeg_command()
            cmd_str = " ".join(cmd)
            logger.info(f"Starting stream with command:\n{cmd_str}")
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            time.sleep(1)
            if self.process.poll() is not None:
                stderr_output = self.process.stderr.read()
                raise StreamError(f"Failed to start ffmpeg: {stderr_output}")
                
            return True
            
        except Exception as e:
            raise StreamError(f"Error starting stream: {e}")
    
    def stop(self) -> None:
        """Stop the streaming process."""
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                logger.info("Streaming stopped")
            except Exception as e:
                logger.error(f"Error stopping stream: {e}")


class ScreenManager:
    """Main class that manages both display setup and streaming."""
    
    def __init__(self):
        """Initialize the screen manager."""
        self.streamer = None
        self.output_name = None
        self.mode_name = None
        self.cleanup_needed = False
        self.fallback_width = 1920
        self.fallback_height = 1200
    
    def is_crtc_error(self, error_output: str) -> bool:
        """
        Check if the error output contains a CRTC error.
        
        Args:
            error_output: Error message to check
            
        Returns:
            True if contains CRTC error
        """
        crtc_errors = [
            "CRTC",
            "crtc",
            "failed to set mode",
            "cannot set mode",
            "Configure crtc"
        ]
        return any(error.lower() in error_output.lower() for error in crtc_errors)
    
    def setup_display(self, config: Union[DisplayConfig, Dict]) -> bool:
        """
        Setup the display with the given configuration, with fallback to 1920x1200 on CRTC error.
        
        Args:
            config: Display configuration
            
        Returns:
            True if successful
            
        Raises:
            DisplayError: If display setup fails
        """
        # Convert dict to DisplayConfig if needed
        if isinstance(config, dict):
            display_config = DisplayConfig(
                width=config['width'],
                height=config['height'],
                refresh_rate=config['refresh_rate'],
                output=config['output'],
                position=Position(config['position']),
                relative_to=config['relative_to']
            )
        else:
            display_config = config
        
        self.output_name = display_config.output
        original_width = display_config.width
        original_height = display_config.height
        
        def attempt_setup(width: int, height: int) -> Tuple[bool, str]:
            """Helper function to attempt display setup with given resolution."""
            try:
                # Update config with new dimensions
                display_config.width = width
                display_config.height = height
                
                # Clean up any existing mode first
                mode_name = f"{width}x{height}_{display_config.refresh_rate:.2f}"
                DisplayManager.cleanup_display(self.output_name, mode_name)
                
                # Create new mode
                mode_name = DisplayManager.create_mode(display_config)
                
                # Set up display with the mode
                DisplayManager.setup_display(display_config, mode_name)
                
                self.mode_name = mode_name
                self.cleanup_needed = True
                logger.info(f"Successfully set up display with resolution {width}x{height}")
                return True, ""
                
            except DisplayError as e:
                error_msg = str(e)
                logger.warning(f"Setup failed with resolution {width}x{height}: {error_msg}")
                DisplayManager.cleanup_display(self.output_name, mode_name)
                return False, error_msg
        
        # First try with original resolution
        success, error_msg = attempt_setup(original_width, original_height)
        if success:
            return True
        
        # If failed with CRTC error, try fallback resolution
        if self.is_crtc_error(error_msg):
            logger.info(f"CRTC error detected, trying fallback resolution {self.fallback_width}x{self.fallback_height}")
            success, _ = attempt_setup(self.fallback_width, self.fallback_height)
            if success:
                # If we succeeded with fallback, update the original config
                if isinstance(config, dict):
                    config['width'] = self.fallback_width
                    config['height'] = self.fallback_height
                return True
        
        return False
    
    def start_streaming(self, config: Union[StreamConfig, Dict]) -> bool:
        """
        Start streaming with the given configuration.
        
        Args:
            config: Streaming configuration
            
        Returns:
            True if successful
            
        Raises:
            StreamError: If streaming fails to start
        """
        # Convert dict to StreamConfig if needed
        if isinstance(config, dict):
            stream_config = StreamConfig(
                width=config['width'],
                height=config['height'],
                framerate=config['framerate'],
                output=config['output'],
                udp_target=config['udp_target'],
                bitrate=config['bitrate'],
                preset=config['preset'],
                tune=config['tune'],
                qp=config['qp'],
                offset_x=config.get('offset_x'),
                offset_y=config.get('offset_y'),
                dimension_wait_time=config.get('dimension_wait_time', 5.0)
            )
        else:
            stream_config = config
        
        self.streamer = ScreenStreamer(stream_config)
        return self.streamer.start()
    
    def setup_and_stream(self, config: Dict) -> bool:
        """
        Set up display and start streaming.
        
        Args:
            config: Combined configuration
            
        Returns:
            True if successful
        """
        try:
            # Create a temporary streamer to try and get client dimensions
            stream_config = StreamConfig(
                width=config['width'],
                height=config['height'],
                framerate=config['framerate'],
                output=config['output'],
                udp_target=config['udp_target'],
                bitrate=config['bitrate'],
                preset=config['preset'],
                tune=config['tune'],
                qp=config['qp'],
                offset_x=config.get('offset_x'),
                offset_y=config.get('offset_y'),
                dimension_wait_time=config.get('dimension_wait_time', 5.0)
            )
            
            temp_streamer = ScreenStreamer(stream_config)
            client_info = temp_streamer.wait_for_client_dimensions()
            
            if client_info:
                width, height, client_ip = client_info
                config['width'] = width
                config['height'] = height
                
                # Update UDP target to use client's IP while keeping port
                udp_parts = config['udp_target'].split(':')
                port = udp_parts[-1] if len(udp_parts) >= 3 else "5001"
                config['udp_target'] = f"udp://{client_ip}:{port}"
                logger.info(f"Updated UDP target to: {config['udp_target']}")
            
            # Set up display
            display_config = DisplayConfig(
                width=config['width'],
                height=config['height'],
                refresh_rate=config['refresh_rate'],
                output=config['output'],
                position=Position(config['position']),
                relative_to=config['relative_to']
            )
            
            if not self.setup_display(display_config):
                return False
            
            # Give the display a moment to stabilize
            time.sleep(2)
            
            # IMPORTANT: Update config with the actual dimensions after setup
            # This ensures we're using the fallback resolution if it was applied
            if self.fallback_width != config['width'] or self.fallback_height != config['height']:
                logger.info(f"Updating stream dimensions to match fallback resolution: {self.fallback_width}x{self.fallback_height}")
                config['width'] = self.fallback_width
                config['height'] = self.fallback_height
            
            # Create a new streamer with the updated config
            stream_config = StreamConfig(
                width=config['width'],
                height=config['height'],
                framerate=config['framerate'],
                output=config['output'],
                udp_target=config['udp_target'],
                bitrate=config['bitrate'],
                preset=config['preset'],
                tune=config['tune'],
                qp=config['qp'],
                offset_x=config.get('offset_x'),
                offset_y=config.get('offset_y'),
                dimension_wait_time=config.get('dimension_wait_time', 5.0)
            )
            
            self.streamer = ScreenStreamer(stream_config)
            # Skip dimension check since we already did it
            return self.streamer.start(skip_dimension_check=True)
            
        except Exception as e:
            logger.error(f"Error in setup and stream: {e}")
            return False
    
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
    display_group.add_argument("--width", type=int, default=1920, 
                              help="Width of the display in pixels")
    display_group.add_argument("--height", type=int, default=1080, 
                              help="Height of the display in pixels")
    display_group.add_argument("--refresh-rate", type=float, default=60.0, 
                              help="Refresh rate in Hz")
    display_group.add_argument("--output", default="HDMI-1",
                              help="Output name for the extended screen")
    display_group.add_argument("--relative-to", default="eDP-1",
                              help="Primary screen to position relative to")
    display_group.add_argument("--position", default="left-of",
                              choices=[p.value for p in Position],
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
    stream_group.add_argument("--dimension-wait-time", type=float, default=5.0,
                             help="Time to wait for client dimensions in seconds")
    
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
    """Main entry point for the program."""
    args = parse_arguments()
    config = vars(args)
    
    # Configure logging based on debug flag
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    manager = ScreenManager()
    
    # Handle Ctrl-C gracefully
    def signal_handler(sig, frame):
        logger.info("Stopping and cleaning up...")
        if not args.no_cleanup:
            manager.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        if args.setup_only:
            if not manager.setup_display(config):
                logger.error("Display setup failed")
                sys.exit(1)
            logger.info("Display setup complete. Press Ctrl+C to clean up...")
            
        elif args.stream_only:
            if not manager.start_streaming(config):
                logger.error("Stream start failed")
                sys.exit(1)
            logger.info("Streaming started. Press Ctrl+C to stop...")
            
        else:
            if not manager.setup_and_stream(config):
                logger.error("Display setup and/or streaming failed")
                sys.exit(1)
            logger.info("Display setup and streaming started. Press Ctrl+C to stop...")
        
        # Main loop
        while True:
            time.sleep(1)
            if (manager.streamer and manager.streamer.process and 
                manager.streamer.process.poll() is not None):
                logger.error("Streaming process ended unexpectedly")
                break
                
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if not args.no_cleanup:
            manager.cleanup()


if __name__ == "__main__":
    main()
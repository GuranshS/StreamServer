#!/usr/bin/env python3
"""
Screen Manager - A tool for creating and streaming virtual displays with Network Service Discovery.

This program sets up custom display resolutions and can stream display content
over UDP with low latency using FFmpeg. Now includes mDNS service discovery
for automatic client connection.

Updated for client-only architecture where clients discover and connect to servers.
Server waits for client connection and dimensions before creating display and streaming.
"""

import argparse
import dataclasses
import enum
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import threading
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

# Service Discovery imports
try:
    from zeroconf import Zeroconf, ServiceInfo
    ZEROCONF_AVAILABLE = True
except ImportError:
    ZEROCONF_AVAILABLE = False
    print("Warning: zeroconf not available. Install with: pip install zeroconf")


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


class ServiceDiscoveryError(ScreenManagerError):
    """Exception for service discovery errors."""
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
    udp_port: int
    client_address: str  # Target client address for streaming
    bitrate: str
    preset: str
    tune: str
    qp: int
    offset_x: Optional[int] = None
    offset_y: Optional[int] = None
    dimension_wait_time: float = 5.0
    low_latency: bool = False
    buffer_size: str = "auto"
    keyframe_interval: Optional[int] = None
    packet_size: int = 1316


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


@dataclass
class ClientConnectionInfo:
    """Information about a connected client."""
    address: str
    port: int
    width: int
    height: int
    framerate: int
    connected_at: float


class ClientListener:
    """Listens for client connections and dimension requests."""
    
    def __init__(self, port: int, timeout: float = 30.0):
        """
        Initialize client listener.
        
        Args:
            port: Port to listen on
            timeout: Timeout for waiting for client connection
        """
        self.port = port
        self.timeout = timeout
        self.socket = None
        self.client_info = None
        self.listening = False
        self._stop_event = threading.Event()
    
    def start_listening(self) -> Optional[ClientConnectionInfo]:
        """
        Start listening for client connections.
        
        Returns:
            ClientConnectionInfo if a client connects, None if timeout
        """
        try:
            # Create UDP socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.settimeout(2.0)  # Short timeout for checking stop event
            self.socket.bind(('0.0.0.0', self.port))
            
            logger.info(f"Listening for client connections on UDP port {self.port}")
            logger.info(f"Waiting for client to connect (timeout: {self.timeout}s)...")
            
            self.listening = True
            start_time = time.time()
            
            while self.listening and (time.time() - start_time) < self.timeout:
                if self._stop_event.is_set():
                    break
                
                try:
                    # Wait for client connection message
                    data, client_addr = self.socket.recvfrom(1024)
                    message = data.decode('utf-8')
                    
                    logger.info(f"Received message from {client_addr[0]}:{client_addr[1]}: {message}")
                    
                    # Parse client connection message
                    # Expected format: "CONNECT:width:height:framerate" or JSON
                    client_info = self._parse_client_message(message, client_addr[0])
                    
                    if client_info:
                        logger.info(f"Client connected: {client_info.address} requesting {client_info.width}x{client_info.height}@{client_info.framerate}fps")
                        
                        # Send acknowledgment back to client
                        ack_message = f"ACK:SERVER_READY"
                        self.socket.sendto(ack_message.encode('utf-8'), client_addr)
                        
                        self.client_info = client_info
                        self.listening = False
                        return client_info
                    
                except socket.timeout:
                    # Continue waiting
                    continue
                except Exception as e:
                    logger.debug(f"Error receiving data: {e}")
                    continue
            
            if time.time() - start_time >= self.timeout:
                logger.warning(f"No client connected within {self.timeout} seconds")
            
            return None
            
        except Exception as e:
            logger.error(f"Error starting client listener: {e}")
            return None
        finally:
            self.stop_listening()
    
    def _parse_client_message(self, message: str, client_address: str) -> Optional[ClientConnectionInfo]:
        """
        Parse client connection message.
        
        Args:
            message: Message from client
            client_address: Client IP address
            
        Returns:
            ClientConnectionInfo if valid, None otherwise
        """
        try:
            # Try JSON format first
            if message.startswith('{'):
                data = json.loads(message)
                return ClientConnectionInfo(
                    address=client_address,
                    port=data.get('port', self.port),
                    width=data.get('width', 1920),
                    height=data.get('height', 1080),
                    framerate=data.get('framerate', 30),
                    connected_at=time.time()
                )
            
            # Try simple format: "CONNECT:width:height:framerate"
            elif message.startswith('CONNECT:'):
                parts = message.split(':')
                if len(parts) >= 4:
                    width = int(parts[1])
                    height = int(parts[2])
                    framerate = int(parts[3])
                    
                    return ClientConnectionInfo(
                        address=client_address,
                        port=self.port,
                        width=width,
                        height=height,
                        framerate=framerate,
                        connected_at=time.time()
                    )
            
            # Default connection (just "CONNECT" or any other message)
            elif 'CONNECT' in message.upper() or message.strip():
                logger.info("Client connected with default parameters")
                return ClientConnectionInfo(
                    address=client_address,
                    port=self.port,
                    width=1920,  # Default dimensions
                    height=1080,
                    framerate=30,
                    connected_at=time.time()
                )
            
        except Exception as e:
            logger.warning(f"Error parsing client message '{message}': {e}")
        
        return None
    
    def stop_listening(self):
        """Stop listening for client connections."""
        self.listening = False
        self._stop_event.set()
        if self.socket:
            try:
                self.socket.close()
            except Exception as e:
                logger.debug(f"Error closing socket: {e}")
            self.socket = None


class ServiceDiscoveryManager:
    """Manages mDNS service discovery for screen streaming (server-only mode)."""
    
    # Match the Android client's service type exactly
    SERVICE_TYPE = "_screenstream._tcp.local."
    
    def __init__(self, service_name: str = None):
        """
        Initialize service discovery manager.
        
        Args:
            service_name: Name for this service (defaults to hostname)
        """
        if not ZEROCONF_AVAILABLE:
            raise ServiceDiscoveryError("zeroconf library not available")
        
        self.zeroconf = Zeroconf()
        self.service_name = service_name or f"LinuxServer_{socket.gethostname()}"
        self.service_info = None
        self._lock = threading.Lock()
        
    def register_service(self, port: int, width: int = 1920, height: int = 1080, 
                        framerate: int = 30, **kwargs) -> None:
        """
        Register this service for discovery by clients.
        
        Args:
            port: UDP port for streaming
            width: Stream width
            height: Stream height
            framerate: Stream framerate
            **kwargs: Additional service properties
        """
        properties = {
            'width': str(width),
            'height': str(height),
            'framerate': str(framerate),
            'version': '1.0',
            'type': 'linux_server',  # Identifies this as a server
            'hostname': socket.gethostname(),
            'os': 'Linux',
            'protocol': 'udp',
            'encoding': 'h264',
            'preset': kwargs.get('preset', 'ultrafast'),
            'tune': kwargs.get('tune', 'zerolatency'),
            'bitrate': kwargs.get('bitrate', '1M'),
            'status': 'waiting',  # Server is waiting for client connection
            **{k: str(v) for k, v in kwargs.items()}
        }
        
        # Encode properties for zeroconf - ensure UTF-8 encoding
        encoded_props = {}
        for k, v in properties.items():
            try:
                key_bytes = k.encode('utf-8')
                val_bytes = str(v).encode('utf-8')
                encoded_props[key_bytes] = val_bytes
            except Exception as e:
                logger.warning(f"Error encoding property {k}={v}: {e}")
        
        # Create service info
        service_name = f"{self.service_name}.{self.SERVICE_TYPE}"
        
        try:
            # Get local IP address
            local_ip = self._get_local_ip()
            ip_bytes = socket.inet_aton(local_ip)
            
            self.service_info = ServiceInfo(
                self.SERVICE_TYPE,
                service_name,
                addresses=[ip_bytes],
                port=port,
                properties=encoded_props,
                server=f"{socket.gethostname()}.local."
            )
            
            self.zeroconf.register_service(self.service_info)
            logger.info(f"Registered server service: {self.service_name} on {local_ip}:{port}")
            logger.info(f"Service type: {self.SERVICE_TYPE}")
            logger.info(f"Service properties: {properties}")
            logger.info(f"Status: Waiting for client connection...")
            
        except Exception as e:
            raise ServiceDiscoveryError(f"Failed to register service: {e}")
    
    def update_service_status(self, status: str, **kwargs) -> None:
        """
        Update service status and properties.
        
        Args:
            status: New status (waiting, connected, streaming)
            **kwargs: Additional properties to update
        """
        if not self.service_info:
            return
        
        try:
            # Get current properties
            current_props = {}
            if self.service_info.properties:
                for k, v in self.service_info.properties.items():
                    try:
                        key_str = k.decode('utf-8') if isinstance(k, bytes) else str(k)
                        val_str = v.decode('utf-8') if isinstance(v, bytes) else str(v)
                        current_props[key_str] = val_str
                    except Exception:
                        pass
            
            # Update with new status and properties
            current_props['status'] = status
            for k, v in kwargs.items():
                current_props[k] = str(v)
            
            # Re-encode properties
            encoded_props = {}
            for k, v in current_props.items():
                try:
                    key_bytes = k.encode('utf-8')
                    val_bytes = str(v).encode('utf-8')
                    encoded_props[key_bytes] = val_bytes
                except Exception as e:
                    logger.warning(f"Error encoding property {k}={v}: {e}")
            
            # Update service info
            self.service_info.properties = encoded_props
            
            # Re-register to update
            self.zeroconf.update_service(self.service_info)
            logger.info(f"Updated service status to: {status}")
            
        except Exception as e:
            logger.error(f"Error updating service status: {e}")
    
    def _get_local_ip(self) -> str:
        """Get the local IP address."""
        try:
            # Create a socket to determine the local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception as e:
            logger.warning(f"Could not determine local IP: {e}")
            return "127.0.0.1"
    
    def unregister_service(self) -> None:
        """Unregister this service."""
        if self.service_info:
            try:
                self.zeroconf.unregister_service(self.service_info)
                logger.info("Unregistered server service")
            except Exception as e:
                logger.error(f"Error unregistering service: {e}")
    
    def close(self) -> None:
        """Close the service discovery manager."""
        self.unregister_service()
        self.zeroconf.close()


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
        self.terminal_process = None
        self.ffmpeg_pid = None
    
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
        
        # Calculate buffer size for latency optimization
        bitrate_value = float(self.config.bitrate[:-1])
        bitrate_unit = self.config.bitrate[-1]
        
        if self.config.buffer_size == "auto":
            if self.config.low_latency:
                bufsize_multiplier = 0.25  # Very small buffer for ultra-low latency
            else:
                bufsize_multiplier = 0.5   # Moderate buffer
        else:
            bufsize_multiplier = float(self.config.buffer_size)
        
        bufsize_value = bitrate_value * bufsize_multiplier
        bufsize = f"{max(int(bufsize_value), 32)}{bitrate_unit}"  # Minimum 32k buffer
        
        # Determine keyframe interval
        keyframe_interval = self.config.keyframe_interval or self.config.framerate
        if self.config.low_latency:
            keyframe_interval = min(keyframe_interval, 15)  # More frequent keyframes for low latency
        
        # Build UDP target to stream to client
        udp_target = f"udp://{self.config.client_address}:{self.config.udp_port}"
        
        cmd = [
            "ffmpeg",
            "-f", "x11grab",
            "-video_size", f"{self.config.width}x{self.config.height}",
            "-framerate", str(self.config.framerate),
        ]
        
        # Add low-latency input options
        if self.config.low_latency:
            cmd.extend([
                "-probesize", "32",
                "-analyzeduration", "0",
                "-fflags", "+nobuffer",
            ])
        else:
            cmd.extend([
                "-probesize", "1000000",  # 1MB probe size
            ])
        
        cmd.extend([
            "-i", f":0.0+{offset_x},{offset_y}",
            "-c:v", "libx264",
            "-preset", self.config.preset,
            "-tune", self.config.tune,
            "-qp", str(self.config.qp),
            "-b:v", self.config.bitrate,
            "-maxrate", self.config.bitrate,
            "-bufsize", bufsize,
            "-g", str(keyframe_interval),
        ])
        
        # Add low-latency encoding options
        if self.config.low_latency:
            cmd.extend([
                "-bf", "0",           # No B-frames
                "-refs", "1",         # Single reference frame
                "-rc-lookahead", "0", # No lookahead
                "-slices", "1",       # Single slice
                "-intra-refresh", "1", # Intra refresh
            ])
        
        cmd.extend([
            "-profile:v", "baseline",
            "-pix_fmt", "yuv420p",
            "-flags", "+global_header",
            "-avoid_negative_ts", "make_zero",
        ])
        
        # Add output fflags for low latency
        if self.config.low_latency:
            cmd.extend(["-fflags", "+nobuffer+flush_packets"])
        
        cmd.extend([
            "-f", "mpegts",
        ])
        
        # Add UDP options for lower latency
        if self.config.low_latency:
            cmd.append(f"{udp_target}?pkt_size={self.config.packet_size}&buffer_size=0&flush_packets=1")
        else:
            cmd.append(f"{udp_target}?pkt_size={self.config.packet_size}")
        
        return cmd
    
    def detect_terminal_emulator(self) -> Optional[Tuple[List[str], str, Optional[str]]]:
        """
        Detect available terminal emulator and return command to launch it.
        
        Returns:
            Tuple of (terminal_cmd, exec_flag, title_flag) or None if none found
        """
        # Optimized for Linux Mint (Cinnamon desktop)
        terminals = [
            # Terminal command, execute flag, title flag (if supported)
            (["gnome-terminal"], "--", "--title"),
            (["x-terminal-emulator"], "-e", None),  # Debian/Ubuntu default
            (["mate-terminal"], "-e", "--title"),
            (["xfce4-terminal"], "-e", "--title"),
            (["konsole"], "-e", "--title"),
            (["terminator"], "-e", "--title"),
            (["tilix"], "-e", "--title"),
            (["alacritty"], "-e", "--title"),
            (["kitty"], "-e", "--title"),
            (["xterm"], "-e", "-title"),
            (["urxvt"], "-e", "-title"),
            (["st"], "-e", "-t"),
        ]
        
        for terminal_cmd, exec_flag, title_flag in terminals:
            try:
                # Check if terminal is available
                result = subprocess.run(
                    ["which", terminal_cmd[0]], 
                    capture_output=True, 
                    text=True
                )
                if result.returncode == 0:
                    logger.info(f"Found terminal emulator: {terminal_cmd[0]}")
                    return terminal_cmd, exec_flag, title_flag
            except Exception:
                continue
        
        logger.warning("No suitable terminal emulator found")
        return None
    
    def launch_in_terminal(self, use_terminal: bool = True) -> bool:
        """
        Launch FFmpeg in a new terminal or as a detached process.
        
        Args:
            use_terminal: Whether to launch in a new terminal window
            
        Returns:
            True if successful
        """
        cmd = self.build_ffmpeg_command()
        cmd_str = " ".join(f'"{arg}"' if " " in arg else arg for arg in cmd)
        
        if use_terminal:
            terminal_info = self.detect_terminal_emulator()
            if terminal_info:
                terminal_cmd, exec_flag, title_flag = terminal_info
                
                # Build terminal command based on terminal type
                if terminal_cmd[0] == "gnome-terminal":
                    # Try modern gnome-terminal syntax first
                    full_cmd = [
                        "gnome-terminal",
                        "--title=Screen Manager - FFmpeg Stream",
                        "--",
                    ] + cmd
                    
                    # For debugging, also try the simpler approach
                    alternative_cmd = [
                        "gnome-terminal",
                        "--",
                    ] + cmd
                    
                elif terminal_cmd[0] == "x-terminal-emulator":
                    # Simple execution without title
                    full_cmd = ["x-terminal-emulator", "-e"] + cmd
                    alternative_cmd = None
                else:
                    # Standard terminal command construction
                    full_cmd = terminal_cmd.copy()
                    
                    # Add title if supported
                    if title_flag:
                        full_cmd.extend([title_flag, "Screen Manager - FFmpeg Stream"])
                    
                    # Add execute flag and command
                    full_cmd.extend([exec_flag] + cmd)
                    alternative_cmd = None
                
                logger.info(f"Launching FFmpeg in terminal: {' '.join(full_cmd)}")
                
                try:
                    self.terminal_process = subprocess.Popen(
                        full_cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        preexec_fn=os.setsid  # Create new process group
                    )
                    
                    # Give it time to start
                    time.sleep(1)
                    
                    # For gnome-terminal and similar, the parent process may exit immediately
                    # but spawn a child process. We need to check for FFmpeg instead.
                    logger.info("Terminal command launched, waiting for FFmpeg to start...")
                    
                    # Wait for FFmpeg process to appear
                    max_wait = 10  # seconds
                    waited = 0
                    ffmpeg_found = False
                    
                    while waited < max_wait:
                        time.sleep(1)
                        waited += 1
                        
                        # Look for FFmpeg process
                        try:
                            result = subprocess.run(
                                ["pgrep", "-f", "ffmpeg.*x11grab"], 
                                capture_output=True, 
                                text=True
                            )
                            if result.returncode == 0 and result.stdout.strip():
                                pids = result.stdout.strip().split('\n')
                                if pids:
                                    self.ffmpeg_pid = int(pids[0])
                                    logger.info(f"FFmpeg process found with PID: {self.ffmpeg_pid}")
                                    ffmpeg_found = True
                                    break
                        except Exception as e:
                            logger.debug(f"Error checking for FFmpeg: {e}")
                    
                    if ffmpeg_found:
                        logger.info("FFmpeg launched successfully in terminal")
                        return True
                    else:
                        logger.warning("FFmpeg process not found, trying alternative command...")
                        # Try alternative command for gnome-terminal
                        if alternative_cmd and terminal_cmd[0] == "gnome-terminal":
                            logger.info(f"Trying alternative command: {' '.join(alternative_cmd)}")
                            try:
                                subprocess.Popen(
                                    alternative_cmd,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL,
                                    preexec_fn=os.setsid
                                )
                                
                                # Wait again for FFmpeg
                                time.sleep(3)
                                waited = 0
                                while waited < 5:
                                    time.sleep(1)
                                    waited += 1
                                    
                                    try:
                                        result = subprocess.run(
                                            ["pgrep", "-f", "ffmpeg.*x11grab"], 
                                            capture_output=True, 
                                            text=True
                                        )
                                        if result.returncode == 0 and result.stdout.strip():
                                            pids = result.stdout.strip().split('\n')
                                            if pids:
                                                self.ffmpeg_pid = int(pids[0])
                                                logger.info(f"FFmpeg process found with alternative command: {self.ffmpeg_pid}")
                                                return True
                                    except Exception:
                                        continue
                            except Exception as e:
                                logger.error(f"Alternative command also failed: {e}")
                        
                        logger.error("Failed to start FFmpeg in terminal")
                        return False
                        
                except Exception as e:
                    logger.error(f"Failed to launch terminal: {e}")
                    return False
            else:
                logger.warning("No terminal found, falling back to detached process")
                use_terminal = False
        
        if not use_terminal:
            # Launch as detached background process
            return self.launch_detached(cmd, cmd_str)
    
    def find_ffmpeg_pid(self) -> None:
        """Try to find the FFmpeg process PID."""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "ffmpeg.*x11grab"], 
                capture_output=True, 
                text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                if pids:
                    self.ffmpeg_pid = int(pids[0])  # Take the first PID
                    logger.debug(f"Found FFmpeg PID: {self.ffmpeg_pid}")
        except Exception as e:
            logger.debug(f"Could not find FFmpeg PID: {e}")
    
    def launch_detached(self, cmd: List[str], cmd_str: str) -> bool:
        """
        Launch FFmpeg as a detached background process.
        
        Args:
            cmd: Command list to execute
            cmd_str: Command string for logging
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Launching FFmpeg as detached process:\n{cmd_str}")
            
            # Launch with nohup-like behavior
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                preexec_fn=os.setsid,  # Create new process group
                start_new_session=True  # Start new session (Python 3.2+)
            )
            
            # Give it time to start
            time.sleep(2)
            
            if self.process.poll() is not None:
                stderr_output = self.process.stderr.read() if self.process.stderr else "No error output"
                raise StreamError(f"Failed to start ffmpeg as detached process: {stderr_output}")
            
            logger.info("FFmpeg launched successfully as detached process")
            return True
            
        except Exception as e:
            raise StreamError(f"Error starting detached process: {e}")
    
    def start(self, skip_dimension_check=False, use_terminal=True) -> bool:
        """
        Start the streaming process.
        
        Args:
            skip_dimension_check: Whether to skip waiting for client dimensions
            use_terminal: Whether to launch in a new terminal window
            
        Returns:
            True if successful
            
        Raises:
            StreamError: If streaming fails to start
        """
        try:
            return self.launch_in_terminal(use_terminal)
            
        except Exception as e:
            raise StreamError(f"Error starting stream: {e}")
    
    def stop(self) -> None:
        """Stop the streaming process."""
        # Stop terminal process if it exists
        if self.terminal_process and self.terminal_process.poll() is None:
            try:
                # Send SIGTERM to the entire process group
                os.killpg(os.getpgid(self.terminal_process.pid), signal.SIGTERM)
                try:
                    self.terminal_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't respond
                    os.killpg(os.getpgid(self.terminal_process.pid), signal.SIGKILL)
                logger.info("Terminal streaming process stopped")
            except Exception as e:
                logger.error(f"Error stopping terminal process: {e}")
        
        # Stop regular process if it exists
        if self.process and self.process.poll() is None:
            try:
                # Send SIGTERM to the entire process group
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't respond
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                logger.info("Detached streaming process stopped")
            except Exception as e:
                logger.error(f"Error stopping detached process: {e}")

    def is_running(self) -> bool:
        """
        Check if the streaming process is still running.
        
        Returns:
            True if running, False otherwise
        """
        # If we have a specific FFmpeg PID, check that first
        if self.ffmpeg_pid:
            try:
                # Check if the specific FFmpeg process is still running
                os.kill(self.ffmpeg_pid, 0)  # Signal 0 just checks if process exists
                return True
            except (OSError, ProcessLookupError):
                # Process doesn't exist anymore
                logger.debug(f"FFmpeg process {self.ffmpeg_pid} no longer exists")
                self.ffmpeg_pid = None  # Clear the dead PID
                return False
        
        # Fallback: check if any ffmpeg x11grab process exists
        try:
            result = subprocess.run(
                ["pgrep", "-f", "ffmpeg.*x11grab"], 
                capture_output=True, 
                text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                # Found an FFmpeg process, update our PID
                pids = result.stdout.strip().split('\n')
                if pids:
                    self.ffmpeg_pid = int(pids[0])
                    logger.debug(f"Found FFmpeg process with PID: {self.ffmpeg_pid}")
                return True
        except Exception as e:
            logger.debug(f"Error checking for FFmpeg process: {e}")
        
        # Check terminal process as last resort
        if self.terminal_process:
            return self.terminal_process.poll() is None
        elif self.process:
            return self.process.poll() is None
            
        return False

    def monitor_process(self) -> bool:
        """
        Monitor the streaming process and return when it ends.
        
        Returns:
            True if process ended normally, False if there was an error
        """
        if not self.is_running():
            logger.error("No streaming process is running")
            return False
        
        try:
            check_interval = 2  # Check every 2 seconds
            logger.info("Monitoring streaming process...")
            logger.info("*** Close the FFmpeg terminal window to stop streaming and cleanup ***")
            
            while True:
                if not self.is_running():
                    logger.info("Streaming process has ended")
                    return True
                
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring interrupted by user")
            return True
        except Exception as e:
            logger.error(f"Error monitoring process: {e}")
            return False


class ScreenManager:
    """Main class that manages both display setup and streaming with client connection waiting."""
    
    def __init__(self, service_name: str = None, enable_discovery: bool = True):
        """
        Initialize the screen manager.
        
        Args:
            service_name: Name for service discovery (defaults to hostname)
            enable_discovery: Whether to enable service discovery
        """
        self.streamer = None
        self.output_name = None
        self.mode_name = None
        self.cleanup_needed = False
        self.fallback_width = 1920
        self.fallback_height = 1200
        self.actual_width = None
        self.actual_height = None
        self.client_info = None
        
        # Service discovery - simplified for server-only mode
        self.enable_discovery = enable_discovery and ZEROCONF_AVAILABLE
        self.discovery_manager = None
        
        if self.enable_discovery:
            try:
                self.discovery_manager = ServiceDiscoveryManager(service_name)
            except ServiceDiscoveryError as e:
                logger.warning(f"Service discovery disabled: {e}")
                self.enable_discovery = False
    
    def wait_for_client(self, port: int, timeout: float = 30.0) -> Optional[ClientConnectionInfo]:
        """
        Wait for a client to connect and provide dimensions.
        
        Args:
            port: Port to listen on
            timeout: Timeout for waiting for client
            
        Returns:
            ClientConnectionInfo if successful, None if timeout
        """
        listener = ClientListener(port, timeout)
        client_info = listener.start_listening()
        
        if client_info:
            self.client_info = client_info
            logger.info(f"Client connection established!")
            logger.info(f"  Address: {client_info.address}")
            logger.info(f"  Dimensions: {client_info.width}x{client_info.height}")
            logger.info(f"  Framerate: {client_info.framerate}")
        
        return client_info
    
    def start_service_discovery(self, port: int = 5001, **kwargs) -> None:
        """
        Start service discovery (register this server).
        
        Args:
            port: Port to advertise
            **kwargs: Additional service properties
        """
        if not self.enable_discovery:
            logger.warning("Service discovery not available")
            return
        
        try:
            self.discovery_manager.register_service(port, **kwargs)
            logger.info("Server registered for discovery - clients can now find this server")
        except Exception as e:
            logger.error(f"Failed to start service discovery: {e}")
    
    def update_service_status(self, status: str, **kwargs) -> None:
        """Update service discovery status."""
        if self.enable_discovery and self.discovery_manager:
            self.discovery_manager.update_service_status(status, **kwargs)
    
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
                self.actual_width = width
                self.actual_height = height
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
    
    def start_streaming(self, config: Union[StreamConfig, Dict], use_terminal: bool = True) -> bool:
        """
        Start streaming with the given configuration.
        
        Args:
            config: Streaming configuration
            use_terminal: Whether to launch in terminal
            
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
                udp_port=config['udp_port'],
                client_address=config['client_address'],
                bitrate=config['bitrate'],
                preset=config['preset'],
                tune=config['tune'],
                qp=config['qp'],
                offset_x=config.get('offset_x'),
                offset_y=config.get('offset_y'),
                dimension_wait_time=config.get('dimension_wait_time', 5.0),
                low_latency=config.get('low_latency', False),
                buffer_size=config.get('buffer_size', 'auto'),
                keyframe_interval=config.get('keyframe_interval'),
                packet_size=config.get('packet_size', 1316)
            )
        else:
            stream_config = config
        
        self.streamer = ScreenStreamer(stream_config)
        return self.streamer.start(use_terminal=use_terminal)
    
    def wait_setup_and_stream(self, config: Dict, use_terminal: bool = True, 
                             client_timeout: float = 30.0) -> bool:
        """
        Wait for client connection, then set up display and start streaming.
        
        Args:
            config: Base configuration
            use_terminal: Whether to launch FFmpeg in terminal
            client_timeout: How long to wait for client connection
            
        Returns:
            True if successful
        """
        try:
            # Start service discovery if enabled
            if self.enable_discovery:
                self.start_service_discovery(
                    port=config.get('udp_port', 5001),
                    width=config.get('width', 1920),  # Default dimensions
                    height=config.get('height', 1080),
                    framerate=config.get('framerate', 30),
                    preset=config.get('preset', 'ultrafast'),
                    tune=config.get('tune', 'zerolatency'),
                    bitrate=config.get('bitrate', '1M')
                )
            
            # Wait for client connection
            logger.info("=" * 60)
            logger.info("WAITING FOR CLIENT CONNECTION")
            logger.info("=" * 60)
            logger.info("Server is ready and waiting for a client to connect...")
            if self.enable_discovery:
                logger.info("Clients can discover this server automatically via mDNS")
            logger.info(f"Listening on UDP port: {config.get('udp_port', 5001)}")
            logger.info("=" * 60)
            
            client_info = self.wait_for_client(
                port=config.get('udp_port', 5001),
                timeout=client_timeout
            )
            
            if not client_info:
                logger.error("No client connected within timeout period")
                return False
            
            # Update service status
            self.update_service_status("connected", 
                                     client_width=client_info.width,
                                     client_height=client_info.height)
            
            # Update config with client information
            config['width'] = client_info.width
            config['height'] = client_info.height
            config['framerate'] = client_info.framerate
            config['client_address'] = client_info.address
            
            logger.info(f"Configured for client: {client_info.address}")
            logger.info(f"Display dimensions: {client_info.width}x{client_info.height}@{client_info.framerate}fps")
            
            # Set up display with client dimensions
            display_config = DisplayConfig(
                width=config['width'],
                height=config['height'],
                refresh_rate=config['refresh_rate'],
                output=config['output'],
                position=Position(config['position']),
                relative_to=config['relative_to']
            )
            
            logger.info("Setting up virtual display...")
            if not self.setup_display(display_config):
                logger.error("Failed to setup display")
                return False
            
            # Give the display a moment to stabilize
            time.sleep(2)
            
            # Update config with actual dimensions after setup
            if self.actual_width and self.actual_height:
                if self.actual_width != config['width'] or self.actual_height != config['height']:
                    logger.info(f"Display created with resolution: {self.actual_width}x{self.actual_height}")
                    config['width'] = self.actual_width
                    config['height'] = self.actual_height
                    
                    # Update service discovery with actual dimensions
                    self.update_service_status("streaming",
                                             actual_width=self.actual_width,
                                             actual_height=self.actual_height)
            
            # Create streamer with the updated config
            stream_config = StreamConfig(
                width=config['width'],
                height=config['height'],
                framerate=config['framerate'],
                output=config['output'],
                udp_port=config['udp_port'],
                client_address=config['client_address'],
                bitrate=config['bitrate'],
                preset=config['preset'],
                tune=config['tune'],
                qp=config['qp'],
                offset_x=config.get('offset_x'),
                offset_y=config.get('offset_y'),
                dimension_wait_time=config.get('dimension_wait_time', 5.0),
                low_latency=config.get('low_latency', False),
                buffer_size=config.get('buffer_size', 'auto'),
                keyframe_interval=config.get('keyframe_interval'),
                packet_size=config.get('packet_size', 1316)
            )
            
            logger.info("Starting video stream...")
            self.streamer = ScreenStreamer(stream_config)
            success = self.streamer.start(skip_dimension_check=True, use_terminal=use_terminal)
            
            if success:
                self.update_service_status("streaming")
                logger.info("Stream started successfully!")
            
            return success
            
        except Exception as e:
            logger.error(f"Error in wait, setup and stream: {e}")
            return False
    
    def cleanup(self, keep_display: bool = False) -> None:
        """
        Clean up resources.
        
        Args:
            keep_display: If True, don't remove the virtual display
        """
        if self.streamer:
            self.streamer.stop()
        
        if self.enable_discovery and self.discovery_manager:
            self.discovery_manager.close()
        
        if not keep_display and self.cleanup_needed and self.output_name and self.mode_name:
            DisplayManager.cleanup_display(self.output_name, self.mode_name)
            self.cleanup_needed = False
        elif keep_display and self.cleanup_needed:
            logger.info(f"Keeping virtual display {self.output_name} active (mode: {self.mode_name})")
            logger.info(f"To manually remove later, run: xrandr --output {self.output_name} --off")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Server that waits for client connections and streams screen content",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Display parameters (defaults, overridden by client)
    display_group = parser.add_argument_group('Display Options')
    display_group.add_argument("--width", type=int, default=1920, 
                              help="Default width (overridden by client)")
    display_group.add_argument("--height", type=int, default=1080, 
                              help="Default height (overridden by client)")
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
                             help="Default framerate (overridden by client)")
    stream_group.add_argument("--udp-port", type=int, default=5001,
                             help="UDP port for client connections and streaming")
    stream_group.add_argument("--bitrate", default="1M",
                             help="Target bitrate (e.g., 1M, 500K)")
    stream_group.add_argument("--preset", default="ultrafast",
                             choices=["ultrafast", "superfast", "veryfast", "faster", "fast"],
                             help="Encoding preset")
    stream_group.add_argument("--tune", default="zerolatency",
                             choices=["zerolatency", "fastdecode"],
                             help="Encoding tuning")
    stream_group.add_argument("--qp", type=int, default=23,
                             help="Quantization parameter (0-51, lower is better quality)")
    stream_group.add_argument("--offset-x", type=int, help="Manual X offset for capture")
    stream_group.add_argument("--offset-y", type=int, help="Manual Y offset for capture")
    
    # Client connection options
    client_group = parser.add_argument_group('Client Connection Options')
    client_group.add_argument("--client-timeout", type=float, default=60.0,
                             help="Timeout for waiting for client connection")
    
    # Low latency options
    latency_group = parser.add_argument_group('Latency Options')
    latency_group.add_argument("--low-latency", action="store_true",
                              help="Enable aggressive low latency settings")
    latency_group.add_argument("--buffer-size", default="auto",
                              help="Buffer size multiplier (auto, 0.1, 0.5, 1.0, 2.0)")
    latency_group.add_argument("--keyframe-interval", type=int, default=None,
                              help="Keyframe interval (default: framerate)")
    latency_group.add_argument("--packet-size", type=int, default=1316,
                              help="UDP packet size for streaming")
    
    # Service Discovery options
    discovery_group = parser.add_argument_group('Service Discovery Options')
    discovery_group.add_argument("--no-discovery", action="store_true",
                                help="Disable service discovery")
    discovery_group.add_argument("--service-name", default=None,
                                help="Service name for discovery (defaults to hostname)")
    
    # Process launching options
    launch_group = parser.add_argument_group('Launch Options')
    launch_group.add_argument("--no-terminal", action="store_true",
                             help="Launch FFmpeg as detached background process instead of terminal")
    
    # Operation modes
    mode_group = parser.add_argument_group('Mode Options')
    mode_group.add_argument("--no-cleanup", action="store_true",
                           help="Don't automatically clean up after setup")
    mode_group.add_argument("--keep-display", action="store_true",
                           help="Keep the virtual display active even after streaming ends")
    mode_group.add_argument("--debug", action="store_true",
                           help="Show additional debug information")
    
    return parser.parse_args()


def main():
    """Main entry point for the program."""
    args = parse_arguments()
    config = vars(args)
    
    # Terminal mode is now the default (use terminal unless --no-terminal is specified)
    use_terminal = not args.no_terminal
    
    # Configure logging based on debug flag
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Check if zeroconf is available for service discovery
    enable_discovery = not args.no_discovery and ZEROCONF_AVAILABLE
    if args.no_discovery:
        logger.info("Service discovery disabled by user")
    elif not ZEROCONF_AVAILABLE:
        logger.warning("Service discovery unavailable (install zeroconf: pip install zeroconf)")
    
    logger.info("=" * 60)
    logger.info("SCREEN STREAMING SERVER")
    logger.info("=" * 60)
    logger.info(f"Launch mode: {'Terminal' if use_terminal else 'Detached process'}")
    logger.info(f"Service discovery: {'Enabled' if enable_discovery else 'Disabled'}")
    logger.info(f"Listening on UDP port: {args.udp_port}")
    logger.info("=" * 60)
    
    manager = ScreenManager(service_name=args.service_name, enable_discovery=enable_discovery)
    
    # Global flag to prevent multiple cleanup calls
    cleanup_done = False
    
    # Handle Ctrl-C gracefully
    def signal_handler(sig, frame):
        nonlocal cleanup_done
        if cleanup_done:
            logger.info("Already cleaning up, forcing exit...")
            sys.exit(1)
            
        cleanup_done = True
        logger.info("Stopping and cleaning up...")
        if not args.no_cleanup:
            manager.cleanup(keep_display=args.keep_display)
        logger.info("Cleanup complete, exiting...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Main server operation: wait for client, setup display, and stream
        success = manager.wait_setup_and_stream(
            config, 
            use_terminal, 
            args.client_timeout
        )
        
        if not success:
            logger.error("Failed to establish client connection or start streaming")
            sys.exit(1)
        
        # Give a moment for everything to stabilize
        time.sleep(2)
        
        # Verify streaming is actually running before continuing
        if not manager.streamer or not manager.streamer.is_running():
            logger.error("Streaming failed to start properly")
            sys.exit(1)
        
        if use_terminal:
            logger.info("=" * 60)
            logger.info("STREAMING ACTIVE")
            logger.info("=" * 60)
            logger.info("Server is now streaming to client!")
            if manager.client_info:
                logger.info(f"Client: {manager.client_info.address}")
                logger.info(f"Resolution: {manager.actual_width or config.get('width', 'unknown')}x{manager.actual_height or config.get('height', 'unknown')}")
            logger.info(f"UDP Port: {config.get('udp_port', 5001)}")
            logger.info("=" * 60)
            logger.info("To stop streaming and cleanup:")
            logger.info("  1. Close the FFmpeg terminal window, OR")
            logger.info("  2. Press Ctrl+C in this window")
            logger.info("=" * 60)
            
            # Monitor the terminal process - this will block until terminal closes
            manager.streamer.monitor_process()
            logger.info("Streaming ended, performing cleanup...")
        else:
            logger.info("Server started as background process. Press Ctrl+C to stop...")
            
            # Monitor the background process
            while True:
                if not manager.streamer.is_running():
                    logger.error("Streaming process ended unexpectedly")
                    break
                time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if not cleanup_done and not args.no_cleanup:
            logger.info("Performing final cleanup...")
            manager.cleanup(keep_display=args.keep_display)
            cleanup_done = True
        elif args.no_cleanup:
            logger.info("Cleanup skipped (--no-cleanup specified)")
            if args.keep_display:
                logger.info("Virtual display will remain active")


if __name__ == "__main__":
    main()
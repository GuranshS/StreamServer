# Screen Manager Documentation

## Overview

Screen Manager is a comprehensive Python tool for creating virtual displays and streaming screen content over UDP with Network Service Discovery (mDNS). It operates in a client-server architecture where the server waits for client connections, dynamically creates virtual displays based on client requirements, and streams the display content using FFmpeg.

## Table of Contents

1. [Installation & Dependencies](#installation--dependencies)
2. [Core Architecture](#core-architecture)
3. [Exception Classes](#exception-classes)
4. [Data Classes](#data-classes)
5. [Core Classes](#core-classes)
6. [Command Line Interface](#command-line-interface)
7. [Usage Examples](#usage-examples)
8. [Configuration Guide](#configuration-guide)

## Installation & Dependencies

### Required Dependencies
```bash
pip install zeroconf  # Optional, for service discovery
```

### System Requirements
- Linux with X11 (tested on Linux Mint/Cinnamon)
- FFmpeg with x11grab support
- xrandr utility
- Terminal emulator (gnome-terminal, xterm, etc.)

### Optional Dependencies
- zeroconf: For automatic client discovery via mDNS

## Core Architecture

The system follows a server-client model:
1. **Server** (this code): Creates virtual displays and streams content
2. **Client**: Connects to server and receives video stream
3. **Service Discovery**: Automatic client-server discovery via mDNS

## Exception Classes

### ScreenManagerError
**Base exception class for all Screen Manager errors.**

```python
class ScreenManagerError(Exception):
    """Base exception for Screen Manager errors."""
```

### DisplayError
**Exception for display-related operations.**

- **Inherits**: `ScreenManagerError`
- **Usage**: Raised when xrandr operations fail, CRTC errors occur, or display setup fails

### StreamError
**Exception for streaming-related operations.**

- **Inherits**: `ScreenManagerError`
- **Usage**: Raised when FFmpeg fails to start or streaming configuration is invalid

### CommandError
**Exception for command execution failures.**

- **Inherits**: `ScreenManagerError`
- **Usage**: Raised when shell commands fail or return non-zero exit codes

### ServiceDiscoveryError
**Exception for service discovery operations.**

- **Inherits**: `ScreenManagerError`
- **Usage**: Raised when mDNS registration fails or zeroconf is unavailable

## Data Classes

### Position
**Enumeration for display positioning.**

```python
class Position(enum.Enum):
    LEFT_OF = "left-of"
    RIGHT_OF = "right-of"
    ABOVE = "above"
    BELOW = "below"
```

**Values**:
- `LEFT_OF`: Position display to the left of reference display
- `RIGHT_OF`: Position display to the right of reference display
- `ABOVE`: Position display above reference display
- `BELOW`: Position display below reference display

### DisplayConfig
**Configuration for display setup.**

```python
@dataclass
class DisplayConfig:
    width: int              # Display width in pixels
    height: int             # Display height in pixels
    refresh_rate: float     # Refresh rate in Hz
    output: str             # Output name (e.g., "HDMI-1")
    position: Position      # Position relative to reference display
    relative_to: str        # Reference display name (e.g., "eDP-1")
```

**Example**:
```python
config = DisplayConfig(
    width=1920,
    height=1080,
    refresh_rate=60.0,
    output="HDMI-1",
    position=Position.RIGHT_OF,
    relative_to="eDP-1"
)
```

### StreamConfig
**Configuration for streaming setup.**

```python
@dataclass
class StreamConfig:
    width: int                          # Stream width in pixels
    height: int                         # Stream height in pixels
    framerate: int                      # Stream framerate in FPS
    output: str                         # Display output to capture
    udp_port: int                       # UDP port for streaming
    client_address: str                 # Target client IP address
    bitrate: str                        # Video bitrate (e.g., "1M", "500K")
    preset: str                         # FFmpeg preset ("ultrafast", "fast", etc.)
    tune: str                           # FFmpeg tuning ("zerolatency", "fastdecode")
    qp: int                            # Quantization parameter (0-51)
    offset_x: Optional[int] = None      # Manual X offset for capture area
    offset_y: Optional[int] = None      # Manual Y offset for capture area
    dimension_wait_time: float = 5.0    # Time to wait for dimensions
    low_latency: bool = False           # Enable aggressive low-latency settings
    buffer_size: str = "auto"           # Buffer size multiplier
    keyframe_interval: Optional[int] = None  # Keyframe interval
    packet_size: int = 1316             # UDP packet size
```

### ScreenInfo
**Information about a screen/display.**

```python
@dataclass
class ScreenInfo:
    name: str                                    # Display name
    status: str                                  # Connection status
    modes: List[str]                            # Available display modes
    current_mode: Optional[str] = None          # Current active mode
    position: Optional[Position] = None         # Position relative to other displays
    primary: bool = False                       # Whether this is the primary display
    geometry: Optional[Tuple[int, int, int, int]] = None  # (width, height, x, y)
```

### ClientConnectionInfo
**Information about connected client.**

```python
@dataclass
class ClientConnectionInfo:
    address: str        # Client IP address
    port: int          # Client port
    width: int         # Requested display width
    height: int        # Requested display height
    framerate: int     # Requested framerate
    connected_at: float # Connection timestamp
```

## Core Classes

### ClientListener
**Handles incoming client connections and dimension requests.**

#### `__init__(port: int, timeout: float = 30.0)`
**Initialize client listener.**

- **Inputs**:
  - `port`: UDP port to listen on
  - `timeout`: Maximum time to wait for client connection
- **Output**: ClientListener instance
- **Example**:
```python
listener = ClientListener(port=5001, timeout=60.0)
```

#### `start_listening() -> Optional[ClientConnectionInfo]`
**Start listening for client connections.**

- **Inputs**: None
- **Output**: `ClientConnectionInfo` if client connects, `None` if timeout
- **Process**:
  1. Creates UDP socket on specified port
  2. Waits for client connection messages
  3. Parses client requirements (dimensions, framerate)
  4. Sends acknowledgment back to client
  5. Returns client information

- **Supported Message Formats**:
  - JSON: `{"width": 1920, "height": 1080, "framerate": 30}`
  - Simple: `"CONNECT:1920:1080:30"`
  - Basic: `"CONNECT"` (uses defaults)

- **Example**:
```python
listener = ClientListener(5001)
client_info = listener.start_listening()
if client_info:
    print(f"Client {client_info.address} wants {client_info.width}x{client_info.height}")
```

#### `stop_listening()`
**Stop listening and close socket.**

- **Inputs**: None
- **Output**: None
- **Process**: Closes UDP socket and sets stop flag

### ServiceDiscoveryManager
**Manages mDNS service discovery for automatic client-server discovery.**

#### `__init__(service_name: str = None)`
**Initialize service discovery manager.**

- **Inputs**:
  - `service_name`: Service name (defaults to hostname)
- **Output**: ServiceDiscoveryManager instance
- **Raises**: `ServiceDiscoveryError` if zeroconf unavailable

#### `register_service(port: int, width: int = 1920, height: int = 1080, framerate: int = 30, **kwargs)`
**Register this server for client discovery.**

- **Inputs**:
  - `port`: UDP port for streaming
  - `width`: Default stream width
  - `height`: Default stream height
  - `framerate`: Default framerate
  - `**kwargs`: Additional service properties
- **Output**: None
- **Process**:
  1. Creates service properties dictionary
  2. Encodes properties for mDNS
  3. Registers service with zeroconf
  4. Advertises on `_screenstream._tcp.local.`

- **Example**:
```python
discovery = ServiceDiscoveryManager("MyServer")
discovery.register_service(port=5001, width=1920, height=1080)
```

#### `update_service_status(status: str, **kwargs)`
**Update service status and properties.**

- **Inputs**:
  - `status`: New status ("waiting", "connected", "streaming")
  - `**kwargs`: Additional properties to update
- **Output**: None
- **Process**: Updates service properties and re-registers with zeroconf

### CommandRunner
**Utility class for executing shell commands.**

#### `run(cmd: str, check: bool = True) -> Tuple[int, str, str]`
**Execute shell command and return results.**

- **Inputs**:
  - `cmd`: Shell command to execute
  - `check`: Whether to raise exception on failure
- **Output**: Tuple of `(return_code, stdout, stderr)`
- **Raises**: `CommandError` if command fails and check=True
- **Example**:
```python
code, stdout, stderr = CommandRunner.run("xrandr")
if code == 0:
    print("Command succeeded:", stdout)
```

### DisplayManager
**Manages virtual displays using xrandr.**

#### `get_screens() -> Dict[str, ScreenInfo]`
**Get information about all connected displays.**

- **Inputs**: None
- **Output**: Dictionary mapping display names to `ScreenInfo` objects
- **Process**:
  1. Runs `xrandr` command
  2. Parses output for connected displays
  3. Extracts modes, geometry, and status
  4. Returns structured display information

- **Example**:
```python
screens = DisplayManager.get_screens()
for name, info in screens.items():
    print(f"{name}: {info.status}, modes: {info.modes}")
```

#### `create_mode(config: DisplayConfig) -> str`
**Create a new display mode using cvt and xrandr.**

- **Inputs**: `DisplayConfig` object with mode specifications
- **Output**: Created mode name (e.g., "1920x1080_60.00")
- **Process**:
  1. Checks if mode already exists
  2. Uses `cvt` to generate modeline for specified resolution
  3. Creates new mode with `xrandr --newmode`
  4. Returns mode name for later use
- **Raises**: `DisplayError` if mode creation fails

- **Example**:
```python
config = DisplayConfig(1920, 1080, 60.0, "HDMI-1", Position.RIGHT_OF, "eDP-1")
mode_name = DisplayManager.create_mode(config)
```

#### `setup_display(config: DisplayConfig, mode_name: str) -> bool`
**Set up display with specified mode and position.**

- **Inputs**:
  - `config`: Display configuration
  - `mode_name`: Mode name returned by `create_mode()`
- **Output**: `True` if successful
- **Process**:
  1. Adds mode to specified output
  2. Activates output with mode and position
  3. Positions relative to reference display
- **Raises**: `DisplayError` if setup fails

#### `cleanup_display(output_name: str, mode_name: Optional[str] = None)`
**Clean up by disabling output and removing mode.**

- **Inputs**:
  - `output_name`: Display output to clean up
  - `mode_name`: Mode to remove (optional)
- **Output**: None
- **Process**:
  1. Removes mode from output
  2. Deletes mode definition
  3. Turns off output

### ScreenStreamer
**Handles FFmpeg-based screen streaming over UDP.**

#### `__init__(config: StreamConfig)`
**Initialize streamer with configuration.**

- **Inputs**: `StreamConfig` object
- **Output**: ScreenStreamer instance

#### `get_display_position() -> Tuple[int, int]`
**Determine capture area position on screen.**

- **Inputs**: None
- **Output**: Tuple of `(x_offset, y_offset)`
- **Process**:
  1. Queries current display configuration
  2. Calculates position based on geometry
  3. Returns offset for FFmpeg capture area

#### `build_ffmpeg_command() -> List[str]`
**Build FFmpeg command for screen capture and streaming.**

- **Inputs**: None
- **Output**: List of command arguments
- **Process**:
  1. Determines capture area offsets
  2. Calculates optimal buffer sizes
  3. Sets encoding parameters based on latency requirements
  4. Builds complete FFmpeg command for x11grab → H.264 → UDP

- **Generated Command Structure**:
```bash
ffmpeg -f x11grab -video_size 1920x1080 -framerate 30 \
  -i :0.0+1920,0 -c:v libx264 -preset ultrafast \
  -tune zerolatency -qp 23 -b:v 1M \
  -f mpegts udp://192.168.1.100:5001
```

#### `start(skip_dimension_check=False, use_terminal=True) -> bool`
**Start the streaming process.**

- **Inputs**:
  - `skip_dimension_check`: Whether to skip client dimension verification
  - `use_terminal`: Whether to launch in terminal window
- **Output**: `True` if started successfully
- **Process**:
  1. Builds FFmpeg command
  2. Launches either in terminal or as background process
  3. Monitors process startup
  4. Returns success status

#### `stop()`
**Stop the streaming process.**

- **Inputs**: None
- **Output**: None
- **Process**: Terminates FFmpeg process and cleans up resources

#### `is_running() -> bool`
**Check if streaming process is active.**

- **Inputs**: None
- **Output**: `True` if process is running
- **Process**: Checks process status and FFmpeg PID

#### `monitor_process() -> bool`
**Monitor streaming process until completion.**

- **Inputs**: None
- **Output**: `True` if ended normally
- **Process**: Continuously monitors process status until termination

### ScreenManager
**Main orchestrator class that coordinates all components.**

#### `__init__(service_name: str = None, enable_discovery: bool = True)`
**Initialize screen manager.**

- **Inputs**:
  - `service_name`: Name for service discovery
  - `enable_discovery`: Whether to enable mDNS discovery
- **Output**: ScreenManager instance

#### `wait_for_client(port: int, timeout: float = 30.0) -> Optional[ClientConnectionInfo]`
**Wait for client connection and get requirements.**

- **Inputs**:
  - `port`: UDP port to listen on
  - `timeout`: Maximum wait time
- **Output**: `ClientConnectionInfo` or `None`
- **Process**: Creates listener and waits for client connection

#### `setup_display(config: Union[DisplayConfig, Dict]) -> bool`
**Set up virtual display with fallback handling.**

- **Inputs**: Display configuration (object or dictionary)
- **Output**: `True` if successful
- **Process**:
  1. Attempts setup with requested resolution
  2. Falls back to 1920x1200 on CRTC errors
  3. Updates internal state with actual resolution
- **Fallback Logic**: Automatically tries 1920x1200 if original resolution fails due to CRTC limitations

#### `start_streaming(config: Union[StreamConfig, Dict], use_terminal: bool = True) -> bool`
**Start streaming with configuration.**

- **Inputs**:
  - `config`: Streaming configuration
  - `use_terminal`: Whether to use terminal window
- **Output**: `True` if successful

#### `wait_setup_and_stream(config: Dict, use_terminal: bool = True, client_timeout: float = 30.0) -> bool`
**Complete workflow: wait for client, setup display, start streaming.**

- **Inputs**:
  - `config`: Base configuration dictionary
  - `use_terminal`: Whether to launch FFmpeg in terminal
  - `client_timeout`: Client connection timeout
- **Output**: `True` if entire process successful
- **Process**:
  1. Starts service discovery
  2. Waits for client connection
  3. Updates configuration with client requirements
  4. Sets up virtual display
  5. Starts streaming to client

#### `cleanup(keep_display: bool = False)`
**Clean up all resources.**

- **Inputs**:
  - `keep_display`: Whether to keep virtual display active
- **Output**: None
- **Process**:
  1. Stops streaming process
  2. Unregisters service discovery
  3. Optionally removes virtual display

## Command Line Interface

### Basic Usage
```bash
python screen_manager.py [options]
```

### Display Options
- `--width INT`: Default width (default: 1920)
- `--height INT`: Default height (default: 1080)
- `--refresh-rate FLOAT`: Refresh rate in Hz (default: 60.0)
- `--output STR`: Output name (default: "HDMI-1")
- `--relative-to STR`: Primary screen name (default: "eDP-1")
- `--position CHOICE`: Position (left-of|right-of|above|below, default: left-of)

### Streaming Options
- `--framerate INT`: Default framerate (default: 30)
- `--udp-port INT`: UDP port (default: 5001)
- `--bitrate STR`: Video bitrate (default: "1M")
- `--preset CHOICE`: Encoding preset (default: "ultrafast")
- `--tune CHOICE`: Encoding tuning (default: "zerolatency")
- `--qp INT`: Quantization parameter 0-51 (default: 23)

### Latency Options
- `--low-latency`: Enable aggressive low-latency mode
- `--buffer-size STR`: Buffer size multiplier (default: "auto")
- `--keyframe-interval INT`: Keyframe interval
- `--packet-size INT`: UDP packet size (default: 1316)

### Service Discovery Options
- `--no-discovery`: Disable mDNS service discovery
- `--service-name STR`: Custom service name

### Process Options
- `--no-terminal`: Launch as background process instead of terminal
- `--keep-display`: Keep virtual display after streaming ends
- `--no-cleanup`: Skip automatic cleanup

## Usage Examples

### Basic Server Startup
```bash
# Start server with default settings
python screen_manager.py

# Start with custom port and resolution
python screen_manager.py --udp-port 8001 --width 2560 --height 1440

# Start with high quality settings
python screen_manager.py --qp 18 --bitrate 5M --preset fast
```

### Low Latency Streaming
```bash
# Ultra low latency mode
python screen_manager.py --low-latency --framerate 60 --buffer-size 0.1

# Custom latency optimization
python screen_manager.py --keyframe-interval 15 --packet-size 1316 --tune zerolatency
```

### Display Configuration
```bash
# Position virtual display to the right of primary
python screen_manager.py --position right-of --relative-to eDP-1

# Create display above primary with custom refresh rate
python screen_manager.py --position above --refresh-rate 75.0
```

### Background Operation
```bash
# Run as background process
python screen_manager.py --no-terminal

# Keep display active after streaming ends
python screen_manager.py --keep-display
```

### Service Discovery
```bash
# Disable automatic discovery
python screen_manager.py --no-discovery

# Custom service name
python screen_manager.py --service-name "MyStreamServer"
```

### Programmatic Usage
```python
from screen_manager import ScreenManager, DisplayConfig, StreamConfig, Position

# Create manager
manager = ScreenManager(service_name="MyServer")

# Wait for client and start streaming
config = {
    'width': 1920,
    'height': 1080,
    'refresh_rate': 60.0,
    'output': 'HDMI-1',
    'position': 'right-of',
    'relative_to': 'eDP-1',
    'framerate': 30,
    'udp_port': 5001,
    'bitrate': '2M',
    'preset': 'fast',
    'tune': 'zerolatency',
    'qp': 20
}

try:
    success = manager.wait_setup_and_stream(config, use_terminal=True)
    if success:
        print("Streaming started successfully")
        # Stream will continue until terminal is closed
except KeyboardInterrupt:
    print("Interrupted by user")
finally:
    manager.cleanup()
```

## Configuration Guide

### Performance Tuning

#### For Low Latency (Gaming/Real-time)
```python
config = {
    'low_latency': True,
    'preset': 'ultrafast',
    'tune': 'zerolatency',
    'qp': 25,
    'framerate': 60,
    'buffer_size': '0.1',
    'keyframe_interval': 15,
    'packet_size': 1316
}
```

#### For High Quality (Media/Presentations)
```python
config = {
    'low_latency': False,
    'preset': 'fast',
    'tune': 'zerolatency',
    'qp': 18,
    'bitrate': '5M',
    'framerate': 30,
    'keyframe_interval': 30
}
```

#### For Bandwidth Constrained Networks
```python
config = {
    'preset': 'ultrafast',
    'qp': 28,
    'bitrate': '500K',
    'framerate': 24,
    'width': 1280,
    'height': 720
}
```

### Display Positioning

#### Multiple Monitor Setup
```python
# Primary display: eDP-1 (laptop screen)
# Secondary: HDMI-1 (external monitor)
# Virtual: HDMI-2 (for streaming)

config = {
    'output': 'HDMI-2',
    'relative_to': 'HDMI-1',  # Position relative to external monitor
    'position': 'right-of',   # Place virtual display to the right
    'width': 1920,
    'height': 1080
}
```

### Client Message Formats

#### JSON Format (Recommended)
```json
{
    "width": 1920,
    "height": 1080,
    "framerate": 30,
    "port": 5001
}
```

#### Simple Format
```
CONNECT:1920:1080:30
```

#### Basic Format
```
CONNECT
```

This documentation provides comprehensive coverage of the Screen Manager system, including all classes, methods, configuration options, and usage examples. The system is designed to be flexible and handle various streaming scenarios from low-latency gaming to high-quality media streaming.

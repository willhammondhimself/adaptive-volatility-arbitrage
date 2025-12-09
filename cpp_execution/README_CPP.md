# C++ Execution Gateway - Build & Usage Guide

## Overview

This is a **low-latency order matching engine** demonstrating "Level 4" quant systems knowledge. It implements:
- **Price-time priority order book**: LIMIT and MARKET orders
- **Multi-threaded architecture**: Network I/O (Thread A) + Matching (Thread B)
- **ZeroMQ communication**: REP socket for orders, PUB socket for market data
- **Protobuf serialization**: Language-agnostic, compact, type-safe
- **Cross-platform**: macOS (Apple Silicon), Windows, Linux

**Performance targets** (local loopback):
- Order processing: <10 microseconds
- Queue latency: <100 nanoseconds
- Network round-trip: <100 microseconds

---

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation](#installation)
   - [macOS (Apple Silicon)](#macos-apple-silicon)
   - [macOS (Intel)](#macos-intel)
   - [Windows](#windows)
   - [Linux (Ubuntu/Debian)](#linux-ubuntudebian)
3. [Building](#building)
4. [Running](#running)
5. [Python Integration](#python-integration)
6. [Testing](#testing)
7. [Troubleshooting](#troubleshooting)
8. [Architecture](#architecture)
9. [Performance Tuning](#performance-tuning)

---

## System Requirements

- **Compiler**: C++17 or later (Clang 10+, GCC 9+, MSVC 2019+)
- **CMake**: 3.15 or later
- **Dependencies**:
  - ZeroMQ (libzmq) ≥4.3
  - cppzmq (C++ bindings)
  - Protocol Buffers ≥3.15
- **RAM**: 100MB minimum
- **Disk**: 50MB for build artifacts

---

## Installation

### macOS (Apple Silicon)

**Step 1: Install Homebrew** (if not already installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**Step 2: Install dependencies**
```bash
brew install cmake zeromq cppzmq protobuf
```

**Step 3: Verify installations**
```bash
cmake --version        # Should be ≥3.15
protoc --version       # Should be ≥3.15
pkg-config --modversion libzmq  # Should be ≥4.3
```

**Common issues**:
- If `cppzmq` not found: `brew install cppzmq`
- If Homebrew paths not in PATH: Add to `~/.zshrc`:
  ```bash
  export PATH="/opt/homebrew/bin:$PATH"
  ```

---

### macOS (Intel)

Same as Apple Silicon, but Homebrew installs to `/usr/local` instead of `/opt/homebrew`.

**Step 1-3**: Same as above

**Verify paths**:
```bash
ls /usr/local/include/zmq.hpp     # Should exist
ls /usr/local/lib/libzmq.a         # Should exist
```

---

### Windows

**Option 1: vcpkg (Recommended)**

**Step 1: Install Visual Studio 2019/2022** with C++ support
- Download from https://visualstudio.microsoft.com/
- Select "Desktop development with C++"

**Step 2: Install vcpkg**
```powershell
git clone https://github.com/Microsoft/vcpkg.git C:\vcpkg
cd C:\vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install
```

**Step 3: Install dependencies**
```powershell
.\vcpkg install zeromq:x64-windows cppzmq:x64-windows protobuf:x64-windows
```

This takes 10-15 minutes (compiles from source).

**Step 4: Install CMake**
- Download from https://cmake.org/download/
- Add to PATH during installation

**Verify**:
```powershell
cmake --version
```

**Option 2: Manual installation** (Advanced)

Download pre-built binaries:
1. ZeroMQ: https://github.com/zeromq/libzmq/releases
2. Protobuf: https://github.com/protocolbuffers/protobuf/releases
3. cppzmq: Header-only, copy `zmq.hpp` to include path

---

### Linux (Ubuntu/Debian)

**Step 1: Update package lists**
```bash
sudo apt update
```

**Step 2: Install dependencies**
```bash
sudo apt install -y \
    cmake \
    build-essential \
    libzmq3-dev \
    libprotobuf-dev \
    protobuf-compiler \
    pkg-config
```

**Step 3: Install cppzmq** (if not available via apt)
```bash
git clone https://github.com/zeromq/cppzmq.git
cd cppzmq
sudo cp zmq.hpp /usr/local/include/
```

**Verify**:
```bash
cmake --version
protoc --version
pkg-config --modversion libzmq
```

---

## Building

### macOS / Linux

```bash
cd cpp_execution

# Create build directory
mkdir build && cd build

# Configure (Release build for performance)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Compile
cmake --build .

# Output: ./execution_gateway
```

**Debug build** (for development):
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build .
```

**With tests** (requires Google Test):
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build .
ctest
```

---

### Windows

**Using vcpkg**:
```powershell
cd cpp_execution
mkdir build
cd build

# Configure (specify vcpkg toolchain)
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake

# Compile (Release build)
cmake --build . --config Release

# Output: .\Release\execution_gateway.exe
```

**Using Visual Studio GUI**:
1. Open `cpp_execution` folder in Visual Studio 2022
2. Visual Studio detects CMakeLists.txt automatically
3. Build → Build All (Ctrl+Shift+B)
4. Output: `out\build\x64-Release\execution_gateway.exe`

---

## Running

### Start the Gateway

**macOS / Linux**:
```bash
cd build
./execution_gateway
```

**Windows**:
```powershell
cd build\Release
.\execution_gateway.exe
```

**Expected output**:
```
========================================
C++ Execution Gateway
Version 1.0.0
========================================

[Main] Execution Gateway initialized
[Main] REP endpoint: tcp://*:5555
[Main] PUB endpoint: tcp://*:5556
[Main] Starting threads...
[Network] Thread started
[Network] Listening on tcp://*:5555
[Matching] Thread started
[Matching] Publishing on tcp://*:5556
[Main] Threads started. Press Ctrl+C to stop.
```

**Custom ports**:
```bash
./execution_gateway tcp://*:6000 tcp://*:6001
```

---

## Python Integration

### Install Python dependencies

```bash
pip install pyzmq protobuf
```

### Generate Python Protobuf code

```bash
cd cpp_execution
protoc --python_out=. proto/messages.proto
```

This creates `messages_pb2.py`.

### Example Python client

```python
import zmq
import time
from proto import messages_pb2 as pb

# Connect to C++ gateway
context = zmq.Context()

# REQ socket for orders
req_socket = context.socket(zmq.REQ)
req_socket.connect("tcp://localhost:5555")

# SUB socket for market data
sub_socket = context.socket(zmq.SUB)
sub_socket.connect("tcp://localhost:5556")
sub_socket.setsockopt_string(zmq.SUBSCRIBE, "AAPL")  # Subscribe to AAPL

# Submit order
order = pb.NewOrderRequest()
order.symbol = "AAPL"
order.side = pb.BUY
order.order_type = pb.LIMIT
order.price = int(150.50 * 10000)  # $150.50 → 1505000
order.quantity = 100
order.client_timestamp = int(time.time() * 1e6)

req_socket.send(order.SerializeToString())

# Receive response
response_bytes = req_socket.recv()
response = pb.OrderResponse()
response.ParseFromString(response_bytes)

print(f"Order ID: {response.order_id}")
print(f"Status: {pb.OrderStatus.Name(response.status)}")

# Subscribe to market data
while True:
    message = sub_socket.recv_string()
    symbol, data = message.split(" ", 1)

    market_data = pb.MarketDataUpdate()
    market_data.ParseFromString(data.encode('latin1'))

    print(f"[{symbol}] Bid: {market_data.best_bid_price / 10000:.2f} x {market_data.best_bid_quantity}")
    print(f"[{symbol}] Ask: {market_data.best_ask_price / 10000:.2f} x {market_data.best_ask_quantity}")
```

---

## Testing

### Manual testing with Python

**Test 1: Single order**
```python
# See example above
```

**Test 2: Matching orders**
```python
# Order 1: SELL 50 @ $150.00 (resting)
order1 = pb.NewOrderRequest()
order1.symbol = "AAPL"
order1.side = pb.SELL
order1.order_type = pb.LIMIT
order1.price = int(150.00 * 10000)
order1.quantity = 50
req_socket.send(order1.SerializeToString())
resp1 = req_socket.recv()

# Order 2: BUY 100 @ $150.00 (aggressive, should match 50)
order2 = pb.NewOrderRequest()
order2.symbol = "AAPL"
order2.side = pb.BUY
order2.order_type = pb.LIMIT
order2.price = int(150.00 * 10000)
order2.quantity = 100
req_socket.send(order2.SerializeToString())
resp2 = req_socket.recv()

# Expected: Order 2 fills 50, rests 50
```

### Unit tests (C++)

```bash
cd build
cmake .. -DBUILD_TESTS=ON
cmake --build .
ctest --output-on-failure
```

---

## Troubleshooting

### Common Build Errors

**Error**: `zmq.hpp not found`
- **macOS**: `brew install cppzmq`
- **Linux**: Copy `zmq.hpp` to `/usr/local/include/`
- **Windows**: Ensure vcpkg installed `cppzmq:x64-windows`

**Error**: `Could not find Protobuf`
- **macOS**: `brew install protobuf`
- **Linux**: `sudo apt install libprotobuf-dev protobuf-compiler`
- **Windows**: `vcpkg install protobuf:x64-windows`

**Error**: `undefined reference to zmq_*`
- **Linux**: Add `-lzmq` to linker flags
- **CMake**: Ensure `target_link_libraries(execution_lib PUBLIC ${ZeroMQ_LIBRARIES})`

**Error**: Apple Silicon CMake finds wrong architecture
```bash
cmake .. -DCMAKE_OSX_ARCHITECTURES=arm64
```

---

### Runtime Errors

**Error**: `Address already in use`
- Another process is using port 5555/5556
- **Solution**: Kill process or use different ports:
  ```bash
  ./execution_gateway tcp://*:6000 tcp://*:6001
  ```

**Error**: `Connection refused` (Python client)
- Gateway not running
- **Solution**: Start `./execution_gateway` first

**Error**: `Protobuf version mismatch`
- Python Protobuf version ≠ C++ Protobuf version
- **Solution**: Regenerate Python code:
  ```bash
  protoc --python_out=. proto/messages.proto
  ```

---

## Architecture

### Thread Model

```
┌─────────────────────────────────────────────────────────────────┐
│                       C++ Execution Gateway                     │
│                                                                 │
│  ┌──────────────────────────┐      ┌──────────────────────────┐│
│  │   Thread A: Network      │      │  Thread B: Matching      ││
│  │   - ZMQ REP socket       │      │  - Order books           ││
│  │   - Deserialize Protobuf │      │  - Price-time priority   ││
│  │   - Validate orders      │      │  - Publish market data   ││
│  └────────────┬─────────────┘      └─────────▲────────────────┘│
│               │                               │                 │
│               │   push()          try_pop()   │                 │
│               └──────────►  QUEUE  ───────────┘                 │
│                        (Thread-safe)                            │
└─────────────────────────────────────────────────────────────────┘
```

**Why two threads?**
- **Separation of concerns**: I/O vs business logic
- **Non-blocking network**: Thread A never blocks Thread B
- **Performance**: Parallel processing (network + matching)

**Why queue?**
- **Only synchronization point**: Minimal contention
- **Backpressure**: Queue full = reject orders

---

### Data Flow

1. Python sends `NewOrderRequest` via ZeroMQ REQ
2. Thread A receives, deserializes, validates
3. Thread A pushes to queue, sends `OrderResponse` (ACK)
4. Thread B pops from queue
5. Thread B submits to order book (matching)
6. If top-of-book changes, Thread B publishes `MarketDataUpdate` via ZeroMQ PUB
7. Python receives market data on SUB socket

---

## Performance Tuning

### Compiler optimizations

**Release build** (default):
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
```
Enables `-O3`, inlining, loop unrolling.

**Native CPU optimization** (not portable):
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native"
```
Uses AVX2, AVX-512 if available.

**Link-Time Optimization** (10-20% speedup):
```bash
cmake .. -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
```

---

### Lock-free queue

Current implementation uses **mutex-based queue** (300ns latency).

To enable **lock-free SPSC queue** (80ns latency):
1. Edit `include/lock_free_queue.hpp`
2. Uncomment `LockFreeSPSCQueue` implementation
3. Change type alias:
   ```cpp
   using OrderQueue = LockFreeSPSCQueue<execution_gateway::NewOrderRequest, 1024>;
   ```
4. Rebuild

**Trade-off**: More complex, harder to debug, requires careful testing.

---

### Profiling

**Linux (perf)**:
```bash
sudo perf record -g ./execution_gateway
sudo perf report
```

**macOS (Instruments)**:
```bash
instruments -t "Time Profiler" ./execution_gateway
```

**Windows (Visual Studio Profiler)**:
1. Debug → Start Performance Profiler
2. Select "CPU Usage"
3. Run executable

---

## License

MIT License - See LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue.

---

**Congratulations!** You now have a fully functional low-latency order matching engine. 🚀

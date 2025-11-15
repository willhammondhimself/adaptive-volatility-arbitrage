// C++ Execution Gateway - Main Server
//
// PURPOSE:
// Entry point for the execution gateway. Implements two-thread architecture:
// - Thread A (Network Server): Receives orders via ZeroMQ REP socket
// - Thread B (Matching Engine): Processes orders from queue, publishes market data
//
// DESIGN RATIONALE:
// - Two threads: Clean separation of I/O (Thread A) from business logic (Thread B)
// - Queue: Only synchronization point between threads (lock-free or mutex-based)
// - ZeroMQ: Language-agnostic IPC, low latency, reliable delivery
// - Protobuf: Compact serialization, type-safe, cross-language
//
// ARCHITECTURE:
//
//   Python Client                    C++ Gateway
//   ┌──────────────┐                ┌────────────────────────────────┐
//   │              │                │                                │
//   │ REQ socket   │──────────────► │ Thread A: Network Server      │
//   │              │    Order       │ (ZMQ REP socket)               │
//   └──────────────┘                │   │                            │
//         ▲                         │   │ Push to queue              │
//         │                         │   ▼                            │
//         │ ACK                     │ ┌────────────────────────────┐│
//         └─────────────────────────│ │   Thread-Safe Queue        ││
//                                   │ └────────────────────────────┘│
//                                   │   │                            │
//   ┌──────────────┐                │   │ Pop from queue             │
//   │              │                │   ▼                            │
//   │ SUB socket   │◄───────────────│ Thread B: Matching Engine     │
//   │              │  Market Data   │ (Order books, fills)           │
//   └──────────────┘                │   │                            │
//                                   │   │ Publish market data        │
//                                   │   ▼                            │
//                                   │ ZMQ PUB socket                 │
//                                   └────────────────────────────────┘
//
// SHUTDOWN:
// - Graceful: SIGINT/SIGTERM sets atomic flag, threads exit cleanly
// - Ungraceful: SIGKILL forces immediate termination (ZeroMQ cleans up)

#include <iostream>
#include <memory>
#include <thread>
#include <atomic>
#include <csignal>
#include <chrono>
#include <unordered_map>

#include <zmq.hpp>

#include "order_book.hpp"
#include "lock_free_queue.hpp"
#include "messages.pb.h"

using namespace execution_gateway;

//=============================================================================
// GLOBAL STATE (for signal handling)
//=============================================================================

// Shutdown flag: Set by signal handler, checked by threads
std::atomic<bool> g_shutdown{false};

// Signal handler: SIGINT (Ctrl+C), SIGTERM (kill)
void signal_handler(int signal) {
  std::cout << "\n[Main] Received signal " << signal << ", shutting down...\n";
  g_shutdown.store(true);
}

//=============================================================================
// EXECUTION GATEWAY CLASS
//=============================================================================

class ExecutionGateway {
 public:
  ExecutionGateway(const std::string& rep_endpoint, const std::string& pub_endpoint)
      : rep_endpoint_(rep_endpoint), pub_endpoint_(pub_endpoint) {
    // Initialize ZeroMQ context
    // Why context? Manages all sockets, threads, inproc communication
    // 1 context per process is standard
    zmq_ctx_ = std::make_unique<zmq::context_t>(1);  // 1 I/O thread

    std::cout << "[Main] Execution Gateway initialized\n";
    std::cout << "[Main] REP endpoint: " << rep_endpoint_ << "\n";
    std::cout << "[Main] PUB endpoint: " << pub_endpoint_ << "\n";
  }

  ~ExecutionGateway() {
    shutdown();
  }

  //-------------------------------------------------------------------------
  // RUN: Start threads and block until shutdown
  //-------------------------------------------------------------------------
  void run() {
    std::cout << "[Main] Starting threads...\n";

    // Start Thread A (Network Server)
    network_thread_ = std::thread(&ExecutionGateway::run_network_server, this);

    // Start Thread B (Matching Engine)
    matching_thread_ = std::thread(&ExecutionGateway::run_matching_engine, this);

    std::cout << "[Main] Threads started. Press Ctrl+C to stop.\n";

    // Wait for threads to finish (when g_shutdown becomes true)
    network_thread_.join();
    matching_thread_.join();

    std::cout << "[Main] Threads stopped. Exiting.\n";
  }

  //-------------------------------------------------------------------------
  // SHUTDOWN: Signal threads to stop
  //-------------------------------------------------------------------------
  void shutdown() {
    g_shutdown.store(true);
  }

 private:
  // Configuration
  std::string rep_endpoint_;              // ZeroMQ REP socket (e.g., "tcp://*:5555")
  std::string pub_endpoint_;              // ZeroMQ PUB socket (e.g., "tcp://*:5556")

  // ZeroMQ context
  std::unique_ptr<zmq::context_t> zmq_ctx_;

  // Thread-safe queue
  OrderQueue order_queue_;

  // Order books (one per symbol)
  std::unordered_map<std::string, std::unique_ptr<OrderBook>> order_books_;

  // Threads
  std::thread network_thread_;
  std::thread matching_thread_;

  //-------------------------------------------------------------------------
  // THREAD A: NETWORK SERVER
  //-------------------------------------------------------------------------
  // RESPONSIBILITIES:
  // 1. Receive NewOrderRequest via REP socket
  // 2. Deserialize Protobuf message
  // 3. Validate basic parameters
  // 4. Push to queue (non-blocking)
  // 5. Send OrderResponse (ACCEPTED/REJECTED)
  //
  // ERROR HANDLING:
  // - Protobuf parse errors → Send REJECTED response
  // - Queue full → Send REJECTED response (backpressure)
  // - ZeroMQ errors → Log and continue (don't crash)
  //
  // PERFORMANCE TARGET:
  // - <50μs per order (deserialize + validate + queue + ACK)
  void run_network_server() {
    try {
      std::cout << "[Network] Thread started\n";

      // Create REP socket
      // REP = Reply socket (pairs with Python's REQ socket)
      // Guarantees: Reliable delivery, request-reply pattern
      zmq::socket_t rep_socket(*zmq_ctx_, zmq::socket_type::rep);

      // Socket options
      // linger: Wait 1000ms for pending messages on close
      // rcvtimeo: 100ms receive timeout (allows shutdown check)
      rep_socket.set(zmq::sockopt::linger, 1000);
      rep_socket.set(zmq::sockopt::rcvtimeo, 100);

      // Bind to endpoint
      rep_socket.bind(rep_endpoint_);
      std::cout << "[Network] Listening on " << rep_endpoint_ << "\n";

      // Main loop: Receive orders until shutdown
      while (!g_shutdown.load()) {
        // Receive message (non-blocking with 100ms timeout)
        zmq::message_t request_msg;
        auto recv_result = rep_socket.recv(request_msg, zmq::recv_flags::none);

        if (!recv_result) {
          // Timeout or error (likely shutdown)
          continue;
        }

        // Deserialize Protobuf
        NewOrderRequest request;
        if (!request.ParseFromArray(request_msg.data(), request_msg.size())) {
          send_error_response(rep_socket, 0, "Invalid Protobuf message");
          continue;
        }

        // Validate request
        std::string error;
        if (!validate_request(request, error)) {
          send_error_response(rep_socket, 0, error);
          continue;
        }

        // Push to queue
        if (!order_queue_.push(request)) {
          send_error_response(rep_socket, 0, "Queue full (backpressure)");
          continue;
        }

        // Send ACK (order accepted into queue)
        // NOTE: This is NOT a fill report - just acknowledges receipt
        OrderResponse ack;
        ack.set_order_id(0);  // Will be assigned by matching engine
        ack.set_status(OrderStatus::ACCEPTED);
        ack.set_server_timestamp(get_timestamp_micros());

        send_response(rep_socket, ack);
      }

      std::cout << "[Network] Thread stopping\n";

    } catch (const std::exception& e) {
      std::cerr << "[Network] Fatal error: " << e.what() << "\n";
    }
  }

  //-------------------------------------------------------------------------
  // THREAD B: MATCHING ENGINE
  //-------------------------------------------------------------------------
  // RESPONSIBILITIES:
  // 1. Pop orders from queue (blocking wait_and_pop)
  // 2. Get/create order book for symbol
  // 3. Submit order to order book (matching logic)
  // 4. Check if top-of-book changed
  // 5. Publish MarketDataUpdate on PUB socket (if changed)
  //
  // ERROR HANDLING:
  // - Order validation errors → Log (already validated in Thread A)
  // - ZeroMQ errors → Log and continue
  //
  // PERFORMANCE TARGET:
  // - <10μs per order (matching + market data publish)
  void run_matching_engine() {
    try {
      std::cout << "[Matching] Thread started\n";

      // Create PUB socket
      // PUB = Publish socket (Python subscribes with SUB socket)
      // Guarantees: Best-effort delivery, fire-and-forget
      zmq::socket_t pub_socket(*zmq_ctx_, zmq::socket_type::pub);

      // Socket options
      // linger: Don't wait on close (fire-and-forget)
      // sndhwm: High-water mark (10k messages buffered)
      pub_socket.set(zmq::sockopt::linger, 0);
      pub_socket.set(zmq::sockopt::sndhwm, 10000);

      // Bind to endpoint
      pub_socket.bind(pub_endpoint_);
      std::cout << "[Matching] Publishing on " << pub_endpoint_ << "\n";

      // Main loop: Process orders until shutdown
      while (!g_shutdown.load()) {
        // Pop order from queue (with timeout to check shutdown)
        auto order_opt = order_queue_.try_pop();

        if (!order_opt) {
          // Queue empty, sleep briefly to avoid busy-wait
          std::this_thread::sleep_for(std::chrono::microseconds(100));
          continue;
        }

        const auto& request = *order_opt;

        // Get or create order book for symbol
        auto& order_book = get_or_create_order_book(request.symbol());

        // Store previous top-of-book
        auto prev_tob = order_book.get_top_of_book();

        // Submit order to matching engine
        auto match_result = order_book.submit_order(
            proto_to_side(request.side()),
            proto_to_order_type(request.order_type()),
            request.price(),
            request.quantity()
        );

        // Log fills (in production, send ExecutionReports here)
        if (!match_result.fills.empty()) {
          std::cout << "[Matching] Order " << match_result.order_id
                    << " filled " << match_result.total_filled()
                    << " @ avg " << match_result.avg_fill_price() << "\n";

          // TODO: Send ExecutionReport messages for each fill
          // For now, just log (production implementation would publish to separate socket)
        }

        // Check if top-of-book changed
        auto new_tob = order_book.get_top_of_book();
        if (new_tob != prev_tob) {
          publish_market_data(pub_socket, request.symbol(), new_tob);
        }
      }

      std::cout << "[Matching] Thread stopping\n";

    } catch (const std::exception& e) {
      std::cerr << "[Matching] Fatal error: " << e.what() << "\n";
    }
  }

  //-------------------------------------------------------------------------
  // HELPER: Get or create order book
  //-------------------------------------------------------------------------
  OrderBook& get_or_create_order_book(const std::string& symbol) {
    auto it = order_books_.find(symbol);
    if (it == order_books_.end()) {
      // Create new order book
      auto order_book = std::make_unique<OrderBook>(symbol);
      auto* ptr = order_book.get();
      order_books_[symbol] = std::move(order_book);
      std::cout << "[Matching] Created order book for " << symbol << "\n";
      return *ptr;
    }
    return *it->second;
  }

  //-------------------------------------------------------------------------
  // HELPER: Publish market data
  //-------------------------------------------------------------------------
  void publish_market_data(zmq::socket_t& socket,
                           const std::string& symbol,
                           const TopOfBook& tob) {
    MarketDataUpdate update;
    update.set_symbol(symbol);
    update.set_best_bid_price(tob.best_bid_price);
    update.set_best_bid_quantity(tob.best_bid_qty);
    update.set_best_ask_price(tob.best_ask_price);
    update.set_best_ask_quantity(tob.best_ask_qty);
    update.set_timestamp(get_timestamp_micros());

    // Serialize Protobuf
    std::string serialized;
    update.SerializeToString(&serialized);

    // Send on PUB socket (best-effort)
    // Topic: symbol (allows Python to subscribe to specific symbols)
    std::string topic = symbol + " ";  // Space separates topic from data
    std::string message = topic + serialized;

    zmq::message_t zmq_msg(message.size());
    memcpy(zmq_msg.data(), message.data(), message.size());
    socket.send(zmq_msg, zmq::send_flags::dontwait);

    std::cout << "[Matching] Published market data for " << symbol
              << " (bid: " << tob.best_bid_price
              << " x " << tob.best_bid_qty
              << ", ask: " << tob.best_ask_price
              << " x " << tob.best_ask_qty << ")\n";
  }

  //-------------------------------------------------------------------------
  // HELPER: Send error response
  //-------------------------------------------------------------------------
  void send_error_response(zmq::socket_t& socket, OrderID order_id,
                           const std::string& reason) {
    OrderResponse response;
    response.set_order_id(order_id);
    response.set_status(OrderStatus::REJECTED);
    response.set_reject_reason(reason);
    response.set_server_timestamp(get_timestamp_micros());

    send_response(socket, response);
  }

  //-------------------------------------------------------------------------
  // HELPER: Send response
  //-------------------------------------------------------------------------
  void send_response(zmq::socket_t& socket, const OrderResponse& response) {
    std::string serialized;
    response.SerializeToString(&serialized);

    zmq::message_t zmq_msg(serialized.size());
    memcpy(zmq_msg.data(), serialized.data(), serialized.size());

    socket.send(zmq_msg, zmq::send_flags::none);
  }

  //-------------------------------------------------------------------------
  // HELPER: Validate request
  //-------------------------------------------------------------------------
  bool validate_request(const NewOrderRequest& request, std::string& error) {
    if (request.symbol().empty()) {
      error = "Symbol cannot be empty";
      return false;
    }

    if (request.side() == Side::SIDE_UNSPECIFIED) {
      error = "Invalid side (must be BUY or SELL)";
      return false;
    }

    if (request.order_type() == OrderType::ORDER_TYPE_UNSPECIFIED) {
      error = "Invalid order type (must be LIMIT or MARKET)";
      return false;
    }

    if (request.quantity() <= 0) {
      error = "Quantity must be positive";
      return false;
    }

    if (request.order_type() == OrderType::LIMIT && request.price() <= 0) {
      error = "Limit order price must be positive";
      return false;
    }

    return true;
  }

  //-------------------------------------------------------------------------
  // HELPER: Get timestamp in microseconds
  //-------------------------------------------------------------------------
  int64_t get_timestamp_micros() const {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto duration = now.time_since_epoch();
    return duration_cast<microseconds>(duration).count();
  }
};

//=============================================================================
// MAIN ENTRY POINT
//=============================================================================

int main(int argc, char** argv) {
  // Print banner
  std::cout << "========================================\n";
  std::cout << "C++ Execution Gateway\n";
  std::cout << "Version 1.0.0\n";
  std::cout << "========================================\n\n";

  // Parse command-line arguments (optional: support custom ports)
  std::string rep_endpoint = "tcp://*:5555";
  std::string pub_endpoint = "tcp://*:5556";

  if (argc >= 3) {
    rep_endpoint = argv[1];
    pub_endpoint = argv[2];
  }

  // Install signal handlers
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  // Create and run gateway
  ExecutionGateway gateway(rep_endpoint, pub_endpoint);
  gateway.run();

  std::cout << "Gateway stopped. Goodbye!\n";
  return 0;
}

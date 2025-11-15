#pragma once

// C++ Execution Gateway - Order Book
//
// PURPOSE:
// Implements a central limit order book (CLOB) with price-time priority matching.
// This is the core of the matching engine where buy and sell orders meet.
//
// DESIGN RATIONALE:
// - Price-time priority: Standard exchange matching algorithm
//   1. Price priority: Better prices match first (highest bid, lowest ask)
//   2. Time priority: Among same price, earlier orders match first (FIFO)
//
// - Why std::map for price levels? Automatic sorting + O(log n) insert/find
// - Why std::deque for orders at each level? FIFO queue, O(1) push_back/pop_front
// - Alternative: Skip list or array-based for lower latency (more complex)
//
// MATCHING RULES:
// - LIMIT orders: Match if price crosses, else rest in book
// - MARKET orders: Match immediately at best available price or reject
// - Partial fills: One aggressive order can match multiple resting orders
//
// THREAD SAFETY:
// - NOT thread-safe (called only from Thread B - Matching Engine)
// - Why single-threaded? Simplifies logic, avoids lock contention
// - Thread A (network) uses queue to communicate with Thread B
//
// PERFORMANCE TARGETS:
// - submit_order(): <1μs for simple limit order
// - Multi-level fill: <5μs for 10-level sweep
// - get_top_of_book(): <50ns (cached, O(1))

#include <cstdint>
#include <map>
#include <deque>
#include <memory>
#include <string>
#include <vector>
#include <optional>

#include "messages.pb.h"

namespace execution_gateway {

//=============================================================================
// TYPE DEFINITIONS
//=============================================================================

// Fixed-point price (scaled by 10000)
// Example: $100.25 → 1002500
// Why int64? Matches Protobuf, avoids float precision issues
using Price = int64_t;
using Quantity = int32_t;
using OrderID = uint64_t;

// InternalSide: Buy or Sell
// Note: Protobuf also defines Side, so we use InternalSide to avoid conflicts
enum class InternalSide {
  BUY,
  SELL
};

// Internal order type: Limit or Market
// Note: Protobuf also defines OrderType, so we use InternalOrderType to avoid conflicts
enum class InternalOrderType {
  LIMIT,
  MARKET
};

//=============================================================================
// ORDER: Internal representation of a single order
//=============================================================================

struct Order {
  OrderID order_id;                 // Unique ID assigned by engine
  std::string symbol;               // e.g., "AAPL"
  InternalSide side;                // BUY or SELL
  InternalOrderType order_type;     // LIMIT or MARKET
  Price price;                      // Fixed-point (0 for MARKET)
  Quantity quantity;                // Original quantity
  Quantity filled_qty;              // Cumulative filled quantity
  int64_t timestamp;                // Unix microseconds (for time priority)
  bool is_active;                   // false if fully filled or cancelled

  // Convenience: remaining quantity
  Quantity remaining_qty() const {
    return quantity - filled_qty;
  }
};

//=============================================================================
// PRICE LEVEL: All orders at one price
//=============================================================================

struct PriceLevel {
  Price price;                                    // Price of this level
  std::deque<std::shared_ptr<Order>> orders;      // FIFO queue of orders
  Quantity total_quantity;                        // Sum of remaining qtys

  PriceLevel() : price(0), total_quantity(0) {}
  explicit PriceLevel(Price p) : price(p), total_quantity(0) {}

  // Add order to this level (back of queue = latest)
  void add_order(std::shared_ptr<Order> order);

  // Remove inactive orders from front of queue
  void cleanup_inactive_orders();

  // Check if level is empty
  bool is_empty() const;
};

//=============================================================================
// TOP-OF-BOOK: Best bid/ask snapshot
//=============================================================================

struct TopOfBook {
  Price best_bid_price = 0;         // 0 if no bids
  Quantity best_bid_qty = 0;
  Price best_ask_price = 0;         // 0 if no asks
  Quantity best_ask_qty = 0;

  // Equality check (for change detection)
  bool operator!=(const TopOfBook& other) const {
    return best_bid_price != other.best_bid_price ||
           best_bid_qty != other.best_bid_qty ||
           best_ask_price != other.best_ask_price ||
           best_ask_qty != other.best_ask_qty;
  }

  bool operator==(const TopOfBook& other) const {
    return !(*this != other);
  }
};

//=============================================================================
// FILL RECORD: Single fill event (for ExecutionReport)
//=============================================================================

struct Fill {
  OrderID order_id;                 // Aggressive order ID
  Quantity fill_quantity;           // Qty filled in this event
  Price fill_price;                 // Fixed-point price
  int32_t leaves_quantity;          // Unfilled qty after this fill
  int32_t cumulative_qty;           // Total filled so far
  int64_t timestamp;                // Unix microseconds

  // Is this the final fill?
  bool is_final() const {
    return leaves_quantity == 0;
  }
};

//=============================================================================
// MATCH RESULT: Result of order submission
//=============================================================================

struct MatchResult {
  OrderID order_id;                 // Assigned order ID
  bool accepted;                    // true if order accepted
  std::string reject_reason;        // Error message if rejected
  std::vector<Fill> fills;          // All fills (empty if no match)

  // Convenience: total filled quantity
  Quantity total_filled() const {
    Quantity total = 0;
    for (const auto& fill : fills) {
      total += fill.fill_quantity;
    }
    return total;
  }

  // Weighted average fill price
  Price avg_fill_price() const {
    if (fills.empty()) return 0;

    int64_t weighted_sum = 0;
    Quantity total_qty = 0;

    for (const auto& fill : fills) {
      weighted_sum += static_cast<int64_t>(fill.fill_quantity) * fill.fill_price;
      total_qty += fill.fill_quantity;
    }

    return (total_qty > 0) ? (weighted_sum / total_qty) : 0;
  }
};

//=============================================================================
// ORDER BOOK: Central limit order book for one symbol
//=============================================================================

class OrderBook {
 public:
  // Constructor: Initialize with symbol
  explicit OrderBook(std::string symbol);

  // Destructor
  ~OrderBook() = default;

  // Delete copy/move (order books are not copyable)
  OrderBook(const OrderBook&) = delete;
  OrderBook& operator=(const OrderBook&) = delete;

  //-------------------------------------------------------------------------
  // SUBMIT ORDER: Main entry point for order processing
  //-------------------------------------------------------------------------
  // Returns: MatchResult with order_id and fills
  // Side effects: Updates order book, generates fills
  //
  // ALGORITHM:
  // 1. Validate order (price > 0 for LIMIT, qty > 0, etc.)
  // 2. Assign unique order ID
  // 3. Match against contra side (price-time priority)
  // 4. If unfilled quantity remains:
  //    - LIMIT: Add to own side (resting order)
  //    - MARKET: Reject (no resting market orders)
  // 5. Return MatchResult with fills
  MatchResult submit_order(InternalSide side, InternalOrderType type, Price price, Quantity qty);

  //-------------------------------------------------------------------------
  // GET TOP-OF-BOOK: Current best bid/ask
  //-------------------------------------------------------------------------
  // Returns: TopOfBook snapshot
  // Performance: O(1) (cached)
  //
  // WHY CACHE?
  // - Called after every order submission to check for changes
  // - O(1) cached read vs O(log n) map lookup
  //
  // WHEN TO INVALIDATE CACHE?
  // - After any order submission (done internally)
  TopOfBook get_top_of_book() const;

  //-------------------------------------------------------------------------
  // DIAGNOSTICS: Get order book state
  //-------------------------------------------------------------------------
  size_t num_bid_levels() const { return bids_.size(); }
  size_t num_ask_levels() const { return asks_.size(); }
  std::string get_symbol() const { return symbol_; }

 private:
  std::string symbol_;              // Symbol (e.g., "AAPL")

  // Bids: Sorted descending (highest price first)
  // Why std::greater? Best bid = highest price = first element
  std::map<Price, PriceLevel, std::greater<Price>> bids_;

  // Asks: Sorted ascending (lowest price first)
  // Why std::less? Best ask = lowest price = first element
  std::map<Price, PriceLevel, std::less<Price>> asks_;

  // Atomic counter for order IDs
  OrderID next_order_id_ = 1;

  // Cached top-of-book (for change detection)
  mutable TopOfBook cached_tob_;
  mutable bool tob_valid_ = false;

  //-------------------------------------------------------------------------
  // INTERNAL MATCHING ALGORITHMS
  //-------------------------------------------------------------------------

  // Match market order (immediate or kill)
  MatchResult match_market_order(InternalSide side, Quantity qty);

  // Match limit order (match if crosses, else add to book)
  MatchResult match_limit_order(InternalSide side, Price price, Quantity qty);

  // Execute trade between aggressor and resting order
  void execute_trade(
      std::shared_ptr<Order> aggressor,
      std::shared_ptr<Order> resting,
      Quantity qty,
      std::vector<Fill>& fills);

  // Add resting order to book
  void add_resting_order(std::shared_ptr<Order> order);

  // Update cached top-of-book
  void update_top_of_book() const;

  // Validate order parameters
  bool validate_order(InternalSide side, InternalOrderType type, Price price, Quantity qty,
                      std::string& reject_reason) const;

  // Get next unique order ID
  OrderID get_next_order_id() {
    return next_order_id_++;
  }

  // Get current timestamp (Unix microseconds)
  int64_t get_timestamp_micros() const;

  // Get contra side map
  std::map<Price, PriceLevel, std::greater<Price>>& get_contra_map(InternalSide side);
  std::map<Price, PriceLevel, std::less<Price>>& get_own_map(InternalSide side);
};

//=============================================================================
// UTILITY FUNCTIONS
//=============================================================================

// Convert Protobuf Side to internal InternalSide
inline InternalSide proto_to_side(execution_gateway::Side proto_side) {
  return (proto_side == execution_gateway::Side::BUY) ? InternalSide::BUY : InternalSide::SELL;
}

// Convert Protobuf OrderType to internal InternalOrderType
inline InternalOrderType proto_to_order_type(execution_gateway::OrderType proto_type) {
  return (proto_type == execution_gateway::OrderType::LIMIT)
             ? InternalOrderType::LIMIT
             : InternalOrderType::MARKET;
}

// Convert internal InternalSide to Protobuf Side
inline execution_gateway::Side side_to_proto(InternalSide side) {
  return (side == InternalSide::BUY) ? execution_gateway::Side::BUY
                                      : execution_gateway::Side::SELL;
}

}  // namespace execution_gateway

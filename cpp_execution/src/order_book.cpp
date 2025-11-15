#include "order_book.hpp"

#include <chrono>
#include <algorithm>
#include <iostream>

namespace execution_gateway {

//=============================================================================
// PRICE LEVEL IMPLEMENTATION
//=============================================================================

void PriceLevel::add_order(std::shared_ptr<Order> order) {
  orders.push_back(order);
  total_quantity += order->remaining_qty();
}

void PriceLevel::cleanup_inactive_orders() {
  while (!orders.empty() && !orders.front()->is_active) {
    orders.pop_front();
  }
}

bool PriceLevel::is_empty() const {
  return orders.empty() || total_quantity == 0;
}

//=============================================================================
// ORDER BOOK IMPLEMENTATION
//=============================================================================

OrderBook::OrderBook(std::string symbol)
    : symbol_(std::move(symbol)) {}

//-----------------------------------------------------------------------------
// SUBMIT ORDER: Main entry point
//-----------------------------------------------------------------------------

MatchResult OrderBook::submit_order(InternalSide side, InternalOrderType type, Price price, Quantity qty) {
  MatchResult result;
  result.order_id = get_next_order_id();
  result.accepted = false;

  // Step 1: Validate order
  if (!validate_order(side, type, price, qty, result.reject_reason)) {
    return result;
  }

  result.accepted = true;

  // Step 2: Create order object
  auto order = std::make_shared<Order>();
  order->order_id = result.order_id;
  order->symbol = symbol_;
  order->side = side;
  order->order_type = type;
  order->price = price;
  order->quantity = qty;
  order->filled_qty = 0;
  order->timestamp = get_timestamp_micros();
  order->is_active = true;

  // Step 3: Match based on order type
  if (type == InternalOrderType::MARKET) {
    result = match_market_order(side, qty);
    result.order_id = order->order_id;
  } else {
    result = match_limit_order(side, price, qty);
    result.order_id = order->order_id;
  }

  // Step 4: Invalidate top-of-book cache (will be recalculated on next access)
  tob_valid_ = false;

  return result;
}

//-----------------------------------------------------------------------------
// MATCH MARKET ORDER: Immediate or kill
//-----------------------------------------------------------------------------

MatchResult OrderBook::match_market_order(InternalSide side, Quantity qty) {
  MatchResult result;
  result.accepted = true;

  Quantity remaining = qty;
  Quantity cumulative_filled = 0;

  // Handle BUY and SELL separately due to different map comparators
  if (side == InternalSide::BUY) {
    // BUY orders match against asks (lowest price first)
    auto it = asks_.begin();
    while (it != asks_.end() && remaining > 0) {
      PriceLevel& level = it->second;
      level.cleanup_inactive_orders();

      if (level.is_empty()) {
        it = asks_.erase(it);
        continue;
      }

      // Process orders at this price level (FIFO)
      while (!level.orders.empty() && remaining > 0) {
        auto resting_order = level.orders.front();

        if (!resting_order->is_active) {
          level.orders.pop_front();
          continue;
        }

        // Determine fill quantity
        Quantity fill_qty = std::min(remaining, resting_order->remaining_qty());

        // Execute trade
        resting_order->filled_qty += fill_qty;
        remaining -= fill_qty;
        cumulative_filled += fill_qty;
        level.total_quantity -= fill_qty;

        // Create fill record
        Fill fill;
        fill.order_id = result.order_id;
        fill.fill_quantity = fill_qty;
        fill.fill_price = resting_order->price;
        fill.leaves_quantity = remaining;
        fill.cumulative_qty = cumulative_filled;
        fill.timestamp = get_timestamp_micros();

        result.fills.push_back(fill);

        // Mark resting order as inactive if fully filled
        if (resting_order->remaining_qty() == 0) {
          resting_order->is_active = false;
          level.orders.pop_front();
        }
      }

      // If level empty, remove it
      if (level.is_empty()) {
        it = asks_.erase(it);
      } else {
        ++it;
      }
    }
  } else {
    // SELL orders match against bids (highest price first)
    auto it = bids_.begin();
    while (it != bids_.end() && remaining > 0) {
      PriceLevel& level = it->second;
      level.cleanup_inactive_orders();

      if (level.is_empty()) {
        it = bids_.erase(it);
        continue;
      }

      // Process orders at this price level (FIFO)
      while (!level.orders.empty() && remaining > 0) {
        auto resting_order = level.orders.front();

        if (!resting_order->is_active) {
          level.orders.pop_front();
          continue;
        }

        // Determine fill quantity
        Quantity fill_qty = std::min(remaining, resting_order->remaining_qty());

        // Execute trade
        resting_order->filled_qty += fill_qty;
        remaining -= fill_qty;
        cumulative_filled += fill_qty;
        level.total_quantity -= fill_qty;

        // Create fill record
        Fill fill;
        fill.order_id = result.order_id;
        fill.fill_quantity = fill_qty;
        fill.fill_price = resting_order->price;
        fill.leaves_quantity = remaining;
        fill.cumulative_qty = cumulative_filled;
        fill.timestamp = get_timestamp_micros();

        result.fills.push_back(fill);

        // Mark resting order as inactive if fully filled
        if (resting_order->remaining_qty() == 0) {
          resting_order->is_active = false;
          level.orders.pop_front();
        }
      }

      // If level empty, remove it
      if (level.is_empty()) {
        it = bids_.erase(it);
      } else {
        ++it;
      }
    }
  }

  // Market orders don't rest - if unfilled, it's a partial fill
  // (In production, you might reject fully unfilled market orders)
  return result;
}

//-----------------------------------------------------------------------------
// MATCH LIMIT ORDER: Match if crosses, else add to book
//-----------------------------------------------------------------------------

MatchResult OrderBook::match_limit_order(InternalSide side, Price price, Quantity qty) {
  MatchResult result;
  result.accepted = true;

  Quantity remaining = qty;
  Quantity cumulative_filled = 0;

  // Handle BUY and SELL separately due to different map comparators
  if (side == InternalSide::BUY) {
    // BUY orders match against asks (lowest price first)
    // BUY crosses if bid_price >= best_ask_price
    auto it = asks_.begin();
    while (it != asks_.end() && remaining > 0) {
      PriceLevel& level = it->second;
      Price contra_price = level.price;

      // Check if price crosses
      if (price < contra_price) {
        break;  // No more matches possible
      }

      level.cleanup_inactive_orders();

      if (level.is_empty()) {
        it = asks_.erase(it);
        continue;
      }

      // Process orders at this price level (FIFO)
      while (!level.orders.empty() && remaining > 0) {
        auto resting_order = level.orders.front();

        if (!resting_order->is_active) {
          level.orders.pop_front();
          continue;
        }

        // Determine fill quantity
        Quantity fill_qty = std::min(remaining, resting_order->remaining_qty());

        // Execute trade at resting order's price (price-time priority)
        resting_order->filled_qty += fill_qty;
        remaining -= fill_qty;
        cumulative_filled += fill_qty;
        level.total_quantity -= fill_qty;

        // Create fill record
        Fill fill;
        fill.order_id = result.order_id;
        fill.fill_quantity = fill_qty;
        fill.fill_price = resting_order->price;
        fill.leaves_quantity = remaining;
        fill.cumulative_qty = cumulative_filled;
        fill.timestamp = get_timestamp_micros();

        result.fills.push_back(fill);

        // Mark resting order as inactive if fully filled
        if (resting_order->remaining_qty() == 0) {
          resting_order->is_active = false;
          level.orders.pop_front();
        }
      }

      // If level empty, remove it
      if (level.is_empty()) {
        it = asks_.erase(it);
      } else {
        ++it;
      }
    }
  } else {
    // SELL orders match against bids (highest price first)
    // SELL crosses if ask_price <= best_bid_price
    auto it = bids_.begin();
    while (it != bids_.end() && remaining > 0) {
      PriceLevel& level = it->second;
      Price contra_price = level.price;

      // Check if price crosses
      if (price > contra_price) {
        break;  // No more matches possible
      }

      level.cleanup_inactive_orders();

      if (level.is_empty()) {
        it = bids_.erase(it);
        continue;
      }

      // Process orders at this price level (FIFO)
      while (!level.orders.empty() && remaining > 0) {
        auto resting_order = level.orders.front();

        if (!resting_order->is_active) {
          level.orders.pop_front();
          continue;
        }

        // Determine fill quantity
        Quantity fill_qty = std::min(remaining, resting_order->remaining_qty());

        // Execute trade at resting order's price (price-time priority)
        resting_order->filled_qty += fill_qty;
        remaining -= fill_qty;
        cumulative_filled += fill_qty;
        level.total_quantity -= fill_qty;

        // Create fill record
        Fill fill;
        fill.order_id = result.order_id;
        fill.fill_quantity = fill_qty;
        fill.fill_price = resting_order->price;
        fill.leaves_quantity = remaining;
        fill.cumulative_qty = cumulative_filled;
        fill.timestamp = get_timestamp_micros();

        result.fills.push_back(fill);

        // Mark resting order as inactive if fully filled
        if (resting_order->remaining_qty() == 0) {
          resting_order->is_active = false;
          level.orders.pop_front();
        }
      }

      // If level empty, remove it
      if (level.is_empty()) {
        it = bids_.erase(it);
      } else {
        ++it;
      }
    }
  }

  // If unfilled quantity remains, add as resting order
  if (remaining > 0) {
    auto resting_order = std::make_shared<Order>();
    resting_order->order_id = result.order_id;
    resting_order->symbol = symbol_;
    resting_order->side = side;
    resting_order->order_type = InternalOrderType::LIMIT;
    resting_order->price = price;
    resting_order->quantity = qty;
    resting_order->filled_qty = cumulative_filled;
    resting_order->timestamp = get_timestamp_micros();
    resting_order->is_active = true;

    add_resting_order(resting_order);
  }

  return result;
}

//-----------------------------------------------------------------------------
// ADD RESTING ORDER: Insert order into book
//-----------------------------------------------------------------------------

void OrderBook::add_resting_order(std::shared_ptr<Order> order) {
  if (order->side == InternalSide::BUY) {
    // Add to bids
    auto it = bids_.find(order->price);
    if (it == bids_.end()) {
      // Create new price level
      PriceLevel level(order->price);
      level.add_order(order);
      bids_[order->price] = level;
    } else {
      // Add to existing level
      it->second.add_order(order);
    }
  } else {
    // Add to asks
    auto it = asks_.find(order->price);
    if (it == asks_.end()) {
      // Create new price level
      PriceLevel level(order->price);
      level.add_order(order);
      asks_[order->price] = level;
    } else {
      // Add to existing level
      it->second.add_order(order);
    }
  }
}

//-----------------------------------------------------------------------------
// GET TOP-OF-BOOK: Current best bid/ask
//-----------------------------------------------------------------------------

TopOfBook OrderBook::get_top_of_book() const {
  if (tob_valid_) {
    return cached_tob_;
  }

  update_top_of_book();
  return cached_tob_;
}

void OrderBook::update_top_of_book() const {
  TopOfBook tob;

  // Best bid: First element of bids_ (highest price)
  if (!bids_.empty()) {
    const auto& level = bids_.begin()->second;
    tob.best_bid_price = level.price;
    tob.best_bid_qty = level.total_quantity;
  }

  // Best ask: First element of asks_ (lowest price)
  if (!asks_.empty()) {
    const auto& level = asks_.begin()->second;
    tob.best_ask_price = level.price;
    tob.best_ask_qty = level.total_quantity;
  }

  cached_tob_ = tob;
  tob_valid_ = true;
}

//-----------------------------------------------------------------------------
// VALIDATE ORDER: Check order parameters
//-----------------------------------------------------------------------------

bool OrderBook::validate_order(
    InternalSide side,
    InternalOrderType type,
    Price price,
    Quantity qty,
    std::string& reject_reason) const {

  // Check quantity
  if (qty <= 0) {
    reject_reason = "Quantity must be positive";
    return false;
  }

  if (qty > 1'000'000) {
    reject_reason = "Quantity exceeds maximum (1,000,000)";
    return false;
  }

  // Check price for LIMIT orders
  if (type == InternalOrderType::LIMIT) {
    if (price <= 0) {
      reject_reason = "Limit order price must be positive";
      return false;
    }

    // Sanity check: price not absurdly high/low
    if (price > 100'000'000'000) {  // $10M per share
      reject_reason = "Price exceeds maximum";
      return false;
    }
  }

  return true;
}

//-----------------------------------------------------------------------------
// GET TIMESTAMP: Unix microseconds
//-----------------------------------------------------------------------------

int64_t OrderBook::get_timestamp_micros() const {
  using namespace std::chrono;
  auto now = system_clock::now();
  auto duration = now.time_since_epoch();
  return duration_cast<microseconds>(duration).count();
}

}  // namespace execution_gateway

# Currency Emergence Simulation

Agent-based model simulating which good spontaneously emerges as **currency** from a pool of tradeable goods.

## Hypothesis

Goods with high **durability** (entropy resistance), sufficient trade flow, and sustained demand through **generational turnover** converge to become the dominant medium of exchange.

**Key metric:** Cumulative Liquidity = Σ(volume[t] × durability^(now − t))

## How It Works

| Component | Role |
|---|---|
| **Goods** | Five goods (gold → grain) differing in durability; other properties held equal |
| **Agents** | Utility-maximising agents with heterogeneous preferences, diminishing marginal utility, and EMA-based memory of each good's marketability |
| **Marketplace** | Decentralised double auction matching buy/sell offers |
| **Generational turnover** | Agents die and pass inventory to new agents with different preferences; durable goods survive inheritance better |
| **Metrics** | Cumulative liquidity (durability-weighted trade volume) tracks which good wins |

## Key Results

| Turnover Rate | Emergent Currency | Why |
|---|---|---|
| 0% | Silver | Gold saturates inventory without turnover to reset demand |
| 1–5% | **Gold** | Gold's superior durability compounds across generations |

## Quick Start

```bash
pip install numpy

# Default run (5% turnover, 5000 ticks)
python currency_emergence.py

# No turnover — silver wins
python currency_emergence.py 0.0

# Compare across turnover rates
python currency_emergence.py compare
```

## Project Structure

```
currency_emergence.py   # Full simulation (single file)
README.md
LICENSE
```

## Architecture

```
GoodProperties → Good (decay, replenishment)
TradingMemory  → Agent (utility, offers, learning)
Marketplace    → match & execute trades
MetricsCollector → cumulative liquidity, currency detection
SimulationEngine → orchestrates tick loop
```

### Agent Decision Model

Each agent evaluates goods by combining:

1. **Direct utility** — preference × direct_utility / (1 + α × quantity) — diminishing returns
2. **Indirect exchange value** — marketability × durability discount × (1 + S2F premium + stability premium)

The indirect component grows as agents accumulate trading experience (maturity factor), modelling the gradual discovery of money.

## Configuration

All parameters are set via `SimConfig`:

| Parameter | Default | Description |
|---|---|---|
| `num_agents` | 30 | Number of trading agents |
| `num_ticks` | 5000 | Simulation length |
| `turnover_rate` | 0.0 | Per-tick probability of agent death/replacement |
| `inheritance_rate` | 0.5 | Fraction of inventory passed to next generation |
| `memory_decay` | 0.98 | EMA decay for marketability learning |
| `diminishing_rate` | 0.15 | Utility diminishing rate |
| `indirect_weight` | 0.5 | Weight of indirect exchange value |

## Theoretical Background

This simulation formalises ideas from Carl Menger's theory of the origin of money and extends them with:

- **Entropy resistance** as the primary driver of monetary emergence (durability as decay factor)
- **Stock-to-flow** ratio as a learned scarcity signal
- **Generational turnover** as the mechanism that reveals durability's long-term advantage
- **Cumulative liquidity** as a measurable proxy for "moneyness"

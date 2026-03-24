#!/usr/bin/env python3
"""
Currency Emergence Simulation — Agent-Based Model.

Simulates which good emerges as "currency" from a liquidity pool.

Hypothesis: Goods with high durability (entropy resistance), sufficient trade
flow, and sustained demand through generational turnover converge to currency.

Key metric: Cumulative Liquidity = Σ(volume[t] × durability^(now − t))
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

# ── Good Definition ──────────────────────────────────────────


@dataclass
class GoodProperties:
    """Good attributes. All in [0, 1] except direct_utility."""
    durability: float       # Entropy resistance. 1.0=indestructible
    marginal_cost: float    # Higher = scarcer = costly to produce
    divisibility: float     # 1.0=infinitely divisible
    transport_cost: float   # Higher = harder to move
    recognition: float      # 1.0=universally known
    direct_utility: float = 1.0  # Non-monetary demand (ornament etc.)

    def __post_init__(self):
        for name in ("durability", "marginal_cost", "divisibility",
                     "transport_cost", "recognition"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0,1], got {v}")


@dataclass
class Good:
    """A tradeable good type. Agents hold quantities of each Good."""
    name: str
    properties: GoodProperties

    def decay_factor(self) -> float:
        """Per-tick survival rate (nonlinear model).

        High-durability goods decay very slowly:
          gold  (dur=0.995): 0.995^0.005 ≈ 0.999975
          grain (dur=0.3):   0.3^0.7     ≈ 0.383
        """
        d = self.properties.durability
        return d ** (1.0 - d)

    def replenish_amount(self, base_rate: float, current_stock: float) -> float:
        """Supply replenishment based on marginal cost and stock (S/F model)."""
        mc = max(self.properties.marginal_cost, 0.01)
        stock_factor = 1.0 + current_stock / 100.0
        return base_rate / (mc * stock_factor)


# ── Agent Memory & Learning ──────────────────────────────────


@dataclass
class TradingMemory:
    """EMA-based memory of each good's marketability."""
    marketability: dict[str, float] = field(default_factory=dict)
    observed_s2f: dict[str, float] = field(default_factory=dict)
    value_stability: dict[str, float] = field(default_factory=dict)

    def record_trade(self, good: str, success: bool, decay: float = 0.95):
        prev = self.marketability.get(good, 0.0)
        obs = 1.0 if success else 0.0
        self.marketability[good] = decay * prev + (1.0 - decay) * obs

    def record_s2f(self, good: str, s2f: float, decay: float = 0.98):
        prev = self.observed_s2f.get(good, 0.0)
        self.observed_s2f[good] = decay * prev + (1.0 - decay) * s2f

    def record_retention(self, good: str, retention: float, decay: float = 0.98):
        prev = self.value_stability.get(good, 0.5)
        self.value_stability[good] = decay * prev + (1.0 - decay) * retention

    def get_marketability(self, good: str) -> float:
        return self.marketability.get(good, 0.0)

    def get_s2f_score(self, good: str) -> float:
        """Convert S2F ratio to 0-1 score (logistic function)."""
        s2f = self.observed_s2f.get(good, 1.0)
        return 1.0 / (1.0 + math.exp(-0.1 * (s2f - 10)))


# ── Agent ─────────────────────────────────────────────────────


@dataclass
class TradeOffer:
    agent_id: int
    sell_good: str
    sell_qty: float
    buy_good: str
    min_buy_qty: float
    expires_tick: int = 0


class Agent:
    """Utility-maximizing agent.

    Trade decisions combine direct utility (consumption) and indirect
    exchange value (resaleability).
    """

    def __init__(
        self,
        agent_id: int,
        good_names: list[str],
        preferences: np.ndarray,
        location: tuple[float, float],
        diminishing_rate: float = 0.1,
        indirect_weight: float = 0.5,
        memory_decay: float = 0.95,
    ):
        self.id = agent_id
        self.good_names = good_names
        self.preferences = dict(zip(good_names, preferences))
        self.inventory: dict[str, float] = {n: 0.0 for n in good_names}
        self.location = location
        self.diminishing_rate = diminishing_rate
        self.indirect_weight = indirect_weight
        self.memory_decay = memory_decay
        self.memory = TradingMemory()
        self.experience_ticks = 0

    def marginal_utility(self, good: str, registry: dict[str, Good]) -> float:
        """Diminishing marginal utility: pref × direct_utility / (1 + α × qty)"""
        pref = self.preferences.get(good, 0.0)
        qty = self.inventory.get(good, 0.0)
        du = registry[good].properties.direct_utility if good in registry else 1.0
        return pref * du / (1.0 + self.diminishing_rate * qty)

    def indirect_exchange_value(self, good: str, registry: dict[str, Good]) -> float:
        """Indirect exchange value combining three factors:
        1. marketability × durability discount (short-term)
        2. S2F premium (long-term, maturity-weighted)
        3. Value stability premium (long-term)
        """
        mk = self.memory.get_marketability(good)
        dur = registry[good].properties.durability if good in registry else 0.5
        dur_discount = dur ** 5  # Assumes 5-tick holding period
        maturity = 1.0 - math.exp(-self.experience_ticks / 1000.0)
        s2f_premium = self.memory.get_s2f_score(good) * 0.5 * maturity
        stability = self.memory.value_stability.get(good, 0.5) * 0.3 * maturity
        base = mk * dur_discount
        return base * (1.0 + s2f_premium + stability) * self.indirect_weight

    def total_value(self, good: str, registry: dict[str, Good]) -> float:
        return (self.marginal_utility(good, registry)
                + self.indirect_exchange_value(good, registry))

    def evaluate_trade(
        self, sell: str, sell_qty: float,
        buy: str, buy_qty: float,
        registry: dict[str, Good],
    ) -> float:
        """Net utility gain of a trade. Positive = accept."""
        loss = self.total_value(sell, registry) * sell_qty
        gain = self.total_value(buy, registry) * buy_qty
        bp, sp = registry[buy].properties, registry[sell].properties
        cost = ((bp.transport_cost + sp.transport_cost) * 0.1
                + (1.0 - bp.recognition) * 0.05)
        return gain - loss - cost

    def generate_offers(
        self, registry: dict[str, Good], tick: int,
        expiry: int = 5, max_offers: int = 3,
        rng: np.random.Generator | None = None,
    ) -> list[TradeOffer]:
        """Generate trade offers based on inventory and utility."""
        candidates = []
        for sell in self.good_names:
            sq = self.inventory.get(sell, 0.0)
            if sq < 0.1:
                continue
            for buy in self.good_names:
                if buy == sell:
                    continue
                sv = self.total_value(sell, registry)
                bv = self.total_value(buy, registry)
                if bv <= sv * 0.2:
                    continue
                ratio = sv / max(bv, 1e-9)
                frac = rng.uniform(0.2, 0.5) if rng else 0.3
                offer_qty = sq * frac
                candidates.append((bv - sv, sell, offer_qty, buy, offer_qty * ratio))
        candidates.sort(key=lambda x: -x[0])
        return [
            TradeOffer(self.id, s, sq, b, mbq, tick + expiry)
            for _, s, sq, b, mbq in candidates[:max_offers]
        ]

    def consume(self, registry: dict[str, Good], rate: float = 0.05):
        """Preference-weighted consumption to maintain trade incentives."""
        for name in self.good_names:
            qty = self.inventory.get(name, 0.0)
            if qty <= 0:
                continue
            pref = self.preferences.get(name, 0.0)
            du = registry[name].properties.direct_utility if name in registry else 1.0
            self.inventory[name] = max(0.0, qty - qty * pref * rate * du)

    def apply_decay(self, registry: dict[str, Good]):
        """Entropy-driven inventory decay."""
        for name, good in registry.items():
            if name in self.inventory:
                self.inventory[name] *= good.decay_factor()
                if self.inventory[name] < 1e-6:
                    self.inventory[name] = 0.0


# ── Trade Matching ────────────────────────────────────────────


@dataclass
class CompletedTrade:
    tick: int
    buyer_id: int
    seller_id: int
    good_a: str
    qty_a: float
    good_b: str
    qty_b: float


class Marketplace:
    """Decentralised double auction."""

    def __init__(self):
        self.open_offers: list[TradeOffer] = []

    def match_and_execute(
        self, agents: dict[int, Agent],
        registry: dict[str, Good], tick: int,
    ) -> list[CompletedTrade]:
        trades: list[CompletedTrade] = []
        matched: set[int] = set()

        # Group offers by (sell, buy) pair
        groups: dict[tuple[str, str], list[tuple[int, TradeOffer]]] = defaultdict(list)
        for i, o in enumerate(self.open_offers):
            groups[(o.sell_good, o.buy_good)].append((i, o))

        done_pairs: set[tuple[str, str]] = set()
        for (sa, ba), offers_a in groups.items():
            rev = (ba, sa)
            if rev in done_pairs:
                continue
            done_pairs.add((sa, ba))
            offers_b = groups.get(rev, [])
            if not offers_b:
                continue

            sort_key = lambda x: x[1].sell_qty / max(x[1].min_buy_qty, 1e-9)
            a_sorted = sorted(
                ((i, o) for i, o in offers_a if i not in matched),
                key=sort_key, reverse=True,
            )
            b_sorted = sorted(
                ((i, o) for i, o in offers_b if i not in matched),
                key=sort_key, reverse=True,
            )

            for ia, oa in a_sorted:
                if ia in matched:
                    continue
                for ib, ob in b_sorted:
                    if ib in matched or oa.agent_id == ob.agent_id:
                        continue
                    qa, qb = oa.sell_qty, ob.sell_qty
                    if qa < ob.min_buy_qty * 0.5 or qb < oa.min_buy_qty * 0.5:
                        continue
                    aa, ab = agents[oa.agent_id], agents[ob.agent_id]
                    if aa.inventory.get(sa, 0) < qa or ab.inventory.get(ba, 0) < qb:
                        continue
                    if (aa.evaluate_trade(sa, qa, ba, qb, registry) > 0
                            and ab.evaluate_trade(ba, qb, sa, qa, registry) > 0):
                        aa.inventory[sa] -= qa
                        aa.inventory[ba] += qb
                        ab.inventory[ba] -= qb
                        ab.inventory[sa] += qa
                        trades.append(CompletedTrade(
                            tick, oa.agent_id, ob.agent_id, sa, qa, ba, qb,
                        ))
                        matched.update([ia, ib])
                        break

        self.open_offers = [
            o for i, o in enumerate(self.open_offers) if i not in matched
        ]
        return trades


# ── Metrics — Cumulative Liquidity ────────────────────────────


@dataclass
class TickMetrics:
    tick: int
    trade_volume: dict[str, float] = field(default_factory=dict)
    num_trades: dict[str, int] = field(default_factory=dict)
    total_supply: dict[str, float] = field(default_factory=dict)
    stock_to_flow: dict[str, float] = field(default_factory=dict)
    cumulative_liquidity: dict[str, float] = field(default_factory=dict)
    liquidity_score: dict[str, float] = field(default_factory=dict)
    total_trades: int = 0


class MetricsCollector:
    def __init__(self, good_names: list[str]):
        self.good_names = good_names
        self.history: list[TickMetrics] = []
        self._cum_liq: dict[str, float] = {n: 0.0 for n in good_names}

    def record(
        self, tick: int, trades: list[CompletedTrade],
        agents: list[Agent], registry: dict[str, Good],
        flow_in: dict[str, float],
    ) -> TickMetrics:
        m = TickMetrics(tick=tick)

        vol: dict[str, float] = defaultdict(float)
        cnt: dict[str, int] = defaultdict(int)
        for t in trades:
            vol[t.good_a] += t.qty_a
            vol[t.good_b] += t.qty_b
            cnt[t.good_a] += 1
            cnt[t.good_b] += 1

        supply: dict[str, float] = defaultdict(float)
        for a in agents:
            for n, q in a.inventory.items():
                supply[n] += q

        for n in self.good_names:
            m.trade_volume[n] = vol.get(n, 0.0)
            m.num_trades[n] = cnt.get(n, 0)
            m.total_supply[n] = supply.get(n, 0.0)
            fi = flow_in.get(n, 0.0)
            m.stock_to_flow[n] = (
                supply.get(n, 0.0) / max(fi, 1e-9) if fi > 0.01 else 999.0
            )
            # Cumulative liquidity: past volumes decayed by durability
            decay = registry[n].decay_factor() if n in registry else 0.95
            self._cum_liq[n] = self._cum_liq[n] * decay + vol.get(n, 0.0)
            m.cumulative_liquidity[n] = self._cum_liq[n]

        m.total_trades = len(trades)

        # Normalised liquidity score
        max_cl = max(m.cumulative_liquidity.values(), default=1e-9) or 1e-9
        for n in self.good_names:
            m.liquidity_score[n] = m.cumulative_liquidity[n] / max_cl

        self.history.append(m)
        return m

    def detect_currency(self, window: int = 50) -> str | None:
        """Detect the good with highest avg liquidity score over *window* ticks."""
        if len(self.history) < window:
            return None
        recent = self.history[-window:]
        avg: dict[str, float] = defaultdict(float)
        for m in recent:
            for n in self.good_names:
                avg[n] += m.liquidity_score.get(n, 0.0) / window
        if not avg:
            return None
        best = max(avg, key=lambda k: avg[k])
        vals = sorted(avg.values(), reverse=True)
        if len(vals) > 1 and vals[0] > vals[1] * 1.5:
            return best
        return None


# ── Simulation Engine ─────────────────────────────────────────


@dataclass
class SimConfig:
    goods: dict[str, Good]
    num_agents: int = 30
    num_ticks: int = 5000
    seed: int = 42
    # Agent params
    dirichlet_alpha: float = 2.0
    endowment_scale: float = 100.0
    diminishing_rate: float = 0.15
    indirect_weight: float = 0.5
    memory_decay: float = 0.98
    consumption_rate: float = 0.03
    # Market params
    offer_expiry: int = 10
    max_offers: int = 5
    # Replenishment
    replenish_threshold: float = 0.5
    replenish_base_rate: float = 20.0
    # Generational turnover
    turnover_rate: float = 0.0
    inheritance_rate: float = 0.5


class SimulationEngine:
    def __init__(self, config: SimConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self.registry = config.goods
        self.names = list(config.goods.keys())
        self.market = Marketplace()
        self.metrics = MetricsCollector(self.names)
        self.tick = 0
        self.agents = self._create_agents()
        self.agents_dict = {a.id: a for a in self.agents}
        self.initial_supply = {
            n: sum(a.inventory.get(n, 0.0) for a in self.agents)
            for n in self.names
        }

    def _create_agents(self) -> list[Agent]:
        agents = []
        for i in range(self.cfg.num_agents):
            prefs = self.rng.dirichlet([self.cfg.dirichlet_alpha] * len(self.names))
            loc = (self.rng.uniform(0, 100), self.rng.uniform(0, 100))
            a = Agent(
                i, self.names, prefs, loc,
                self.cfg.diminishing_rate, self.cfg.indirect_weight,
                self.cfg.memory_decay,
            )
            for n in self.names:
                mc = self.registry[n].properties.marginal_cost
                base = self.cfg.endowment_scale / len(self.names)
                chance = max(0.15, 1.0 - mc)
                if self.rng.random() < chance:
                    a.inventory[n] = self.rng.exponential(base / chance)
                else:
                    a.inventory[n] = self.rng.exponential(base * 0.05)
            agents.append(a)
        return agents

    def step(self):
        """Execute one tick: decay → turnover → consume → replenish → trade → learn."""
        pre_supply = {
            n: sum(a.inventory.get(n, 0.0) for a in self.agents)
            for n in self.names
        }
        if self.cfg.turnover_rate > 0:
            self._turnover()
        for a in self.agents:
            a.consume(self.registry, self.cfg.consumption_rate)
            a.apply_decay(self.registry)

        flow_in = self._replenish()
        self._broadcast_info(pre_supply, flow_in)
        for a in self.agents:
            a.experience_ticks = self.tick

        for a in self.agents:
            offers = a.generate_offers(
                self.registry, self.tick,
                self.cfg.offer_expiry, self.cfg.max_offers, self.rng,
            )
            self.market.open_offers.extend(offers)
        self.market.open_offers = [
            o for o in self.market.open_offers if o.expires_tick > self.tick
        ]
        trades = self.market.match_and_execute(
            self.agents_dict, self.registry, self.tick,
        )
        self._learn(trades)
        self.metrics.record(self.tick, trades, self.agents, self.registry, flow_in)
        self.tick += 1

    def run(self, num_ticks: int | None = None, progress: bool = False) -> MetricsCollector:
        ticks = num_ticks or self.cfg.num_ticks
        for t in range(ticks):
            self.step()
            if progress and (t + 1) % 500 == 0:
                cur = self.metrics.detect_currency()
                print(f"  tick {t + 1}/{ticks} | currency: {cur or '---'}")
        return self.metrics

    # ── Internal helpers ──

    def _turnover(self):
        """Generational turnover: death → inheritance → new agent.

        Inheritance survival = inheritance_rate × durability.
        New agents have different preferences, creating fresh trade motives.
        """
        for i, agent in enumerate(self.agents):
            if self.rng.random() >= self.cfg.turnover_rate:
                continue
            inherited = {}
            for n in self.names:
                qty = agent.inventory.get(n, 0.0)
                if qty > 0:
                    dur = self.registry[n].properties.durability
                    inherited[n] = qty * self.cfg.inheritance_rate * dur

            prefs = self.rng.dirichlet([self.cfg.dirichlet_alpha] * len(self.names))
            loc = (self.rng.uniform(0, 100), self.rng.uniform(0, 100))
            new = Agent(
                agent.id, self.names, prefs, loc,
                self.cfg.diminishing_rate, self.cfg.indirect_weight,
                self.cfg.memory_decay,
            )
            for n, q in inherited.items():
                new.inventory[n] = q
            for n in self.names:
                mk = agent.memory.get_marketability(n)
                if mk > 0:
                    new.memory.marketability[n] = mk * 0.5
            self.agents[i] = new
            self.agents_dict[new.id] = new

    def _replenish(self) -> dict[str, float]:
        """Replenish goods whose supply falls below threshold."""
        flow: dict[str, float] = {n: 0.0 for n in self.names}
        for n in self.names:
            current = sum(a.inventory.get(n, 0.0) for a in self.agents)
            threshold = self.initial_supply[n] * self.cfg.replenish_threshold
            if current < threshold:
                amount = self.registry[n].replenish_amount(
                    self.cfg.replenish_base_rate, current,
                )
                flow[n] = amount
                per = amount / len(self.agents)
                for a in self.agents:
                    a.inventory[n] = a.inventory.get(n, 0.0) + per
        return flow

    def _broadcast_info(self, pre_supply: dict[str, float], flow_in: dict[str, float]):
        """Propagate S2F ratio and value retention to all agents."""
        for n in self.names:
            current = sum(a.inventory.get(n, 0.0) for a in self.agents)
            fi = flow_in.get(n, 0.0)
            s2f = min(current / max(fi, 0.01), 999.0) if fi > 0.01 else 999.0
            pre = pre_supply.get(n, 0.0)
            post = current - flow_in.get(n, 0.0)
            retention = (
                min(max(post / max(pre, 1e-9), 0.0), 1.0) if pre > 1e-9 else 1.0
            )
            for a in self.agents:
                a.memory.record_s2f(n, s2f, self.cfg.memory_decay)
                a.memory.record_retention(n, retention, self.cfg.memory_decay)

    def _learn(self, trades: list[CompletedTrade]):
        """Learn marketability from direct experience + social learning."""
        traded_by: dict[int, set[str]] = defaultdict(set)
        for t in trades:
            traded_by[t.buyer_id].update([t.good_a, t.good_b])
            traded_by[t.seller_id].update([t.good_a, t.good_b])
        for a in self.agents:
            for n in traded_by.get(a.id, set()):
                a.memory.record_trade(n, True, self.cfg.memory_decay)
        # Social learning: info about the most-traded good spreads weakly
        if trades:
            counts: dict[str, int] = defaultdict(int)
            for t in trades:
                counts[t.good_a] += 1
                counts[t.good_b] += 1
            top = max(counts, key=lambda k: counts[k])
            for a in self.agents:
                a.memory.record_trade(top, True, 0.998)


# ── Experiments ───────────────────────────────────────────────


def make_goods(specs: dict[str, dict]) -> dict[str, Good]:
    return {name: Good(name, GoodProperties(**props)) for name, props in specs.items()}


_DEFAULT_SPECS = {
    "gold":   dict(durability=0.995, marginal_cost=0.15, divisibility=0.8,
                   transport_cost=0.2, recognition=0.8),
    "silver": dict(durability=0.95,  marginal_cost=0.15, divisibility=0.8,
                   transport_cost=0.2, recognition=0.8),
    "copper": dict(durability=0.80,  marginal_cost=0.15, divisibility=0.8,
                   transport_cost=0.2, recognition=0.8),
    "salt":   dict(durability=0.50,  marginal_cost=0.15, divisibility=0.8,
                   transport_cost=0.2, recognition=0.8),
    "grain":  dict(durability=0.30,  marginal_cost=0.15, divisibility=0.8,
                   transport_cost=0.2, recognition=0.8),
}


def experiment_pure_durability(
    num_ticks: int = 5000, turnover: float = 0.0, seed: int = 42,
) -> MetricsCollector:
    """Five goods differing only in durability.

    turnover=0.0  → silver wins (inventory saturation)
    turnover=0.05 → gold wins (generational turnover effect)
    """
    goods = make_goods(_DEFAULT_SPECS)
    cfg = SimConfig(goods=goods, num_ticks=num_ticks, seed=seed, turnover_rate=turnover)
    engine = SimulationEngine(cfg)
    return engine.run(progress=True)


def experiment_generational_comparison(seeds: int = 5):
    """Compare currency emergence across turnover rates."""
    print("=" * 70)
    print("EXPERIMENT: Generational Turnover Effect on Currency Emergence")
    print("=" * 70)

    for rate in [0.0, 0.01, 0.03, 0.05]:
        wins = 0
        cum_liq_totals: dict[str, float] = defaultdict(float)
        for s in range(seeds):
            goods = make_goods(_DEFAULT_SPECS)
            cfg = SimConfig(
                goods=goods, num_ticks=5000, seed=s + 100, turnover_rate=rate,
            )
            engine = SimulationEngine(cfg)
            m = engine.run()
            if m.detect_currency() == "gold":
                wins += 1
            if m.history:
                last = m.history[-1]
                for n in last.cumulative_liquidity:
                    cum_liq_totals[n] += last.cumulative_liquidity[n]

        print(f"\n--- Turnover rate: {rate * 100:.0f}% ---")
        print(f"  Gold wins: {wins}/{seeds}")
        print(f"  Avg cumulative liquidity:")
        for n in sorted(cum_liq_totals, key=lambda k: -cum_liq_totals[k]):
            print(f"    {n}: {cum_liq_totals[n] / seeds:.1f}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        experiment_generational_comparison()
    else:
        turnover = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
        print(f"Running with turnover_rate={turnover}")
        m = experiment_pure_durability(num_ticks=5000, turnover=turnover)
        currency = m.detect_currency()
        print(f"\nEmergent currency: {currency or 'None'}")
        if m.history:
            last = m.history[-1]
            print("\nFinal cumulative liquidity:")
            for n in sorted(last.cumulative_liquidity,
                            key=lambda k: -last.cumulative_liquidity[k]):
                print(f"  {n}: {last.cumulative_liquidity[n]:.1f}")

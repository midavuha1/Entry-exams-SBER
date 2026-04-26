"""
Для выполнения задания сделал граф переходов(цепь Маркова 1-го порядка) и как и сказано
посчитал P(j | i) — вероятность того, что сразу после i будет просмотрен j. Но мне не понравилось
два момента: то что по данной формуле очень легко сделать вероятность 1 при малом количестве наблюдений
поэтому попытался сделать хоть какой-то баланс, то есть при подсчете вероятности
начал учитывать глобальную популярность товара. И второй момент это необходимость
правильно ранжировать когда у нас p = 0.9 по 7 и 0.7 по 900, для этого использую нижнюю границу
доверительного интервала Уилсона, который эффективен при малой выборке.

Данные не мешал, так как можно было случайно обучать граф уже на будущем.

Результат хита10 0.524, когда по бейзлайн 0.384

Программа печатает основную информацию в консоли + строит 4 графика: распределение длин сессий,
топ-25 популярных товаров, тепловая карта переходов между топ-25 товарами, товары по частоте появлений.

также в файле с логами лежит конкретная по каждой сессии, что было выведено, получилось ли угадать.
"""

import json
import math
import collections
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_sessions(path):
    sessions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                sessions.append(json.loads(line))
    return sessions


def train_test_split(sessions):
    train_sessions = [session[:-1] for session in sessions]
    test_targets   = [session[-1]  for session in sessions]
    return train_sessions, test_targets


def build_transition_graph(train_sessions):
    transitions = collections.defaultdict(collections.Counter)
    item_counts = collections.Counter()
    for session in train_sessions:
        for idx in range(len(session) - 1):
            src = session[idx]
            dst = session[idx + 1]
            transitions[src][dst] += 1
            item_counts[src] += 1
    return dict(transitions), item_counts


def wilson_lower_bound(p_hat, n, z=1.96):
    if n == 0:
        return 0.0
    denominator = 1 + z**2 / n
    center      = p_hat + z**2 / (2 * n)
    spread      = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return (center - spread) / denominator


def recommend(last_item, transitions, train_item_counts, global_freq, top_popular, lam=10.0, k=10):
    if last_item not in transitions:
        return [item for item in top_popular if item != last_item][:k]

    neighbor_counts = transitions[last_item]
    c_i             = train_item_counts[last_item]
    candidates      = set(neighbor_counts.keys())

    scored = []
    for j, c_ij in neighbor_counts.items():
        p_j     = global_freq.get(j, 0)
        p_bayes = (c_ij + lam * p_j) / (c_i + lam)
        score   = wilson_lower_bound(p_bayes, n=c_i)
        scored.append((j, score))

    scored.sort(key=lambda x: -x[1])
    result = [item for item, _ in scored if item != last_item]

    if len(result) < k:
        for item in top_popular:
            if item not in candidates and item != last_item:
                result.append(item)
            if len(result) >= k:
                break

    return result[:k]


def hit_at_k(recommendations, true_items, k=10):
    assert len(recommendations) == len(true_items)
    hits = sum(1 for recs, true_item in zip(recommendations, true_items) if true_item in recs[:k])
    return hits / len(true_items)


def write_log(path, train_sessions, model_recs, test_targets, model_hit, baseline_hit):
    lines = []
    lines.append(f"{'№':<6}  {'история':<35}  {'цель':>5}  {'топ-10':<50}  {'позиция':>7}  результат")
    lines.append("")

    for idx, (session, recs, true_item) in enumerate(zip(train_sessions, model_recs, test_targets)):
        history  = session[-6:] if len(session) > 6 else session
        position = str(recs.index(true_item) + 1) if true_item in recs[:10] else "-"
        result   = "hit" if true_item in recs[:10] else "miss"
        lines.append(
            f"{idx:<6}  {str(history):<35}  {true_item:>5}  {str(recs):<50}  {position:>7}  {result}"
        )

    hits_n = sum(1 for r, t in zip(model_recs, test_targets) if t in r[:10])
    lines.append("")
    lines.append(f"сессий     {len(train_sessions)}")
    lines.append(f"hit        {hits_n}  ({model_hit*100:.2f}%)")
    lines.append(f"miss       {len(train_sessions) - hits_n}  ({(1-model_hit)*100:.2f}%)")
    lines.append(f"hit@10     {model_hit:.4f}")
    lines.append(f"бейзлайн   {baseline_hit:.4f}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


sessions = load_sessions("sessions.jsonl")

all_items       = [item for session in sessions for item in session]
unique_items    = set(all_items)
session_lengths = [len(s) for s in sessions]
item_counts     = collections.Counter(all_items)

print(f"сессий:              {len(sessions)}")
print(f"уникальных товаров:  {len(unique_items)}")
print(f"взаимодействий:      {len(all_items)}")
print(f"средняя длина:       {sum(session_lengths)/len(session_lengths):.2f}")
print(f"медиана длины:       {sorted(session_lengths)[len(session_lengths)//2]}")
print(f"мин / макс:          {min(session_lengths)} / {max(session_lengths)}")

min_freq     = min(item_counts.values())
min_freq_n   = sum(1 for c in item_counts.values() if c == min_freq)
with_repeats = sum(1 for s in sessions if len(s) != len(set(s)))
print(f"мин. частота товара: {min_freq}  (таких товаров: {min_freq_n})")
print(f"сессий с повторами:  {with_repeats}  ({100*with_repeats/len(sessions):.1f}%)")

train_sessions, test_targets   = train_test_split(sessions)
transitions, train_item_counts = build_transition_graph(train_sessions)

all_train_items = [item for s in train_sessions for item in s]
global_counts   = collections.Counter(all_train_items)
total_train     = len(all_train_items)
global_freq     = {item: count / total_train for item, count in global_counts.items()}
top_popular     = [item for item, _ in global_counts.most_common()]

N        = 25
top_n    = [item for item, _ in item_counts.most_common(N)]
heatmap  = [[0.0] * N for _ in range(N)]
for i, src in enumerate(top_n):
    c_i = train_item_counts[src]
    if c_i == 0:
        continue
    for j, dst in enumerate(top_n):
        c_ij = transitions.get(src, {}).get(dst, 0)
        heatmap[i][j] = c_ij / c_i

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(session_lengths, bins=30, color="#4C72B0", edgecolor="white", linewidth=0.5)
ax1.axvline(sum(session_lengths)/len(session_lengths), color="red", linestyle="--",
            label=f"среднее = {sum(session_lengths)/len(session_lengths):.1f}")
ax1.set_title("распределение длин сессий")
ax1.set_xlabel("длина сессии")
ax1.set_ylabel("количество")
ax1.legend()

ax2 = fig.add_subplot(gs[0, 1])
top25  = item_counts.most_common(25)
labels = [str(x[0]) for x in top25]
values = [x[1] for x in top25]
ax2.barh(labels[::-1], values[::-1], color="#55A868")
ax2.set_title("топ-25 популярных товаров")
ax2.set_xlabel("появлений")

ax3 = fig.add_subplot(gs[1, 0])
buckets = collections.Counter()
for c in item_counts.values():
    if   c == 1:  buckets["1"]    += 1
    elif c <= 5:  buckets["2–5"]  += 1
    elif c <= 20: buckets["6–20"] += 1
    else:         buckets[">20"]  += 1
keys = ["1", "2–5", "6–20", ">20"]
ax3.bar(keys, [buckets[k] for k in keys], color="#8172B2")
ax3.set_title("товары по частоте появлений")
ax3.set_xlabel("частота")
ax3.set_ylabel("товаров")

ax4  = fig.add_subplot(gs[1, 1])
im   = ax4.imshow(heatmap, aspect="auto", cmap="Blues")
ticks = [str(x) for x in top_n]
ax4.set_xticks(range(N))
ax4.set_xticklabels(ticks, rotation=45, ha="right", fontsize=8)
ax4.set_yticks(range(N))
ax4.set_yticklabels(ticks, fontsize=8)
ax4.set_title(f"P(j|i) — топ-{N} товаров")
ax4.set_xlabel("следующий товар j")
ax4.set_ylabel("текущий товар i")
plt.colorbar(im, ax=ax4)

plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
plt.show()

model_recs = [
    recommend(session[-1], transitions, train_item_counts, global_freq, top_popular)
    for session in train_sessions
]

model_hit    = hit_at_k(model_recs, test_targets)
baseline_hit = hit_at_k([top_popular[:10]] * len(test_targets), test_targets)

print(f"\nhit@10 модель:    {model_hit:.4f}  ({model_hit*100:.2f}%)")
print(f"hit@10 бейзлайн:  {baseline_hit:.4f}  ({baseline_hit*100:.2f}%)")

write_log("predictions_log.txt", train_sessions, model_recs, test_targets, model_hit, baseline_hit)
print("лог сохранён → predictions_log.txt")

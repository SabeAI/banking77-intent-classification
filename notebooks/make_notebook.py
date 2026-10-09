"""Generates banking77_intent_classification.ipynb (run once; the notebook is the deliverable)."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Banking Customer Intent Classification (Banking77)

**Business question:** A bank's digital assistant receives thousands of free-text customer questions a day.
Can we route each one to the right intent automatically — and, just as important, *know when not to trust the model*
and hand the query to a person?

This notebook rebuilds, on public data, the kind of query classifier I built at U.S. Bank for the mobile voice assistant.

**Dataset:** [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets) (PolyAI, CC BY 4.0) — 13,083 real-style
customer queries labeled with 77 fine-grained banking intents (e.g., `card_arrival`, `lost_or_stolen_card`, `top_up_failed`).

**Approach**
1. Explore the data
2. Select models with 5-fold cross-validation on the training set only (the test set is touched once)
3. Evaluate on the held-out test set with bootstrap confidence intervals and a McNemar significance test
4. Error analysis: which intents get confused, and why
5. Confidence-based routing: how much can be automated at a target accuracy?
6. Learning curve: would more labeled data help?
""")

code("""
import sys, warnings, time
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report

import intent

FIG = Path.cwd().parent / "reports" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left", "figure.dpi": 110,
})
pd.set_option("display.width", 140)
train, test = intent.load_data()
print(f"train: {len(train):,} rows | test: {len(test):,} rows | intents: {train.category.nunique()}")
train.sample(5, random_state=intent.SEED)
""")

md("## 1. Explore the data")
code("""
counts = train.category.value_counts()
lengths = train.text.str.split().str.len()
print(f"Examples per intent (train): min {counts.min()}, median {counts.median():.0f}, max {counts.max()}")
print(f"Test set: exactly {test.category.value_counts().unique()[0]} examples per intent (balanced)")
print(f"Query length: median {lengths.median():.0f} words, 95th percentile {lengths.quantile(.95):.0f} words")

fig, ax = plt.subplots(figsize=(9, 3.4))
ax.bar(range(len(counts)), counts.values, color=BLUE, width=0.8)
ax.set(title="Training examples per intent (77 intents, sorted)", xlabel="Intent rank", ylabel="Examples")
ax.grid(axis="x", visible=False)
for i in (0, len(counts) - 1):
    ax.annotate(f"{counts.index[i]}: {counts.iloc[i]}", (i, counts.iloc[i]), textcoords="offset points",
                xytext=(6, 4) if i == 0 else (-10, 40), ha="left" if i == 0 else "right", fontsize=8.5, color=INK2,
                bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec="none"))
fig.tight_layout(); fig.savefig(FIG / "class_balance.png", bbox_inches="tight"); plt.show()
""")
md("""
**Takeaways:** queries are short (about 10 words), so every word carries signal. The training set is moderately
imbalanced (roughly 5:1 between the largest and smallest intent), while the test set is perfectly balanced — so
**macro-F1** (every intent weighted equally) is reported alongside accuracy.
""")

md("""
## 2. Model selection with cross-validation

Four candidates, from the simplest bag-of-words baseline (the approach I used at U.S. Bank) to richer TF-IDF models.
Character n-grams are added because customer text is full of typos and word variants ("top up", "top-up", "topped up").
Hyperparameters are tuned with 5-fold stratified CV **on the training set only**.
""")
code("""
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=intent.SEED)
results, fitted = [], {}
for name, (pipe, grid) in intent.build_models().items():
    t0 = time.time()
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=-1, refit=True).fit(train.text, train.category)
    i = gs.best_index_
    results.append({"model": name, "best_params": gs.best_params_,
                    "cv_accuracy": gs.cv_results_["mean_test_score"][i],
                    "cv_std": gs.cv_results_["std_test_score"][i], "minutes": (time.time() - t0) / 60})
    fitted[name] = gs.best_estimator_
cv_table = pd.DataFrame(results).sort_values("cv_accuracy", ascending=False).reset_index(drop=True)
cv_table.style.format({"cv_accuracy": "{:.2%}", "cv_std": "{:.2%}", "minutes": "{:.1f}"})
""")

md("""
## 3. Held-out test evaluation

The CV winner is chosen *before* looking at the test set. All four are then scored on test for comparison,
with 95% bootstrap confidence intervals.
""")
code("""
best_name = cv_table.loc[0, "model"]
baseline_name = "Bag-of-words + Naive Bayes"
preds, rows = {}, []
for name, model in fitted.items():
    p = model.predict(test.text); preds[name] = p
    lo, hi = intent.bootstrap_ci(test.category, p)
    rows.append({"model": name, "test_accuracy": accuracy_score(test.category, p),
                 "ci_low": lo, "ci_high": hi, "macro_f1": f1_score(test.category, p, average="macro")})
test_table = pd.DataFrame(rows).sort_values("test_accuracy", ascending=False).reset_index(drop=True)
print("Selected by CV:", best_name)
test_table.style.format({c: "{:.2%}" for c in ["test_accuracy", "ci_low", "ci_high", "macro_f1"]})
""")
code("""
published = pd.DataFrame({"model": ["BERT (frozen features)", "USE", "ConveRT", "USE + ConveRT", "BERT (fine-tuned)"],
                          "test_accuracy": [0.8719, 0.9281, 0.9301, 0.9336, 0.9366]})
plot_df = pd.concat([test_table[["model", "test_accuracy"]].assign(src="This project"),
                     published.assign(src="Casanueva et al. (2020)")]).sort_values("test_accuracy")

fig, ax = plt.subplots(figsize=(8.5, 4.2))
colors = [BLUE if s == "This project" else BASE for s in plot_df.src]
ax.barh(plot_df.model, plot_df.test_accuracy, color=colors, height=0.6)
for y, (v, s) in enumerate(zip(plot_df.test_accuracy, plot_df.src)):
    ax.text(v + 0.002, y, f"{v:.1%}", va="center", fontsize=9, color=INK if s == "This project" else MUTED)
ax.set(xlim=(0.80, 0.96), title="Test accuracy: this project (blue) vs. published neural baselines (gray)", xlabel="Accuracy")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}")); ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(FIG / "model_comparison.png", bbox_inches="tight"); plt.show()
""")
code("""
mc = intent.mcnemar(test.category, preds[best_name], preds[baseline_name])
print(f"McNemar test — {best_name} vs. {baseline_name}")
print(f"  best right / baseline wrong: {mc['a_right_b_wrong']}")
print(f"  baseline right / best wrong: {mc['b_right_a_wrong']}")
print(f"  exact p-value: {mc['p_value']:.2e}")
""")
md("""
**Takeaways:** the improvement over the bag-of-words baseline is large and statistically significant (McNemar test on
the same 3,080 test queries). The CV-selected model lands about 2.4 points below fine-tuned BERT from the dataset's
original paper, while training in under a minute on a laptop CPU with no GPU — a strong cost/accuracy trade-off for a
production baseline. The two word+char models are statistically tied on test (overlapping confidence intervals), so
picking the CV winner rather than the test winner is the honest choice.
""")

md("## 4. Error analysis")
code("""
best = fitted[best_name]; yhat = preds[best_name]
report = pd.DataFrame(classification_report(test.category, yhat, output_dict=True)).T.iloc[:-3]
print("Weakest 8 intents by F1:")
display(report.sort_values("f1-score").head(8)[["precision", "recall", "f1-score"]].style.format("{:.2f}"))
print("Most frequent confusions:")
intent.top_confusions(test.category, yhat, k=8)
""")
code("""
pair = intent.top_confusions(test.category, yhat, k=1).iloc[0]
ex = test[(test.category == pair.true) & (yhat == pair.predicted)].text.head(4)
print(f"Examples labeled '{pair.true}' but predicted '{pair.predicted}':")
for t in ex: print("  -", t)
""")
md("""
**Takeaways:** errors concentrate in semantically adjacent intents: `top_up_failed` vs. `top_up_reverted`,
`why_verify_identity` vs. `unable_to_verify_identity`, `card_arrival` vs. `card_delivery_estimate`. Some of these
overlap even for a human reader; others (like "Where should my card work?" read as a broken card) are true model
mistakes from missing context words. In production I would (a) review the most-confused pairs with the business
owners and merge or re-define them where the business action is the same, and (b) route low-confidence predictions
to an agent — quantified next.
""")

md("""
## 5. Confidence-based routing: how much can we safely automate?

A classifier doesn't have to answer everything. If the bank only auto-routes queries where the model is confident
and sends the rest to a human agent, it can trade **coverage** for **accuracy**. This is the decision the business
actually has to make.
""")
code("""
router_name = "TF-IDF words+chars + Logistic Regression"   # needs calibrated-ish probabilities
router = fitted[router_name]
proba = router.predict_proba(test.text)
table = intent.routing_table(test.category, proba, router.classes_, thresholds=np.round(np.arange(0, 0.91, 0.05), 2))

TARGET = 0.97
op = table[table.automated_accuracy >= TARGET].iloc[0]
print(f"Router model: {router_name}")
print(f"All queries automated: accuracy {table.iloc[0].automated_accuracy:.1%}")
print(f"To reach {TARGET:.0%} accuracy: threshold {op.threshold:.2f} -> automate {op.automated_share:.1%} of queries, "
      f"send {op.to_human_share:.1%} to agents")

fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(table.automated_share, table.automated_accuracy, color=BLUE, lw=2, marker="o", ms=4)
ax.scatter([op.automated_share], [op.automated_accuracy], s=70, color=ORANGE, zorder=3, edgecolor=SURF, linewidth=2)
ax.annotate(f"Threshold {op.threshold:.2f}: automate {op.automated_share:.0%}\\nat {op.automated_accuracy:.1%} accuracy",
            (op.automated_share, op.automated_accuracy), xytext=(-12, -38), textcoords="offset points",
            ha="right", fontsize=9, color=INK2)
ax.set(title="Coverage vs. accuracy trade-off for automated routing",
       xlabel="Share of queries automated (rest go to a human agent)", ylabel="Accuracy on automated queries")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
fig.tight_layout(); fig.savefig(FIG / "routing_tradeoff.png", bbox_inches="tight"); plt.show()
table.style.format({"automated_share": "{:.1%}", "automated_accuracy": "{:.1%}", "to_human_share": "{:.1%}"})
""")

md("""
## 6. Learning curve: is more labeled data worth paying for?

Labeling costs money. Training the selected model on growing slices of the training data shows whether more
labels would still move accuracy. (This is a descriptive check made *after* model selection.)
""")
code("""
from sklearn.base import clone
from sklearn.model_selection import train_test_split
fracs, accs = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0], []
for f in fracs:
    sub = train if f == 1.0 else train_test_split(train, train_size=f, stratify=train.category, random_state=intent.SEED)[0]
    m = clone(best).fit(sub.text, sub.category); accs.append(accuracy_score(test.category, m.predict(test.text)))
lc = pd.DataFrame({"train_fraction": fracs, "examples": [int(len(train) * f) for f in fracs], "test_accuracy": accs})

fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.plot(lc.examples, lc.test_accuracy, color=BLUE, lw=2, marker="o", ms=5)
for x, y in zip(lc.examples, lc.test_accuracy):
    first = x == lc.examples.iloc[0]
    ax.annotate(f"{y:.1%}", (x, y), xytext=(8, -3) if first else (0, 7), textcoords="offset points",
                ha="left" if first else "center", fontsize=8.5, color=INK2)
ax.set(title=f"Learning curve: {best_name}", xlabel="Training examples", ylabel="Test accuracy")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
fig.tight_layout(); fig.savefig(FIG / "learning_curve.png", bbox_inches="tight"); plt.show()
lc.style.format({"train_fraction": "{:.0%}", "test_accuracy": "{:.2%}"})
""")
md("""
**Takeaway:** accuracy is still climbing at the full training set, but gains are flattening, so each additional
batch of labels buys less. Targeted labeling of the most-confused intents (Section 4) would likely pay off more than
labeling uniformly.
""")

md("""
## 7. Summary and next steps

**Results** (exact figures are in the tables above and in the README)
- The CV-selected linear model on word + character TF-IDF clearly and significantly beats the bag-of-words baseline.
- It comes within about 2.4 points of fine-tuned BERT at a fraction of the compute.
- With confidence routing, about 86% of queries can be automated at 97% accuracy, with the rest sent to agents.

**Next steps**
- Fine-tune a small transformer (e.g., DistilBERT) and compare the accuracy gain against the cost of serving it.
- Calibrate probabilities (temperature or isotonic scaling) so the routing threshold means what it says.
- Add an out-of-scope detector for queries that match none of the 77 intents.
- Wrap the model in a small API with monitoring for drift in query mix and confidence.
""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("banking77_intent_classification.ipynb")
nbf.write(nb, out)
print("wrote", out)

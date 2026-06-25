import sweepData from "@/public/data/sweep.json";
import rougeSpace from "@/public/data/rouge_space.json";
import rougeHasos from "@/public/data/rouge_hasos.json";
import { METHOD_IDS, METHOD_META, type MethodId } from "@/lib/space";
import type { Dataset, MethodSweep, SweepData, SweepPoint } from "@/lib/sweep";

type RougeBlob = Record<
  MethodId,
  {
    by_split?: {
      all?: {
        MACRO?: {
          rouge1?: number;
          rouge2?: number;
          rougeL?: number;
        };
      };
    };
  }
>;

const sweep = sweepData as SweepData;
const rougeByDataset: Record<Dataset, RougeBlob> = {
  space: rougeSpace as RougeBlob,
  hasos: rougeHasos as RougeBlob,
};

const DATASETS: Dataset[] = ["space", "hasos"];

function fmt(value: number | null | undefined, digits = 5) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return value.toFixed(digits);
}

function pct(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${Math.round(value * 100)}%`;
}

function labelValue(value: number | string | null | undefined) {
  return value === null || value === undefined ? "-" : String(value);
}

function methodLabel(method: MethodId) {
  return METHOD_META[method]?.short ?? method.toUpperCase();
}

function methodName(method: MethodId) {
  return METHOD_META[method]?.label ?? method.toUpperCase();
}

function thresholdSweep(dataset: Dataset, method: MethodId): MethodSweep | null {
  return sweep.datasets?.[dataset]?.threshold?.methods?.[method] ?? null;
}

function tokenSweep(dataset: Dataset, method: MethodId): MethodSweep | null {
  return sweep.datasets?.[dataset]?.tokabs?.methods?.[method] ?? null;
}

function bestPoint(methodSweep: MethodSweep | null): SweepPoint | null {
  return methodSweep?.points.find((point) => point.is_best) ?? null;
}

function defaultPoint(methodSweep: MethodSweep | null): SweepPoint | null {
  return methodSweep?.points.find((point) => point.is_default) ?? null;
}

function baselineMacro(dataset: Dataset, method: MethodId) {
  return rougeByDataset[dataset]?.[method]?.by_split?.all?.MACRO ?? null;
}

function bestBaseline(dataset: Dataset) {
  let winner: MethodId = "m1";
  let best = -Infinity;
  for (const method of METHOD_IDS) {
    const value = baselineMacro(dataset, method)?.rouge1;
    if (typeof value === "number" && value > best) {
      best = value;
      winner = method;
    }
  }
  return { method: winner, rouge1: best };
}

function recommendedPoint(methodSweep: MethodSweep | null): SweepPoint | null {
  if (!methodSweep) return null;
  if (
    methodSweep.verdict?.status === "switch" ||
    methodSweep.verdict?.status === "default_optimal"
  ) {
    return bestPoint(methodSweep);
  }
  return defaultPoint(methodSweep) ?? bestPoint(methodSweep);
}

function FigureBar({
  point,
  max,
  active,
}: {
  point: SweepPoint;
  max: number;
  active: boolean;
}) {
  const width = Math.max(2, Math.min(100, (point.rouge1 / max) * 100));
  return (
    <div className="grid grid-cols-[70px_1fr_74px_66px] items-center gap-3 text-sm">
      <span className="font-mono text-xs text-[var(--muted)]">
        T={labelValue(point.value)}
      </span>
      <span className="h-2 overflow-hidden rounded-full bg-[var(--rule)]">
        <span
          className={active ? "block h-full bg-[var(--accent)]" : "block h-full bg-[var(--ink-weak)]"}
          style={{ width: `${width}%` }}
        />
      </span>
      <span className={active ? "font-mono text-[var(--accent)]" : "font-mono text-[var(--ink)]"}>
        {fmt(point.rouge1)}
      </span>
      <span className="font-mono text-xs text-[var(--muted)]">{pct(point.coverage)}</span>
    </div>
  );
}

function ThresholdFigure({ dataset, method }: { dataset: Dataset; method: MethodId }) {
  const block = thresholdSweep(dataset, method);
  if (!block) return null;
  const max = sweep.dataset_meta?.[dataset]?.maxBar ?? 0.4;
  const def = defaultPoint(block);
  const best = bestPoint(block);
  return (
    <section className="figure-panel">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="figure-kicker">{sweep.dataset_meta[dataset].label}</p>
          <h3 className="figure-title">{methodName(method)}</h3>
        </div>
        <div className="text-right text-xs text-[var(--muted)]">
          <div>default T={labelValue(block.default)}</div>
          <div>best T={labelValue(block.best)}</div>
        </div>
      </div>
      <div className="space-y-2">
        {block.points.map((point) => (
          <FigureBar
            key={labelValue(point.value)}
            point={point}
            max={max}
            active={point.is_default || point.is_best}
          />
        ))}
      </div>
      <p className="mt-4 text-xs leading-5 text-[var(--muted)]">
        {def && best
          ? `Default R1 ${fmt(def.rouge1)} at ${pct(def.coverage)} coverage; best R1 ${fmt(best.rouge1)} at ${pct(best.coverage)} coverage.`
          : "Default or best point is not available for this series."}
      </p>
    </section>
  );
}

function BaselineTable() {
  return (
    <div className="overflow-x-auto">
      <table className="paper-table">
        <thead>
          <tr>
            <th>Dataset</th>
            <th>Method</th>
            <th>ROUGE-1</th>
            <th>ROUGE-2</th>
            <th>ROUGE-L</th>
            <th>Note</th>
          </tr>
        </thead>
        <tbody>
          {DATASETS.flatMap((dataset) => {
            const winner = bestBaseline(dataset);
            return METHOD_IDS.map((method) => {
              const macro = baselineMacro(dataset, method);
              return (
                <tr key={`${dataset}-${method}`}>
                  <td>{sweep.dataset_meta[dataset].label}</td>
                  <td>{methodName(method)}</td>
                  <td className="mono-cell">{fmt(macro?.rouge1)}</td>
                  <td className="mono-cell">{fmt(macro?.rouge2)}</td>
                  <td className="mono-cell">{fmt(macro?.rougeL)}</td>
                  <td>{winner.method === method ? "Best baseline on ROUGE-1" : ""}</td>
                </tr>
              );
            });
          })}
        </tbody>
      </table>
    </div>
  );
}

function OptimizedHasosTable() {
  const methods: MethodId[] = ["m2", "m3", "m4"];
  return (
    <div className="overflow-x-auto">
      <table className="paper-table">
        <thead>
          <tr>
            <th>Method</th>
            <th>Base T</th>
            <th>Base token budget</th>
            <th>Macro R1</th>
            <th>Macro R2</th>
            <th>Macro RL</th>
            <th>Coverage</th>
          </tr>
        </thead>
        <tbody>
          {methods.map((method) => {
            const threshold = recommendedPoint(thresholdSweep("hasos", method));
            const token = recommendedPoint(tokenSweep("hasos", method));
            return (
              <tr key={`hasos-optimized-${method}`}>
                <td>{methodName(method)}</td>
                <td className="mono-cell">{labelValue(threshold?.value)}</td>
                <td className="mono-cell">{labelValue(token?.value)}</td>
                <td className="mono-cell">{fmt(token?.rouge1)}</td>
                <td className="mono-cell">{fmt(token?.rouge2)}</td>
                <td className="mono-cell">{fmt(token?.rougeL)}</td>
                <td className="mono-cell">{pct(token?.coverage)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function OptimalityTable() {
  return (
    <div className="overflow-x-auto">
      <table className="paper-table">
        <thead>
          <tr>
            <th>Dataset</th>
            <th>Method</th>
            <th>Default T</th>
            <th>Best T</th>
            <th>Default R1</th>
            <th>Best R1</th>
            <th>Coverage</th>
            <th>Conclusion</th>
          </tr>
        </thead>
        <tbody>
          {DATASETS.flatMap((dataset) =>
            (["m2", "m3", "m4"] as MethodId[]).map((method) => {
              const block = thresholdSweep(dataset, method);
              const def = defaultPoint(block);
              const best = bestPoint(block);
              return (
                <tr key={`${dataset}-${method}`}>
                  <td>{sweep.dataset_meta[dataset].label}</td>
                  <td>{methodLabel(method)}</td>
                  <td className="mono-cell">{labelValue(block?.default)}</td>
                  <td className="mono-cell">{labelValue(block?.best)}</td>
                  <td className="mono-cell">{fmt(def?.rouge1)}</td>
                  <td className="mono-cell">{fmt(best?.rouge1)}</td>
                  <td className="mono-cell">{pct(def?.coverage)}</td>
                  <td>
                    {block?.verdict?.status === "default_optimal"
                      ? "Default is optimal in the tested grid"
                      : block?.verdict?.status === "switch"
                        ? "Switch to best value"
                        : "Keep default: lower-coverage artifact"}
                  </td>
                </tr>
              );
            }),
          )}
        </tbody>
      </table>
    </div>
  );
}

export function ResearchPaperView() {
  const spaceWinner = bestBaseline("space");
  const hasosWinner = bestBaseline("hasos");
  return (
    <article className="paper">
      <header className="paper-hero">
        <div className="paper-label">Empirical report</div>
        <h1>Aspect-Based Hotel Review Summarization with Fixed-Denominator Evaluation</h1>
        <p className="paper-subtitle">
          A compact research view of the SPACE and HASOS experiments, including method
          comparison, evidence-threshold sweeps, coverage checks, and the current
          hyperparameter decision.
        </p>
        <div className="paper-meta">
          <span>Generated from committed JSON artifacts</span>
          <span>{new Date(sweep.generated_at).toLocaleString("en-US", { timeZone: "UTC" })} UTC</span>
          <span>{sweep.decision_metric}</span>
        </div>
      </header>

      <section className="paper-section paper-abstract">
        <h2>Abstract</h2>
        <p>
          The project compares four summarization variants for hotel-review aspect
          summaries: extractive SemAE evidence, flat abstractive rewriting, keyword
          sentiment splitting, and BERT-ABSA sentiment splitting. The latest sweep
          verifies method-specific HASOS settings rather than a single inherited
          default. The updated HASOS base uses M2 T=0.0075 with B=128, M3 T=0.0055
          with B=96, and M4 T=0.005 with B=96. Coverage is reported beside ROUGE
          because stricter thresholds can otherwise look cleaner by answering fewer
          aspect-entity instances.
        </p>
      </section>

      <section className="paper-section">
        <h2>1. Experimental Setup</h2>
        <div className="paper-columns">
          <p>
            SPACE is evaluated over six flat hotel aspects. HASOS is evaluated over
            four parent aspects aggregated from the hotel taxonomy. ROUGE is computed
            with the same official driver used by the baseline reports.
          </p>
          <p>
            Threshold sweeps use a fixed denominator: every gold-bearing entity stays
            in the average, and a missing generated summary contributes ROUGE 0. This
            makes the threshold study conservative and exposes evidence starvation via
            the coverage column.
          </p>
        </div>
      </section>

      <section className="paper-section">
        <h2>2. Main Results</h2>
        <p className="section-lede">
          Baseline ROUGE-1 winners are {methodLabel(spaceWinner.method)} on SPACE
          ({fmt(spaceWinner.rouge1)}) and {methodLabel(hasosWinner.method)} on HASOS
          ({fmt(hasosWinner.rouge1)}). The table below reports macro scores on the
          all split.
        </p>
        <BaselineTable />
      </section>

      <section className="paper-section">
        <h2>3. Optimized HASOS Base</h2>
        <p className="section-lede">
          The web base below uses the completed exact-threshold and token-budget
          sweeps. Scores are from the token-budget cell at the selected threshold,
          so this is the current recommended HASOS configuration, not the old
          shared default.
        </p>
        <OptimizedHasosTable />
      </section>

      <section className="paper-section">
        <h2>4. Threshold Optimality</h2>
        <p className="section-lede">
          The threshold grid now includes exact HASOS cells above 0.005. SPACE keeps
          T=0.0082; HASOS switches to M2 T=0.0075 and M3 T=0.0055, while M4 keeps
          T=0.005.
        </p>
        <OptimalityTable />
        <div className="figure-grid">
          <ThresholdFigure dataset="space" method="m2" />
          <ThresholdFigure dataset="space" method="m3" />
          <ThresholdFigure dataset="space" method="m4" />
          <ThresholdFigure dataset="hasos" method="m2" />
          <ThresholdFigure dataset="hasos" method="m3" />
          <ThresholdFigure dataset="hasos" method="m4" />
        </div>
      </section>

      <section className="paper-section">
        <h2>5. Interpretation</h2>
        <div className="finding-list">
          <p>
            <strong>Thresholds are not arbitrary.</strong> HASOS needed exact higher-T
            runs: M2 and M3 improve over the old shared threshold, while M4 stays at
            the old default.
          </p>
          <p>
            <strong>HASOS M2 needs hierarchical scoring.</strong> The M2 HASOS cells
            are synthesized with the parent pass and scored through the parent output
            directory, aligning system keys with FACILITY, AMENITY, SERVICE, and
            EXPERIENCE.
          </p>
          <p>
            <strong>Token budget is now part of the base.</strong> M2 prefers 128 new
            tokens, while M3 and M4 prefer 96. The old 192-token setting is no longer
            the HASOS base for abstractive methods.
          </p>
        </div>
      </section>
    </article>
  );
}
